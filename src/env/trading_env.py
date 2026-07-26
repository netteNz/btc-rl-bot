"""
CryptoTradingEnv — Binary PPO long/flat trading environment for spot crypto.

Ported from the sibling stock bot's TradingEnv (reinforcement-learning-stocks/src/trading_env.py).
The class is intentionally *not forked* into divergent stock/crypto implementations — per
CLAUDE.md ("24/7 calendar handling — parameterize the existing env; do not fork it"), the
mechanics below are already market-agnostic:

- The env steps through whatever rows are in `df` bar-by-bar. It has no concept of sessions,
  opens/closes, or trading-day boundaries, so 24/7 crypto data "just works" without changes.
- The fee model (`transaction_cost_rate`, `trade_penalty`, `spread_bps`, `slippage_bps`) is
  already a generic per-notional/flat-fee model. Crypto callers pass taker/maker economics
  here (see src/experiments.py defaults); stock callers passed commission-era assumptions.
  There is no separate "crypto fee model" code path — only different config defaults.

The only crypto-specific deltas live in the *caller*: src/market_data.py builds a 24/7 feature
frame (with hour-of-day / day-of-week cyclical features instead of session features), and
src/experiments.py benchmarks alpha against buy-and-hold BTC instead of QQQ.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


LEADERBOARD_VERSION = 1

class PositionManager:
    """
    Handles portfolio math, position sizing, and transaction costs.

    Position size (`shares_held`) is a fractional quantity of the underlying asset, not a
    whole-unit count. Ported from the stock bot, whole-share flooring (`int(value // price)`)
    was survivable there because equity prices are small relative to typical account balances.
    On BTC-USD (observed range ~$16.5K-$126K in this repo's backfill) it is not: with the
    $1,000 class default, floor(1000 / price) is 0 for every single bar in the dataset --
    the agent is structurally incapable of ever entering a position, independent of reward
    shaping. Coinbase spot supports ~1e-8 BTC precision, so fractional sizing is both the
    crypto-native fix and the one that works at any balance/price ratio, rather than requiring
    a balance large enough to afford a whole coin (which then makes 1-unit granularity too
    coarse against max_weight_delta_per_step).
    """
    def __init__(self, initial_balance, transaction_cost_rate, trade_penalty, spread_bps=0.0,
                 slippage_bps=0.0, min_trade_notional=1.0):
        self.initial_balance = float(initial_balance)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.trade_penalty = float(trade_penalty)
        self.spread_bps = max(float(spread_bps), 0.0)
        self.slippage_bps = max(float(slippage_bps), 0.0)
        # Coinbase's practical minimum order size is ~$1 notional. Below this, skip execution
        # entirely rather than firing a trade -- this also absorbs the tiny geometric
        # fee-on-fee rebalancing residual that fractional weight-targeting produces after
        # entry (target_value = target_weight * net_worth recomputes every bar, so a fee-
        # induced cash residual re-triggers a shrinking correction trade for a few bars unless
        # floored out here). A full close to flat is exempt -- see step().
        self.min_trade_notional = max(float(min_trade_notional), 0.0)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.current_weight = 0.0
        self.time_in_position = 0
        self.entry_price = 0.0

    def step(self, target_weight, current_price):
        target_weight = float(np.clip(target_weight, -1.0, 1.0))

        # Debounce: Do not incur fees for micro-adjustments smaller than 5% portfolio shift
        if abs(target_weight - self.current_weight) < 0.05:
            target_weight = self.current_weight

        # Only re-derive the position size from current net worth when the weight target has
        # actually changed. A genuine hold (target == currently realized weight) must leave
        # shares_held untouched -- otherwise fractional sizing recomputes target_shares from
        # net_worth every single bar, and a nonzero cash residual left over from a prior fee
        # (fees are additive on top of notional, not deducted from it) drifts in and out of a
        # tradeable notional as price moves, firing "trades" for a decision that never changed.
        # Whole-share flooring hid this by accident (quantization too coarse to notice); it is
        # real under fractional sizing and must be blocked here, not papered over with a bigger
        # min_trade_notional floor.
        if target_weight == self.current_weight and self.shares_held != 0.0:
            target_shares = self.shares_held
        else:
            target_value = target_weight * self.net_worth
            target_shares = target_value / current_price
        delta_shares = target_shares - self.shares_held

        # A full close to flat always executes regardless of size, so the notional floor can
        # never leave the agent stuck holding a position it wants to exit.
        is_full_close = target_shares == 0.0 and self.shares_held != 0.0
        candidate_notional = abs(delta_shares) * current_price
        trade_executed = delta_shares != 0.0 and (is_full_close or candidate_notional >= self.min_trade_notional)
        fee = 0.0
        execution_price = current_price
        gross_value = 0.0

        trade_penalty_paid = 0.0

        if trade_executed:
            side = 1.0 if delta_shares > 0 else -1.0
            spread_component = (self.spread_bps * 1e-4) * 0.5
            slippage_component = self.slippage_bps * 1e-4
            impact = spread_component + slippage_component
            execution_price = max(current_price * (1.0 + (side * impact)), 1e-8)

            gross_value = abs(delta_shares) * execution_price
            fee = gross_value * self.transaction_cost_rate

            self.balance -= (delta_shares * execution_price) + fee
            self.shares_held = target_shares

            if self.trade_penalty > 0:
                trade_penalty_paid = float(self.trade_penalty)
                self.balance -= trade_penalty_paid

            if target_weight != 0 and self.current_weight == 0:
                self.entry_price = current_price
                self.time_in_position = 0

        if self.shares_held == 0:
            self.current_weight = 0.0
            self.entry_price = 0.0
            self.time_in_position = 0
        else:
            # Update current_weight based on the actual target weight chosen
            self.current_weight = target_weight
            self.time_in_position += 1

        prev_net_worth = self.net_worth
        # Net worth is cash balance + market value of shares
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)

        portfolio_return = (self.net_worth / max(prev_net_worth, 1e-8)) - 1.0
        drawdown = (self.peak_net_worth - self.net_worth) / max(self.peak_net_worth, 1e-8)

        # unrealized_pnl_pct for the observation state
        unrealized_pnl_pct = 0.0
        if self.entry_price > 0:
            ratio = current_price / self.entry_price
            unrealized_pnl_pct = ratio - 1.0 if self.shares_held > 0 else 1.0 - ratio

        return trade_executed, portfolio_return, drawdown, unrealized_pnl_pct, execution_price, fee, trade_penalty_paid, gross_value


class RewardEvaluator:
    """Strategy pattern for varying reward schemas (Legacy, Sharpe, Sortino, Hybrid)."""
    def __init__(
        self,
        mode,
        rolling_window,
        epsilon,
        return_scale,
        pnl_scale,
        dir_scale,
        hold_scale,
        dd_scale,
        action_scale,
        turnover_scale,
        clip,
        sharpe_scale=0.0,
    ):
        self.mode = mode.lower()
        self.epsilon = epsilon
        self.return_scale = return_scale
        self.pnl_scale = pnl_scale
        self.dir_scale = dir_scale
        self.hold_scale = hold_scale
        self.dd_scale = dd_scale
        self.action_scale = action_scale
        self.turnover_scale = turnover_scale
        self.clip = clip
        self.sharpe_scale = sharpe_scale  # Sharpe as secondary regularizer
        self.returns_buffer = deque(maxlen=rolling_window)

    def calculate(self, portfolio_return, realized_return, exposure_weight, weight_change, trade_executed, drawdown):
        strategy_realized_return = exposure_weight * realized_return
        self.returns_buffer.append(strategy_realized_return)

        pnl_reward = self._pnl_reward(portfolio_return)
        hold_penalty = -self.hold_scale * abs(realized_return) if abs(exposure_weight) < 0.1 else 0.0
        action_bonus = self.action_scale if trade_executed else 0.0
        turnover_penalty = -self.turnover_scale * abs(weight_change)
        dd_penalty = -self.dd_scale * drawdown

        directional_reward = strategy_realized_return

        base_reward = 0.0
        risk_metric = 0.0

        if self.mode == "sharpe":
            risk_metric = self._sharpe()
            base_reward = self.return_scale * risk_metric
        elif self.mode == "sortino":
            risk_metric = self._sortino()
            base_reward = self.return_scale * risk_metric
        else:  # legacy
            base_reward = self.return_scale * portfolio_return

        # Directional shaping and penalties apply across all reward modes
        # so hyperparameters behave consistently in legacy/sharpe/sortino experiments.
        total = base_reward + pnl_reward + (self.dir_scale * directional_reward) + hold_penalty + action_bonus + turnover_penalty + dd_penalty
        return (
            float(np.clip(total, -self.clip, self.clip)),
            risk_metric,
            strategy_realized_return,
            directional_reward,
            pnl_reward,
            hold_penalty,
            action_bonus,
            turnover_penalty,
            dd_penalty,
        )

    def _pnl_reward(self, portfolio_return):
        return self.pnl_scale * float(portfolio_return)

    def _sharpe(self):
        if len(self.returns_buffer) < 2: return 0.0
        rets = np.array(self.returns_buffer)
        std = np.std(rets)
        return np.mean(rets) / std if std > self.epsilon else 0.0

    def _sortino(self):
        if len(self.returns_buffer) < 2: return 0.0
        rets = np.array(self.returns_buffer)
        downside = rets[rets < 0]
        mean = np.mean(rets)
        if len(downside) < 1: return mean / self.epsilon if mean > 0 else 0.0
        d_std = np.std(downside)
        return mean / d_std if d_std > self.epsilon else (mean / self.epsilon if mean > 0 else 0.0)

    def reset(self):
        self.returns_buffer.clear()


class CryptoTradingEnv(gym.Env):
    """
    Binary-first trading environment for spot crypto (also supports continuous sizing,
    retained generically even though CLAUDE.md's architecture decision is binary-only
    for this project — see module docstring).

    min_hold_bars: When > 0, enforces a minimum holding period between position flips.
    After any executed trade, further execution is blocked for min_hold_bars steps.
    The agent's desired action continues to be recorded normally; only execution is
    deferred. Default=0 disables the constraint (full backward compatibility).

    Crypto note: min_hold_bars is not just a training-stability knob here — it is the
    fee-floor control. At 1h bars, taker/taker round trips cost ~1.2% of notional, so
    min_hold_bars must be >= 6 (or use 4h bars with min_hold_bars >= 3) or expected
    edge per trade cannot clear the fee floor. See CLAUDE.md "Fees dominate".
    """
    def __init__(
        self,
        df,
        initial_balance=1000,
        include_position_in_observation=True,
        transaction_cost_rate=0.0,
        trade_penalty=0.0,
        min_trade_notional=1.0,
        reward_return_scale=1.0,
        reward_pnl_scale=0.0,
        reward_direction_scale=0.40,
        reward_hold_penalty_scale=0.10,
        reward_drawdown_penalty_scale=0.10,
        reward_action_bonus_scale=0.02,
        reward_turnover_penalty_scale=0.05,
        reward_clip=1.0,
        reward_ignore_transaction_cost=False,
        execution_mode="next_bar",
        spread_bps=0.0,
        slippage_bps=0.0,
        max_weight_delta_per_step=0.0,
        market_feature_columns=None,
        reward_mode="legacy",
        rolling_reward_window=100,
        reward_epsilon=1e-6,
        reward_sharpe_scale=0.0,
        long_only=False,
        binary_actions=False,
        max_episode_steps=0,
        random_start=False,
        min_hold_bars=0,
        use_cooldown_obs=False,
    ):
        super(CryptoTradingEnv, self).__init__()
        self.df = df
        self.include_position = bool(include_position_in_observation)
        self.current_step = 0
        self.start_step = 0
        self.max_episode_steps = int(max_episode_steps)
        self.random_start = bool(random_start)
        self.buy_hold_shares = 0.0
        self.price_column = 'RawClose' if 'RawClose' in self.df.columns else 'Close'
        self.execution_mode = str(execution_mode).lower()
        self.long_only = bool(long_only)
        self.binary_actions = bool(binary_actions)
        self.reward_ignore_transaction_cost = bool(reward_ignore_transaction_cost)
        self.max_weight_delta_per_step = float(max_weight_delta_per_step)
        self.min_hold_bars = int(min_hold_bars)
        self.use_cooldown_obs = bool(use_cooldown_obs)
        if self.max_weight_delta_per_step < 0.0:
            raise ValueError("max_weight_delta_per_step must be >= 0.0")
        if self.max_weight_delta_per_step > 2.0:
            raise ValueError("max_weight_delta_per_step must be <= 2.0")
        if self.min_hold_bars < 0:
            raise ValueError("min_hold_bars must be >= 0")
        if self.execution_mode not in {"same_bar", "next_bar"}:
            raise ValueError("execution_mode must be one of: same_bar, next_bar")
        self.pending_target_weight = 0.0
        # Cooldown counter: tracks bars elapsed since the last executed trade.
        # Initialized to min_hold_bars so bar 0 is freely tradeable.
        self.bars_since_last_trade = self.min_hold_bars

        # OOP Subsystems
        self.pm = PositionManager(
            initial_balance,
            transaction_cost_rate,
            trade_penalty,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            min_trade_notional=min_trade_notional,
        )
        self.re = RewardEvaluator(
            reward_mode, rolling_reward_window, reward_epsilon,
            reward_return_scale, reward_pnl_scale, reward_direction_scale, reward_hold_penalty_scale,
            reward_drawdown_penalty_scale, reward_action_bonus_scale, reward_turnover_penalty_scale, reward_clip,
            sharpe_scale=reward_sharpe_scale
        )

        # Action Space: Discrete(2) for binary_actions mode, else continuous Box
        # binary_actions=True -> Discrete(2): 0=flat/cash, 1=full long (the only spot-legal shape)
        # binary_actions=False -> Box(-1, 1): continuous fractional sizing (not used for crypto spot)
        if self.binary_actions:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space Configuration
        if market_feature_columns is None:
            if all(col in self.df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
                self.market_feature_columns = ["Open", "High", "Low", "Close", "Volume"]
            else:
                potential_cols = [
                    "LogReturn", "VolLogDiff", "RelRange", "RelOpen", "RelMACD", "RSI_Centered",
                    "RelATR", "BB_Width", "BB_Upper_Dist", "BB_Lower_Dist", "SMA_Trend",
                    "RelVWAP", "MACD_Signal_Rel", "MACD_Hist_Rel",
                    "HourSin", "HourCos", "DowSin", "DowCos",
                ]
                self.market_feature_columns = [col for col in potential_cols if col in self.df.columns]
        else:
            self.market_feature_columns = market_feature_columns

        # No news/sentiment features for crypto (see CLAUDE.md — not in scope).
        self.active_news_columns = []

        # Base market + [balance, shares_held]
        state_count = 2

        # + [current_weight, unrealized_pnl, time_in_position]
        if self.include_position:
            state_count += 3

        if self.use_cooldown_obs:
            state_count += 1

        observation_size = len(self.market_feature_columns) + len(self.active_news_columns) + state_count
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32)

    def _apply_max_weight_delta(self, current_weight, desired_weight):
        """Cap exposure changes per step when realism gating is enabled."""
        desired_weight = float(np.clip(desired_weight, -1.0, 1.0))
        if self.max_weight_delta_per_step <= 0.0:
            return desired_weight, False

        lower = max(-1.0, float(current_weight) - self.max_weight_delta_per_step)
        upper = min(1.0, float(current_weight) + self.max_weight_delta_per_step)
        capped_weight = float(np.clip(desired_weight, lower, upper))
        was_limited = abs(capped_weight - desired_weight) > 1e-12
        return capped_weight, was_limited

    def _get_obs(self, unrealized_pnl=0.0):
        row = self.df.loc[self.current_step]
        market_values = [float(row[col]) for col in self.market_feature_columns]
        news_values = [float(row[col]) for col in self.active_news_columns]

        if self.use_cooldown_obs:
            current_price = max(float(row[self.price_column]), 1e-8)
            norm_balance = self.pm.balance / self.pm.initial_balance
            norm_shares = (self.pm.shares_held * current_price) / self.pm.initial_balance
            account_state = [norm_balance, norm_shares]
        else:
            account_state = [self.pm.balance, self.pm.shares_held]

        if self.include_position:
            account_state.extend([self.pm.current_weight, unrealized_pnl, self.pm.time_in_position])

        if self.use_cooldown_obs:
            cooldown_active = self.min_hold_bars > 0 and self.bars_since_last_trade < self.min_hold_bars
            account_state.append(1.0 if cooldown_active else 0.0)

        return np.array(market_values + news_values + account_state, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean array indicating which actions are valid.
        Used by sb3_contrib.MaskablePPO for discrete action spaces.
        """
        if not self.binary_actions:
            return np.array([True, True])

        cooldown_active = self.min_hold_bars > 0 and self.bars_since_last_trade < self.min_hold_bars
        if cooldown_active:
            # If cooldown is active, the agent cannot trade and must stick to the current position.
            current_is_long = self.pm.current_weight > 0.5
            if current_is_long:
                return np.array([False, True])  # Only Action 1 is valid
            else:
                return np.array([True, False])  # Only Action 0 is valid
        else:
            return np.array([True, True])  # Both actions are valid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start and self.max_episode_steps > 0 and len(self.df) > self.max_episode_steps:
            self.start_step = np.random.randint(0, len(self.df) - self.max_episode_steps)
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.pending_target_weight = 0.0
        self.pm.reset()
        self.re.reset()
        # Reset cooldown: start ready-to-trade (counter at min_hold_bars means no cooldown active)
        self.bars_since_last_trade = self.min_hold_bars

        current_price = max(float(self.df.loc[self.current_step, self.price_column]), 1e-8)
        self.buy_hold_shares = self.pm.initial_balance / current_price

        return self._get_obs(), {}

    @property
    def net_worth(self):
        # Expose net_worth for backwards compatibility scripts reading the env
        return self.pm.net_worth

    @property
    def position(self):
        # Translate continuous to discrete notation for backwards compatibility with downstream analytics
        if self.pm.current_weight > 0.1: return 1
        if self.pm.current_weight < -0.1: return 2
        return 0

    def step(self, action):
        # Accept scalar, 0-d ndarray, or 1-d action arrays from different callers/SB3 wrappers.
        action_array = np.asarray(action)
        if action_array.ndim == 0:
            raw_action = action_array.item()
        else:
            raw_action = action_array.reshape(-1)[0]

        if self.binary_actions:
            # Discrete(2) space: 0=flat/cash, 1=full long.
            discrete_action = int(round(float(raw_action)))
            discrete_action = max(0, min(1, discrete_action))
            desired_target_weight = 1.0 if discrete_action == 1 else 0.0
        elif isinstance(raw_action, (int, np.integer)) and raw_action in (0, 1, 2):
            # Backward-compat for legacy discrete policies:
            # 0=Hold, 1=Buy(Long), 2=Sell(Short) -> map into continuous target weights.
            desired_target_weight = {0: 0.0, 1: 1.0, 2: -1.0}[int(raw_action)]
        else:
            desired_target_weight = float(raw_action)
            if self.long_only:
                desired_target_weight = max(0.0, desired_target_weight)

        if self.execution_mode == "next_bar":
            execution_target_weight = self.pending_target_weight
            self.pending_target_weight = desired_target_weight
        else:
            execution_target_weight = desired_target_weight

        execution_target_weight_pre_cap = float(execution_target_weight)

        current_price = max(float(self.df.loc[self.current_step, self.price_column]), 1e-8)
        pre_trade_weight = float(self.pm.current_weight)
        execution_target_weight, weight_delta_limited = self._apply_max_weight_delta(
            pre_trade_weight,
            execution_target_weight,
        )

        # Minimum hold period: if cooldown is active, lock execution to the current position.
        cooldown_active = self.min_hold_bars > 0 and self.bars_since_last_trade < self.min_hold_bars
        if cooldown_active:
            execution_target_weight = pre_trade_weight

        prev_net_worth = float(self.pm.net_worth)

        prev_step = max(0, self.current_step - 1)
        prev_price = max(float(self.df.loc[prev_step, self.price_column]), 1e-8)
        realized_return = (current_price / prev_price) - 1.0 if self.current_step > 0 else 0.0

        # Execute trade via Position Manager
        trade_executed, portfolio_return, drawdown, unrealized_pnl, execution_price, execution_fee, trade_penalty_paid, execution_notional = self.pm.step(
            execution_target_weight,
            current_price,
        )

        # Update cooldown counter: reset on any executed trade, increment otherwise.
        if trade_executed:
            self.bars_since_last_trade = 0
        else:
            self.bars_since_last_trade = min(self.bars_since_last_trade + 1, self.min_hold_bars)

        # Reward attribution uses prior exposure for this bar's realized return.
        exposure_weight = pre_trade_weight
        weight_change = float(self.pm.current_weight - pre_trade_weight)

        if self.reward_ignore_transaction_cost:
            ignored_cost = float(execution_fee + trade_penalty_paid)
            cost_ratio = ignored_cost / max(prev_net_worth, 1e-8)
            reward_portfolio_return = float(portfolio_return + cost_ratio)
        else:
            reward_portfolio_return = float(portfolio_return)

        # Evaluate reward
        reward, risk_metric, strategy_realized_return, dir_rew, pnl_rew, hold_pen, action_bon, turnover_pen, dd_pen = self.re.calculate(
            reward_portfolio_return,
            realized_return,
            exposure_weight,
            weight_change,
            trade_executed,
            drawdown,
        )

        self.current_step += 1

        episode_ended = False
        if self.max_episode_steps > 0 and (self.current_step - self.start_step) >= self.max_episode_steps:
            episode_ended = True

        terminated = episode_ended or self.current_step >= len(self.df) - 1
        truncated = False

        if self.re.mode == "sparse":
            if terminated:
                current_price_for_bh = max(float(self.df.loc[min(self.current_step, len(self.df)-1), self.price_column]), 1e-8)
                buy_hold_equity = self.buy_hold_shares * current_price_for_bh
                reward = (self.pm.net_worth / max(buy_hold_equity, 1e-8)) - 1.0
            else:
                reward = 0.0

        info = {
            "reward_total": reward,
            "reward_portfolio_return": portfolio_return,
            "reward_direction": dir_rew,
            "reward_pnl": pnl_rew,
            "reward_hold_penalty": hold_pen,
            "reward_action_bonus": action_bon,
            "reward_turnover_penalty": turnover_pen,
            "reward_drawdown_penalty": dd_pen,
            "realized_return": realized_return,
            "reward_drawdown": drawdown,
            "reward_net_worth": self.pm.net_worth,
            "reward_mode": self.re.mode,
            "reward_risk_metric": risk_metric,
            "execution_mode": self.execution_mode,
            "max_weight_delta_per_step": self.max_weight_delta_per_step,
            "min_hold_bars": self.min_hold_bars,
            "cooldown_active": int(cooldown_active),
            "bars_since_last_trade": self.bars_since_last_trade,
            "execution_weight_delta_limited": int(weight_delta_limited),
            "execution_target_weight": execution_target_weight,
            "execution_target_weight_pre_cap": execution_target_weight_pre_cap,
            "desired_target_weight": desired_target_weight,
            "execution_price": execution_price,
            "execution_fee": execution_fee,
            "execution_trade_penalty": trade_penalty_paid,
            "execution_notional": execution_notional,
        }

        return self._get_obs(unrealized_pnl), reward, terminated, truncated, info
