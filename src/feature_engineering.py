from __future__ import annotations
import numpy as np
import pandas as pd

# Single source of truth for the env's observation columns and the signal probe's feature
# set (HANDOFF.md Branch E Step 0: these two lists were hand-duplicated between
# src/env/trading_env.py and scripts/signal_probe.py, which is exactly the kind of thing
# that silently drifts -- both now import this instead).
STATIONARY_FEATURE_COLUMNS = [
    "LogReturn", "VolLogDiff", "RelRange", "RelOpen", "RelMACD", "RSI_Centered",
    "RelATR", "BB_Width", "BB_Upper_Dist", "BB_Lower_Dist", "SMA_Trend",
    "RelVWAP", "MACD_Signal_Rel", "MACD_Hist_Rel",
    "HourSin", "HourCos", "DowSin", "DowCos",
    "VolRatio_6_48", "RelVolumeByHourOfWeek",
    "EthBtcRelReturn",
]


def compute_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stationary technical-indicator features from normalized OHLCV data.

    Ported unchanged from the sibling stock bot (src/feature_engineering.py) — this
    function is market-agnostic, it just needs the same normalized-column contract.

    IMPORTANT — input contract:
    This function receives the output of build_normalized_frame() in src/market_data.py, where:
        df["Open"]   = OpenNorm  = (RawOpen / PrevRawClose) - 1   (a return, ~0.005)
        df["High"]   = HighNorm  = (RawHigh / RawClose) - 1        (a ratio deviation, ~0.02)
        df["Low"]    = LowNorm   = (RawLow  / RawClose) - 1        (a ratio deviation, ~-0.01)
        df["Volume"] = VolumeNorm = ln(V_t+1) - ln(V_{t-1}+1)     (already a log-diff)
        df["RawClose"] = raw closing price in quote currency (used as price anchor)

    Raw H/L are reconstructed from normalized columns:
        RawHigh = RawClose * (1 + HighNorm)
        RawLow  = RawClose * (1 + LowNorm)

    Features included (14 total):
        1.  LogReturn       - ln(Close_t / Close_{t-1})
        2.  VolLogDiff      - VolumeNorm passed through directly (already log-diff)
        3.  RelRange        - (RawHigh - RawLow) / RawClose
        4.  RelOpen         - OpenNorm passed through directly (already a gap return)
        5.  RelMACD         - (EMA12 - EMA26) / RawClose
        6.  RSI_Centered    - (RSI - 50) / 50
        7.  RelATR          - ATR(14) / RawClose, using reconstructed raw H/L
        8.  BB_Width        - 4 * std20 / sma20
        9.  BB_Upper_Dist   - (Upper Band - Close) / Close
        10. BB_Lower_Dist   - (Close - Lower Band) / Close
        11. SMA_Trend       - +1.0 / -1.0 / 0.0 crossover signal
        12. RelVWAP         - (Close - VWAP20) / VWAP20, using reconstructed raw H/L
        13. MACD_Signal_Rel - Signal line / RawClose
        14. MACD_Hist_Rel   - MACD histogram / RawClose

    Returns:
        DataFrame with 14 stationary feature columns (+ Date if present).
    """
    feat = pd.DataFrame(index=df.index)
    if "Date" in df.columns:
        feat["Date"] = df["Date"]

    close_col = "RawClose" if "RawClose" in df.columns else "Close"
    close = df[close_col]

    raw_high = close * (1.0 + df["High"])
    raw_low = close * (1.0 + df["Low"])

    volume_norm = df["Volume"]

    feat["LogReturn"] = np.log(close / close.shift(1)).fillna(0.0)
    feat["VolLogDiff"] = volume_norm.fillna(0.0)
    feat["RelRange"] = ((raw_high - raw_low) / close).fillna(0.0)
    feat["RelOpen"] = df["Open"].fillna(0.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    feat["RelMACD"] = ((ema12 - ema26) / close).fillna(0.0)

    def _compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    feat["RSI_Centered"] = (_compute_rsi(close) - 50.0) / 50.0

    prev_close = close.shift(1)
    tr1 = raw_high - raw_low
    tr2 = (raw_high - prev_close).abs()
    tr3 = (raw_low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    feat["RelATR"] = (atr / close).fillna(0.0)

    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    feat["BB_Width"] = (4.0 * std20 / sma20).fillna(0.0)
    feat["BB_Upper_Dist"] = ((sma20 + 2.0 * std20) - close) / close
    feat["BB_Lower_Dist"] = (close - (sma20 - 2.0 * std20)) / close

    sma50 = close.rolling(window=50).mean()
    feat["SMA_Trend"] = np.where(sma20 > sma50, 1.0, -1.0)
    feat["SMA_Trend"] = np.where(sma20.isna() | sma50.isna(), 0.0, feat["SMA_Trend"])

    typical_price = (raw_high + raw_low + close) / 3.0
    tp_v = typical_price * volume_norm.clip(lower=0.0)  # clip: log-diff can be negative
    rolling_tp_v = tp_v.rolling(window=20).sum()
    rolling_vol = volume_norm.clip(lower=0.0).rolling(window=20).sum()
    vwap = (rolling_tp_v / (rolling_vol + 1e-9)).fillna(close)
    feat["RelVWAP"] = ((close - vwap) / (vwap + 1e-9)).fillna(0.0)

    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    feat["MACD_Signal_Rel"] = (signal_line / close).fillna(0.0)
    feat["MACD_Hist_Rel"] = (macd_hist / close).fillna(0.0)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feat


def compute_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hour-of-day and day-of-week cyclical features for the 24/7 crypto calendar.

    CLAUDE.md: "Keep hour-of-day + day-of-week cyclical features (real intraday/weekly
    seasonality: Asia/US session flows, weekend liquidity thinning). Drop all
    market-session/open-close features." There is no session/open-close concept to drop
    in this codebase (crypto has none, and the ported stock indicators never encoded one),
    so this function only needs to *add* the two cyclical pairs.

    Sin/cos encoding avoids the discontinuity of a raw 0-23 / 0-6 integer at the wraparound
    boundary (hour 23 -> 0, Sunday -> Monday), which would otherwise look like a huge jump
    to the policy network.

    Requires a `Date` column (UTC, tz-naive). Returns a 4-column frame:
        HourSin, HourCos, DowSin, DowCos
    """
    if "Date" not in df.columns:
        raise ValueError("compute_cyclical_time_features requires a Date column.")

    feat = pd.DataFrame(index=df.index)
    dates = pd.to_datetime(df["Date"])

    hour = dates.dt.hour.astype(float)
    dow = dates.dt.dayofweek.astype(float)  # Monday=0 ... Sunday=6

    feat["HourSin"] = np.sin(2.0 * np.pi * hour / 24.0)
    feat["HourCos"] = np.cos(2.0 * np.pi * hour / 24.0)
    feat["DowSin"] = np.sin(2.0 * np.pi * dow / 7.0)
    feat["DowCos"] = np.cos(2.0 * np.pi * dow / 7.0)

    return feat


def compute_vol_and_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HANDOFF.md Branch E, Phase E0 (2026-07-27): candidate features probed after
    scripts/signal_probe.py found no confirmed edge in the original 14 ported indicators +
    4 cyclical columns. Kept separate from compute_stationary_features so that function's
    "ported unchanged from the stock bot" provenance stays honest -- these are new,
    crypto-batch-specific, not part of the port.

    Requires the output of build_normalized_frame() (needs RawClose, RawVolume, Date).

    VolRatio_6_48: short-window (6-bar) realized vol / longer-window (48-bar) realized vol.
    A vol-regime term-structure ratio -- crypto vol is known to cluster and this set has
    nothing that captures regime shifts directly (BB_Width is level, not ratio-of-windows).

    RelVolumeByHourOfWeek: current bar's volume relative to the trailing historical average
    volume for the same (day-of-week, hour-of-day) bucket. Operationalizes CLAUDE.md's
    "weekend liquidity thinning / Asia-US session flow" hypothesis, which until now only had
    time-of-day encoding (HourSin/Cos, DowSin/Cos) with nothing tying it to volume.
    Strictly trailing: shift(1) before the expanding mean, so a bar never sees its own volume
    in its own bucket average -- the one place a naive version of this would leak.
    """
    close_col = "RawClose" if "RawClose" in df.columns else "Close"
    close = df[close_col]

    feat = pd.DataFrame(index=df.index)
    if "Date" in df.columns:
        feat["Date"] = df["Date"]

    log_return = np.log(close / close.shift(1))
    vol_short = log_return.rolling(window=6).std()
    vol_long = log_return.rolling(window=48).std()
    feat["VolRatio_6_48"] = (vol_short / (vol_long + 1e-9)).fillna(0.0)

    if "RawVolume" in df.columns and "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
        bucket = dates.dt.dayofweek.astype(str) + "_" + dates.dt.hour.astype(str)
        volume = df["RawVolume"]
        trailing_avg = volume.groupby(bucket).transform(lambda s: s.shift(1).expanding().mean())
        feat["RelVolumeByHourOfWeek"] = (volume / (trailing_avg + 1e-9)).fillna(1.0)
    else:
        feat["RelVolumeByHourOfWeek"] = 1.0

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat
