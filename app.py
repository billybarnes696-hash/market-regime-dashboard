import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

try:
    from defeatbeta_api import Ticker
except Exception:
    from defeatbeta_api.data.ticker import Ticker

st.set_page_config(layout="wide", page_title="Diamond Scanner Pro")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "TSI_424", "TSI_424_pct", "TSI_424_slope_1", "TSI_747", "TSI_747_pct", "TSI_747_slope_3", "pinned_424",
    "CCI15", "CCI15_pct", "CCI15_days_fading", "CCI_divergence",
    "BBP", "BBP_pct", "BBP_delta_1", "BBP_days_falling",
    "EXT_VWAP", "VWAP_slope", "VWAP_stall",
    "stretch_ATR", "ATR14", "ATR_pct",
    "RSI14", "RSI14_slope_3", "RSI_divergence",
    "MFI14", "MFI14_slope", "MFI_divergence",
    "RS_SLOPE_5", "RS_SLOPE_10", "RS_divergence",
    "ADX14", "ADX14_slope_3",
    "state_424_code", "state_747_code", "state_1377_code",
    "bear_kiss_score", "bull_kiss_score", "is_bear_kiss", "is_bull_kiss",
    "is_hot", "is_cold", "is_fading_stack", "is_repairing_stack",
]

REGIME_COLS = [
    "mkt_above_ema20", "mkt_tsi_747", "mkt_tsi_state_code", "mkt_stretch_atr",
    "mkt_cci15", "mkt_hot", "mkt_fading", "high_vol_regime", "low_vol_regime",
    "leadership_regime", "risk_on_regime",
]

# Weighted feature-vector similarity model: higher-signal dimensions get more influence
# in the analog engine than lower-signal housekeeping fields.
FEATURE_WEIGHTS = {
    # momentum / structure
    "TSI_424": 1.20,
    "TSI_424_pct": 1.55,
    "TSI_424_slope_1": 1.45,
    "TSI_747": 1.10,
    "TSI_747_pct": 1.45,
    "TSI_747_slope_3": 1.35,
    "pinned_424": 1.25,
    # exhaustion / mean reversion
    "CCI15": 1.15,
    "CCI15_pct": 1.55,
    "CCI15_days_fading": 1.45,
    "CCI_divergence": 1.40,
    "BBP": 1.00,
    "BBP_pct": 1.35,
    "BBP_delta_1": 1.10,
    "BBP_days_falling": 1.10,
    "EXT_VWAP": 1.30,
    "VWAP_slope": 1.20,
    "VWAP_stall": 1.20,
    "stretch_ATR": 1.35,
    "ATR14": 1.00,
    "ATR_pct": 1.10,
    # participation / leadership
    "RSI14": 0.95,
    "RSI14_slope_3": 1.00,
    "RSI_divergence": 1.10,
    "MFI14": 1.10,
    "MFI14_slope": 1.15,
    "MFI_divergence": 1.20,
    "RS_SLOPE_5": 1.25,
    "RS_SLOPE_10": 1.10,
    "RS_divergence": 1.20,
    # trend / context
    "ADX14": 1.10,
    "ADX14_slope_3": 1.10,
    "state_424_code": 1.10,
    "state_747_code": 1.20,
    "state_1377_code": 1.05,
    # pattern state
    "bear_kiss_score": 1.45,
    "bull_kiss_score": 1.35,
    "is_bear_kiss": 1.30,
    "is_bull_kiss": 1.25,
    "is_hot": 1.05,
    "is_cold": 1.05,
    "is_fading_stack": 1.10,
    "is_repairing_stack": 1.10,
}

STATE_CODE_MAP = {
    "below_zero": 0,
    "bull_repair": 1,
    "bull_continuation": 2,
    "near_pinned": 3,
    "pinned": 4,
    "bear_kiss": 5,
    "rolling": 6,
    "oversold": 7,
}

for key, default in {
    "scan_results": None,
    "debug_rows": [],
    "detail_rows": {},
    "analogs_map": {},
    "live_rows_map": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def tsi(series: pd.Series, r: int, s: int, signal: int = 7) -> Tuple[pd.Series, pd.Series]:
    mom = series.diff()
    ema1 = mom.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    abs_mom = mom.abs()
    ema3 = abs_mom.ewm(span=r, adjust=False).mean()
    ema4 = ema3.ewm(span=s, adjust=False).mean()
    tsi_val = 100 * ema2 / ema4.replace(0, np.nan)
    tsi_sig = tsi_val.ewm(span=signal, adjust=False).mean()
    return tsi_val, tsi_sig


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def bbpct(close: pd.Series, n: int = 20) -> pd.Series:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def rolling_vwap(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    num = (tp * vol).rolling(lookback, min_periods=max(5, lookback // 2)).sum()
    den = vol.rolling(lookback, min_periods=max(5, lookback // 2)).sum()
    return num / den


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rmf = tp * df["volume"].fillna(0)
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(n).sum()
    neg_sum = neg.rolling(n).sum().replace(0, np.nan)
    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


def count_consecutive(cond: pd.Series) -> pd.Series:
    out = np.zeros(len(cond), dtype=int)
    count = 0
    for i, ok in enumerate(cond.fillna(False).astype(bool).tolist()):
        count = count + 1 if ok else 0
        out[i] = count
    return pd.Series(out, index=cond.index)


def expanding_percentile(series: pd.Series, min_periods: int = 60) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if i + 1 < min_periods or not np.isfinite(v):
            continue
        window = values[: i + 1]
        window = window[np.isfinite(window)]
        if len(window) < min_periods:
            continue
        out[i] = float((window <= v).mean())
    return pd.Series(out, index=series.index)


def weighted_feature_distance(pool: pd.DataFrame, current_row: pd.Series) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build a weighted feature-vector similarity space for analog matching.

    This is the core professional-style tweak: instead of treating every field as equally
    informative, scale the standardized vector by feature importance so momentum/exhaustion
    dimensions drive the nearest-neighbor search more than housekeeping fields.
    """
    X = pool[FEATURE_COLS].astype(float)
    cur_df = pd.DataFrame([current_row[FEATURE_COLS]], columns=FEATURE_COLS).astype(float)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    cur_scaled = pd.DataFrame(scaler.transform(cur_df), index=cur_df.index, columns=cur_df.columns)
    weights = pd.Series({c: FEATURE_WEIGHTS.get(c, 1.0) for c in FEATURE_COLS}, index=FEATURE_COLS, dtype=float)
    X_weighted = X_scaled.mul(weights, axis=1)
    cur_weighted = cur_scaled.mul(weights, axis=1)
    dist = cdist(cur_weighted.values, X_weighted.values)[0]
    return dist, X_weighted, cur_weighted


@st.cache_data(ttl=86400, show_spinner=False)
def get_history(symbol: str, years: int = 3) -> Optional[pd.DataFrame]:
    try:
        t = Ticker(symbol)
        df = t.price()
        rename_map = {
            "report_date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        df = df.rename(columns=rename_map)
        needed = ["date", "open", "high", "low", "close", "volume"]
        if all(c in df.columns for c in needed):
            df = df[needed].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            if not df.empty:
                return df.tail(252 * years)
    except Exception:
        pass

    try:
        df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        needed = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            return None
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["open", "high", "low", "close"])
        return df.tail(252 * years)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_benchmark(symbol: str = "SPY", years: int = 3) -> Optional[pd.Series]:
    hist = get_history(symbol, years=years)
    if hist is None or hist.empty:
        return None
    return hist["close"].rename("bench_close")


@st.cache_data(ttl=900, show_spinner=False)
def get_live_daily(symbol: str, lookback_days: int = 300) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period="2y", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        needed = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            return None
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["open", "high", "low", "close"])
        return df.tail(lookback_days)
    except Exception:
        return None


def classify_tsi_state(tsi_val: float, sig_val: float, slope1: float, pinned_bars: int) -> str:
    if pd.isna(tsi_val) or pd.isna(sig_val):
        return "below_zero"
    if tsi_val < -60:
        return "oversold"
    if tsi_val < 0:
        return "below_zero"
    if pinned_bars >= 3 and abs(slope1) <= 0.5 and tsi_val >= 95:
        return "pinned"
    if tsi_val >= 90 and abs(slope1) <= 1.0:
        return "near_pinned"
    if tsi_val > 70 and slope1 <= 0 and tsi_val <= sig_val + 2:
        return "bear_kiss"
    if tsi_val > sig_val and tsi_val > 0 and slope1 > 0:
        return "bull_continuation"
    if tsi_val > sig_val and tsi_val <= 40:
        return "bull_repair"
    if tsi_val > 0 and slope1 < 0:
        return "rolling"
    return "bull_repair"


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["high"] - out[["open", "close"]].max(axis=1)) / rng
    out["lower_wick_pct"] = (out[["open", "close"]].min(axis=1) - out["low"]) / rng
    out["body_pct"] = (out["close"] - out["open"]).abs() / rng
    out["candle_close_loc"] = (out["close"] - out["low"]) / rng
    out["failed_gap"] = ((out["open"] > out["close"].shift(1)) & (out["close"] < out["open"]))
    out["outside_reversal"] = ((out["high"] > out["high"].shift(1)) & (out["close"] < out["high"].shift(1)))
    out["candle_score"] = 50.0
    out["candle_score"] += out["upper_wick_pct"].fillna(0) * 25
    out["candle_score"] -= out["lower_wick_pct"].fillna(0) * 10
    out["candle_score"] += np.where(out["close"] < out["open"], out["body_pct"].fillna(0) * 10, -out["body_pct"].fillna(0) * 15)
    out["candle_score"] += out["failed_gap"].astype(int) * 12
    out["candle_score"] += out["outside_reversal"].astype(int) * 10
    out["candle_score"] += (0.5 - out["candle_close_loc"].fillna(0.5)) * 20
    out["candle_score"] = out["candle_score"].clip(0, 100)
    return out


def build_features(df: pd.DataFrame, bench: Optional[pd.Series]) -> pd.DataFrame:
    out = df.copy()
    out["TSI_424"], out["TSI_424_sig"] = tsi(out["close"], 4, 2, 4)
    out["TSI_747"], out["TSI_747_sig"] = tsi(out["close"], 7, 4, 7)
    out["TSI_1377"], out["TSI_1377_sig"] = tsi(out["close"], 13, 7, 7)
    out["TSI_25137"], out["TSI_25137_sig"] = tsi(out["close"], 25, 13, 7)

    for col in ["TSI_424", "TSI_747", "TSI_1377", "TSI_25137"]:
        out[f"{col}_minus_sig"] = out[col] - out[f"{col}_sig"]
        out[f"{col}_slope_1"] = out[col].diff(1)
        out[f"{col}_slope_3"] = out[col].diff(3)

    out["TSI_424_pct"] = expanding_percentile(out["TSI_424"], min_periods=80)
    out["TSI_747_pct"] = expanding_percentile(out["TSI_747"], min_periods=80)

    out["CCI15"] = cci(out, 15)
    out["CCI20"] = cci(out, 20)
    out["CCI15_pct"] = expanding_percentile(out["CCI15"], min_periods=80)
    out["CCI15_days_fading"] = count_consecutive(out["CCI15"].diff() < 0)
    out["CCI20_days_fading"] = count_consecutive(out["CCI20"].diff() < 0)
    out["CCI_divergence"] = ((out["close"] >= out["close"].shift(1)) & (out["CCI15"] < out["CCI15"].shift(1))).astype(int)

    out["BBP"] = bbpct(out["close"])
    out["BBP_pct"] = expanding_percentile(out["BBP"], min_periods=80)
    out["BBP_delta_1"] = out["BBP"].diff(1)
    out["BBP_days_falling"] = count_consecutive(out["BBP"].diff() < 0)

    out["RSI14"] = rsi(out["close"], 14)
    out["RSI14_slope_3"] = out["RSI14"].diff(3)
    out["RSI_divergence"] = ((out["close"] >= out["close"].shift(1)) & (out["RSI14"] <= out["RSI14"].shift(1))).astype(int)

    out["VWAP"] = rolling_vwap(out, 20)
    out["EXT_VWAP"] = (out["close"] - out["VWAP"]) / out["VWAP"]
    out["VWAP_slope"] = out["EXT_VWAP"].diff(1)
    out["VWAP_stall"] = ((out["EXT_VWAP"] > 0.03) & (out["VWAP_slope"] <= 0)).astype(int)

    out["SMA10"] = out["close"].rolling(10).mean()
    out["SMA20"] = out["close"].rolling(20).mean()
    out["EXT_SMA10"] = (out["close"] - out["SMA10"]) / out["SMA10"]
    out["EXT_SMA20"] = (out["close"] - out["SMA20"]) / out["SMA20"]

    out["ATR14"] = atr(out, 14)
    out["stretch_ATR"] = (out["close"] - out["SMA20"]) / out["ATR14"].replace(0, np.nan)
    out["ATR_pct"] = out["ATR14"].rolling(60, min_periods=20).rank(pct=True)

    out["MFI14"] = mfi(out, 14)
    out["MFI14_slope"] = out["MFI14"].diff(1)
    out["MFI_divergence"] = ((out["close"] >= out["close"].shift(1)) & (out["MFI14"] < out["MFI14"].shift(1))).astype(int)

    out["ADX14"] = adx(out, 14)
    out["ADX14_slope_3"] = out["ADX14"].diff(3)
    out = compute_candle_features(out)

    out["pinned_424"] = count_consecutive((out["TSI_424"] >= 95) & (out["TSI_424"].diff().abs() <= 1.0))
    out["pinned_747"] = count_consecutive((out["TSI_747"] >= 85) & (out["TSI_747"].diff().abs() <= 1.0))
    out["pinned_1377"] = count_consecutive((out["TSI_1377"] >= 55) & (out["TSI_1377"].diff().abs() <= 1.0))
    out["pinned_25137"] = count_consecutive((out["TSI_25137"] >= 20) & (out["TSI_25137"].diff().abs() <= 0.6))

    if bench is not None and not bench.empty:
        aligned = bench.reindex(out.index).ffill()
        out["bench_close"] = aligned
        out["RS_LINE"] = out["close"] / out["bench_close"].replace(0, np.nan)
        out["RS_SLOPE_5"] = out["RS_LINE"].diff(5) / out["RS_LINE"].shift(5)
        out["RS_SLOPE_10"] = out["RS_LINE"].diff(10) / out["RS_LINE"].shift(10)
        out["RS_divergence"] = ((out["close"] >= out["close"].shift(1)) & (out["RS_SLOPE_5"].fillna(0) < 0)).astype(int)
    else:
        out["RS_LINE"] = 1.0
        out["RS_SLOPE_5"] = 0.0
        out["RS_SLOPE_10"] = 0.0
        out["RS_divergence"] = 0

    # Market / regime context from benchmark proxy
    if bench is not None and not bench.empty:
        bclose = bench.reindex(out.index).ffill()
        bdf = pd.DataFrame({"close": bclose})
        bdf["SMA20"] = bdf["close"].rolling(20).mean()
        bdf["ATR14"] = (bdf["close"].diff().abs()).rolling(14).mean()
        bdf["stretch_atr"] = (bdf["close"] - bdf["SMA20"]) / bdf["ATR14"].replace(0, np.nan)
        bdf["CCI15"] = ((bdf["close"] - bdf["close"].rolling(15).mean()) / (0.015 * (bdf["close"] - bdf["close"].rolling(15).mean()).abs().rolling(15).mean().replace(0, np.nan)))
        bdf["TSI_747"], bdf["TSI_747_sig"] = tsi(bdf["close"], 7, 4, 7)
        bdf["TSI_747_slope_1"] = bdf["TSI_747"].diff(1)
        bdf["pinned_747"] = count_consecutive((bdf["TSI_747"] >= 85) & (bdf["TSI_747"].diff().abs() <= 1.0))
        mkt_state = []
        for _, row in bdf.iterrows():
            mkt_state.append(classify_tsi_state(row.get("TSI_747"), row.get("TSI_747_sig"), row.get("TSI_747_slope_1"), int(row.get("pinned_747", 0)) if pd.notna(row.get("pinned_747", 0)) else 0))
        out["mkt_tsi_747"] = bdf["TSI_747"]
        out["mkt_tsi_state_code"] = pd.Series(mkt_state, index=out.index).map(STATE_CODE_MAP).fillna(0)
        out["mkt_above_ema20"] = (bclose > bclose.ewm(span=20, adjust=False).mean()).astype(int)
        out["mkt_stretch_atr"] = bdf["stretch_atr"].fillna(0)
        out["mkt_cci15"] = bdf["CCI15"].fillna(0)
        out["mkt_hot"] = ((bdf["TSI_747"] > 70) & (bdf["stretch_atr"] > 1.5)).astype(int)
        out["mkt_fading"] = (((bdf["TSI_747"] > 60) & (bdf["TSI_747_slope_1"] <= 0)) | (bdf["CCI15"].diff() < 0)).astype(int)
        out["leadership_regime"] = (out["RS_SLOPE_5"].fillna(0) > 0).astype(int)
        out["risk_on_regime"] = ((out["RS_SLOPE_5"].fillna(0) > 0) & (out["mkt_above_ema20"] == 1)).astype(int)
    else:
        for col, default in {
            "mkt_tsi_747": 0.0,
            "mkt_tsi_state_code": 0,
            "mkt_above_ema20": 0,
            "mkt_stretch_atr": 0.0,
            "mkt_cci15": 0.0,
            "mkt_hot": 0,
            "mkt_fading": 0,
            "leadership_regime": 0,
            "risk_on_regime": 0,
        }.items():
            out[col] = default

    out["high_vol_regime"] = (out["ATR_pct"] >= 0.75).astype(int)
    out["low_vol_regime"] = (out["ATR_pct"] <= 0.25).astype(int)

    for tcol, pcol, scol, codecol in [
        ("TSI_424", "pinned_424", "state_424", "state_424_code"),
        ("TSI_747", "pinned_747", "state_747", "state_747_code"),
        ("TSI_1377", "pinned_1377", "state_1377", "state_1377_code"),
        ("TSI_25137", "pinned_25137", "state_25137", "state_25137_code"),
    ]:
        vals = []
        for _, row in out.iterrows():
            vals.append(classify_tsi_state(row[tcol], row[f"{tcol}_sig"], row[f"{tcol}_slope_1"], int(row[pcol]) if pd.notna(row[pcol]) else 0))
        out[scol] = vals
        out[codecol] = out[scol].map(STATE_CODE_MAP).fillna(0)

    price_higher = out["close"] >= out["close"].rolling(5, min_periods=2).max().shift(1).fillna(out["close"].shift(1)) * 0.98
    tsi_still_strong = (out["TSI_424"] > 80) & (out["TSI_747"] > 60)
    tsi_747_fail = out["TSI_747"] <= out["TSI_747"].shift(1) + 0.3
    cci_early_fade = (out["CCI15"] > 90) & (out["CCI15"].diff() < 0) & (out["CCI15_days_fading"] >= 1)
    bb_stall = (out["BBP"] > 0.95) & ((out["BBP_days_falling"] >= 1) | (out["BBP_delta_1"] <= 0))
    vwap_stall = out["VWAP_stall"] == 1
    mfi_fade = (out["MFI14"] > 70) & (out["MFI14_slope"] < 0)
    rsi_nonconfirm = (out["RSI14"] > 65) & (out["RSI_divergence"] == 1)
    rs_fade = out["RS_divergence"] == 1
    candle_bear = out["candle_score"] >= 55
    strong_trend_penalty = ((out["ADX14"] > 28) & (out["ADX14_slope_3"] > 0)).astype(int)

    out["early_bear_setup"] = (price_higher & cci_early_fade & tsi_still_strong & (out["BBP"] > 0.95)).astype(int)
    out["bear_kiss_score"] = (
        price_higher.astype(int) * 2.0
        + cci_early_fade.astype(int) * 2.5
        + tsi_still_strong.astype(int) * 2.0
        + tsi_747_fail.astype(int) * 1.5
        + bb_stall.astype(int) * 1.5
        + vwap_stall.astype(int) * 1.5
        + mfi_fade.astype(int) * 1.0
        + rsi_nonconfirm.astype(int) * 1.0
        + rs_fade.astype(int) * 1.5
        + candle_bear.astype(int) * 1.0
        - strong_trend_penalty.astype(int) * 1.5
    )
    out["is_bear_kiss"] = (out["bear_kiss_score"] >= 7.5).astype(int)

    price_lower = out["close"] <= out["close"].rolling(5, min_periods=2).min().shift(1).fillna(out["close"].shift(1)) * 1.02
    tsi_747_turn = out["TSI_747"] >= out["TSI_747"].shift(1) - 0.2
    cci_repair = ((out["CCI15"] < -100) & (out["CCI15"].diff() > 0)) | (count_consecutive(out["CCI15"].diff() > 0) >= 2)
    bb_oversold = (out["BBP"] < 0.05) | ((out["BBP"] < 0.15) & (out["BBP_delta_1"] > 0))
    mfi_repair = (out["MFI14"] < 30) & (out["MFI14_slope"] > 0)
    rs_repair = (out["RS_SLOPE_5"].fillna(0) > 0)
    out["bull_kiss_score"] = (
        price_lower.astype(int) * 1.5
        + tsi_747_turn.astype(int) * 2.0
        + cci_repair.astype(int) * 2.5
        + bb_oversold.astype(int) * 2.0
        + mfi_repair.astype(int) * 1.0
        + rs_repair.astype(int) * 1.0
        + (out["candle_score"] >= 60).astype(int) * 1.5
    )
    out["is_bull_kiss"] = (out["bull_kiss_score"] >= 7.0).astype(int)

    out["is_hot"] = ((out["TSI_424"] > 95) & (out["TSI_747"] > 70) & (out["BBP"] > 0.95)).astype(int)
    out["is_cold"] = ((out["TSI_424"] < -95) & (out["BBP"] < 0.05) & (out["CCI15"] < -100)).astype(int)
    out["is_ripping"] = ((out["TSI_424"] > 95) & (out["TSI_747_slope_1"] > 0.8) & (out["RS_SLOPE_5"].fillna(0) > 0)).astype(int)
    out["is_fading_stack"] = ((out["CCI15_days_fading"] >= 2) & (out["TSI_424_slope_1"] <= 0) & (out["TSI_747_slope_1"] <= 0) & ((out["BBP_days_falling"] >= 1) | (out["VWAP_stall"] == 1))).astype(int)
    out["is_repairing_stack"] = ((count_consecutive(out["CCI15"].diff() > 0) >= 2) & (out["TSI_424_slope_1"] > 0) & (out["BBP_delta_1"] > 0)).astype(int)
    return out


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["close"].shift(-1) / out["close"] - 1
    out["ret2"] = out["close"].shift(-2) / out["close"] - 1
    out["ret5"] = out["close"].shift(-5) / out["close"] - 1
    out["dip1"] = (out["ret1"] < 0).astype(float)
    out["dip2"] = (out["ret2"] < -0.005).astype(float)
    out["dip5"] = (out["ret5"] < -0.01).astype(float)
    out["rip1"] = (out["ret1"] > 0).astype(float)
    out["rip2"] = (out["ret2"] > 0.005).astype(float)
    out["rip5"] = (out["ret5"] > 0.01).astype(float)
    return out


def feature_store_path(symbol: str, years: int, benchmark_symbol: str) -> Path:
    key = hashlib.md5(f"{symbol}|{years}|{benchmark_symbol}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"feature_store_{symbol}_{years}_{benchmark_symbol}_{key}.parquet"


def load_or_build_feature_store(symbol: str, years: int, benchmark_symbol: str) -> Optional[pd.DataFrame]:
    path = feature_store_path(symbol, years, benchmark_symbol)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    df = get_history(symbol, years=years)
    if df is None or df.empty:
        return None
    bench = get_benchmark(benchmark_symbol, years=years)
    feat = build_features(df, bench)
    feat = add_returns(feat)
    for col, default in {
        "RS_LINE": 1.0,
        "RS_SLOPE_5": 0.0,
        "RS_SLOPE_10": 0.0,
        "ADX14": 0.0,
        "ADX14_slope_3": 0.0,
        "TSI_424_pct": 0.5,
        "TSI_747_pct": 0.5,
        "CCI15_pct": 0.5,
        "BBP_pct": 0.5,
    }.items():
        if col not in feat.columns:
            feat[col] = default
        feat[col] = feat[col].fillna(default)
    try:
        feat.to_parquet(path)
    except Exception:
        pass
    return feat


def get_row_as_of(feat: pd.DataFrame, as_of_date: pd.Timestamp) -> Optional[pd.Series]:
    if feat is None or feat.empty:
        return None
    eligible = feat[feat.index <= as_of_date]
    if eligible.empty:
        return None
    return eligible.iloc[-1].copy()


def find_analogs(hist_df: pd.DataFrame, current_row: pd.Series, n: int = 30, exclusion_gap: int = 20) -> pd.DataFrame:
    working = hist_df.copy()
    for col, default in {
        "RS_LINE": 1.0,
        "RS_SLOPE_5": 0.0,
        "RS_SLOPE_10": 0.0,
        "ADX14": 0.0,
        "ADX14_slope_3": 0.0,
        "mkt_tsi_747": 0.0,
        "mkt_tsi_state_code": 0,
        "mkt_above_ema20": 0,
        "mkt_stretch_atr": 0.0,
        "mkt_cci15": 0.0,
        "mkt_hot": 0,
        "mkt_fading": 0,
        "high_vol_regime": 0,
        "low_vol_regime": 0,
        "leadership_regime": 0,
        "risk_on_regime": 0,
    }.items():
        if col in working.columns:
            working[col] = working[col].fillna(default)
    base = working.dropna(subset=FEATURE_COLS + REGIME_COLS + ["ret1", "ret2", "ret5"]).copy()
    if len(base) < 100:
        return pd.DataFrame()
    if current_row.name in base.index:
        current_pos = base.index.get_loc(current_row.name)
        pool = base.iloc[:max(0, current_pos - exclusion_gap)].copy()
        if pool.empty:
            pool = base.iloc[:-exclusion_gap].copy()
    else:
        pool = base.iloc[:-exclusion_gap].copy()
    if pool.empty:
        return pd.DataFrame()

    # Regime-aware filtering: narrow analog pool to similar tape first.
    same_state = pool[pool["mkt_tsi_state_code"] == current_row.get("mkt_tsi_state_code", 0)].copy()
    if len(same_state) >= max(40, n * 3):
        pool = same_state
    same_hotfade = pool[(pool["mkt_hot"] == current_row.get("mkt_hot", 0)) & (pool["mkt_fading"] == current_row.get("mkt_fading", 0))].copy()
    if len(same_hotfade) >= max(30, n * 2):
        pool = same_hotfade
    same_vol = pool[(pool["high_vol_regime"] == current_row.get("high_vol_regime", 0)) & (pool["low_vol_regime"] == current_row.get("low_vol_regime", 0))].copy()
    if len(same_vol) >= max(25, int(n * 1.5)):
        pool = same_vol

    dist, X_weighted, cur_weighted = weighted_feature_distance(pool, current_row)
    pool["distance"] = dist
    # Diagnostics: which feature buckets drove the match most?
    momentum_cols = [c for c in ["TSI_424", "TSI_424_slope_1", "TSI_747", "TSI_747_slope_3", "pinned_424", "CCI15", "CCI15_days_fading", "CCI_divergence"] if c in X_weighted.columns]
    stretch_cols = [c for c in ["BBP", "EXT_VWAP", "VWAP_slope", "VWAP_stall", "stretch_ATR", "ATR_pct"] if c in X_weighted.columns]
    participation_cols = [c for c in ["MFI14", "MFI14_slope", "MFI_divergence", "RS_SLOPE_5", "RS_SLOPE_10", "RS_divergence"] if c in X_weighted.columns]
    if momentum_cols:
        pool["momentum_distance"] = np.sqrt(((X_weighted[momentum_cols] - cur_weighted.iloc[0][momentum_cols]) ** 2).sum(axis=1))
    else:
        pool["momentum_distance"] = 0.0
    if stretch_cols:
        pool["stretch_distance"] = np.sqrt(((X_weighted[stretch_cols] - cur_weighted.iloc[0][stretch_cols]) ** 2).sum(axis=1))
    else:
        pool["stretch_distance"] = 0.0
    if participation_cols:
        pool["participation_distance"] = np.sqrt(((X_weighted[participation_cols] - cur_weighted.iloc[0][participation_cols]) ** 2).sum(axis=1))
    else:
        pool["participation_distance"] = 0.0

    # Regime penalties keep analogs inside a similar tape even when feature distances are close.
    pool["regime_penalty"] = (
        (pool["mkt_tsi_state_code"] != current_row.get("mkt_tsi_state_code", 0)).astype(float) * 0.50
        + (pool["mkt_hot"] != current_row.get("mkt_hot", 0)).astype(float) * 0.30
        + (pool["mkt_fading"] != current_row.get("mkt_fading", 0)).astype(float) * 0.35
        + (pool["high_vol_regime"] != current_row.get("high_vol_regime", 0)).astype(float) * 0.25
        + (pool["leadership_regime"] != current_row.get("leadership_regime", 0)).astype(float) * 0.20
        + ((pool["mkt_stretch_atr"] - float(current_row.get("mkt_stretch_atr", 0))) .abs() > 1.0).astype(float) * 0.15
    )
    pool["state_penalty"] = (
        (pool["state_424_code"] != current_row["state_424_code"]).astype(float) * 0.15
        + (pool["state_747_code"] != current_row["state_747_code"]).astype(float) * 0.35
        + (pool["state_1377_code"] != current_row["state_1377_code"]).astype(float) * 0.20
        + (pool["is_bear_kiss"] != current_row["is_bear_kiss"]).astype(float) * 0.20
        + (pool["is_bull_kiss"] != current_row["is_bull_kiss"]).astype(float) * 0.20
    )
    # Slight recency boost: more recent analogs tend to be more useful for short-horizon options timing.
    if isinstance(pool.index, pd.DatetimeIndex) and hasattr(current_row.name, 'to_pydatetime'):
        age_delta = pd.Timestamp(current_row.name) - pd.to_datetime(pool.index)
        age_days = np.maximum((age_delta / np.timedelta64(1, "D")).astype(float), 1.0)
        pool["recency_penalty"] = np.clip(age_days / 2520.0, 0, 0.20)
    else:
        pool["recency_penalty"] = 0.0

    pool["total_distance"] = pool["distance"] + pool["state_penalty"] + pool["regime_penalty"] + pool["recency_penalty"]
    analogs = pool.nsmallest(n, "total_distance").copy()
    analogs["similarity"] = 1 / (1 + analogs["total_distance"])
    return analogs


def weighted_stats(analogs: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if analogs.empty:
        return {}
    w = analogs["similarity"].clip(lower=1e-9)
    stats = {}
    for ret_col, down_col, up_col in [("ret1", "dip1", "rip1"), ("ret2", "dip2", "rip2"), ("ret5", "dip5", "rip5")]:
        r = analogs[ret_col]
        d = analogs[down_col]
        u = analogs[up_col]
        stats[ret_col] = {
            "down_prob": float(np.average(d, weights=w)),
            "up_prob": float(np.average(u, weights=w)),
            "mean": float(np.average(r, weights=w)),
            "median": float(r.median()),
            "p10": float(r.quantile(0.10)),
            "p90": float(r.quantile(0.90)),
        }
    return stats


def confidence_score(analogs: pd.DataFrame, current: pd.Series, stats: Dict[str, Dict[str, float]], side: str) -> float:
    if analogs.empty:
        return 0.0
    size_score = min(len(analogs) / 30.0, 1.0)
    closeness = float(np.clip(1 / (1 + analogs["total_distance"].mean()), 0, 1))
    agreement = 0.0
    if side == "PUT":
        if current.get("is_hot", 0): agreement += 0.10
        if current.get("is_bear_kiss", 0): agreement += 0.25
        if current.get("bear_kiss_score", 0) >= 10: agreement += 0.12
        if current.get("candle_score", 0) >= 68: agreement += 0.15
        if current.get("CCI15_days_fading", 0) >= 2: agreement += 0.10
        if current.get("BBP_days_falling", 0) >= 1: agreement += 0.08
        if current.get("state_747") in {"bear_kiss", "pinned", "near_pinned", "rolling"}: agreement += 0.10
    else:
        if current.get("is_cold", 0): agreement += 0.10
        if current.get("is_bull_kiss", 0): agreement += 0.25
        if current.get("bull_kiss_score", 0) >= 9: agreement += 0.12
        if current.get("candle_score", 0) >= 60: agreement += 0.15
        if current.get("CCI15", 0) < -100: agreement += 0.10
        if current.get("BBP", 0) < 0.10: agreement += 0.08
        if current.get("state_747") in {"oversold", "below_zero", "bull_repair"}: agreement += 0.10
    probs = [stats["ret1"]["down_prob" if side == "PUT" else "up_prob"], stats["ret2"]["down_prob" if side == "PUT" else "up_prob"], stats["ret5"]["down_prob" if side == "PUT" else "up_prob"]]
    outcome_consistency = max(0.0, 1 - float(np.std(probs)) * 2)
    conf = 100 * (0.20 * size_score + 0.35 * closeness + 0.25 * agreement + 0.20 * outcome_consistency)
    return float(np.clip(conf, 0, 100))


def calculate_bull_score(row: pd.Series) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []
    if row.get("TSI_747", 0) < 0:
        score += 1.0; reasons.append("TSI_747 below zero")
    if row.get("is_bull_kiss", 0):
        score += 3.0; reasons.append("bull kiss")
    if row.get("bull_kiss_score", 0) >= 9:
        score += 2.0; reasons.append(f"bull kiss {row['bull_kiss_score']:.0f}")
    if row.get("RSI14", 0) < 30:
        score += 2.0; reasons.append("RSI oversold")
    elif 30 < row.get("RSI14", 0) < 40 and row.get("RSI14_slope_3", 0) > 0:
        score += 1.5; reasons.append("RSI repairing")
    if row.get("CCI15", 0) < -100:
        score += 2.0; reasons.append("CCI extreme oversold")
    if row.get("BBP", 0) < 0.05:
        score += 2.0; reasons.append("BB% washed out")
    if row.get("TSI_424_slope_1", 0) > 0 and row.get("TSI_747_slope_1", 0) >= 0:
        score += 1.0; reasons.append("TSI turning up")
    if row.get("candle_score", 0) >= 60:
        score += 1.0; reasons.append("good reversal candle")
    if row.get("RS_SLOPE_5", 0) > 0:
        score += 0.5; reasons.append("RS firming")
    return min(score, 10.0), reasons[:5]


def calculate_bear_score(row: pd.Series) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []
    if row.get("early_bear_setup", 0):
        score += 2.0; reasons.append("early CCI fade")
    if row.get("TSI_747", 0) > 70 and row.get("TSI_747_slope_1", 0) <= 0:
        score += 1.5; reasons.append("TSI_747 rolling")
    if row.get("is_bear_kiss", 0):
        score += 3.0; reasons.append("bear kiss")
    if row.get("bear_kiss_score", 0) >= 9:
        score += 2.0; reasons.append(f"bear kiss {row['bear_kiss_score']:.0f}")
    if row.get("CCI15", 0) > 100:
        score += 1.5; reasons.append("CCI stretched")
    if row.get("CCI_divergence", 0):
        score += 1.0; reasons.append("CCI divergence")
    if row.get("BBP", 0) > 0.95:
        score += 1.0; reasons.append("BB% hot")
    if row.get("VWAP_stall", 0):
        score += 1.0; reasons.append("VWAP stall")
    if row.get("MFI_divergence", 0):
        score += 0.75; reasons.append("MFI divergence")
    if row.get("RS_divergence", 0):
        score += 0.75; reasons.append("RS divergence")
    if row.get("pinned_424", 0) >= 2:
        score += 0.75; reasons.append("4,2,4 pinned")
    if row.get("ADX14", 0) > 28 and row.get("ADX14_slope_3", 0) > 0:
        score -= 1.0; reasons.append("trend too strong")
    return float(np.clip(score, 0, 10)), reasons[:6]


def get_signal_type(bull_score: float, bear_score: float) -> Tuple[str, Optional[str], float]:
    if bull_score >= 6 and bull_score > bear_score + 1.5:
        return "🟢 BULL", "CALL", bull_score / 10
    if bear_score >= 6 and bear_score > bull_score + 1.5:
        return "🔴 BEAR", "PUT", bear_score / 10
    if bull_score >= 4 and bull_score > bear_score:
        return "🟡 BULLISH BIAS", "CALL", bull_score / 10 * 0.7
    if bear_score >= 4 and bear_score > bull_score:
        return "🟠 BEARISH BIAS", "PUT", bear_score / 10 * 0.7
    return "⚪ NEUTRAL", None, 0.3


def setup_label(row: pd.Series) -> str:
    if row.get("is_bear_kiss", 0) and row.get("TSI_747", 0) > 60 and row.get("CCI15_days_fading", 0) >= 1:
        return "🔥 Diamond Bear"
    if row.get("early_bear_setup", 0):
        return "🟠 Early Bear Setup"
    if row.get("is_bull_kiss", 0) and row.get("candle_score", 0) >= 60 and row.get("CCI15", 0) < -50:
        return "💎 Diamond Bull"
    if row.get("bear_kiss_score", 0) >= 8:
        return "🟠 Near Bear Diamond"
    if row.get("bull_kiss_score", 0) >= 7.5:
        return "🟢 Near Bull Diamond"
    if row.get("pinned_424", 0) >= 2 or row.get("state_747") in {"pinned", "near_pinned"}:
        return "🟡 Pinned Extreme"
    if row.get("is_cold", 0):
        return "🔵 Washed Out"
    return "⚪ Watch"


def why_text(row: pd.Series) -> str:
    reasons = []
    if row.get("pinned_424", 0) >= 2:
        reasons.append(f"{int(row['pinned_424'])}x 4,2,4 pinned")
    if row.get("early_bear_setup", 0):
        reasons.append("early CCI fade")
    if row.get("state_747") == "bear_kiss":
        reasons.append("7,4,7 bear kiss")
    if row.get("bull_kiss_score", 0) >= 2:
        reasons.append(f"{int(row['bull_kiss_score'])}x bull kiss")
    if row.get("CCI15_days_fading", 0) >= 2:
        reasons.append(f"CCI15 fading {int(row['CCI15_days_fading'])}d")
    if row.get("CCI15", 0) < -100:
        reasons.append("CCI oversold")
    if row.get("BBP_days_falling", 0) >= 1 and row.get("BBP", 0) > 0.95:
        reasons.append("BB% stalled high")
    if row.get("VWAP_stall", 0):
        reasons.append("VWAP stall")
    if row.get("MFI_divergence", 0):
        reasons.append("MFI divergence")
    if row.get("RS_divergence", 0):
        reasons.append("RS divergence")
    if row.get("BBP", 0) < 0.05:
        reasons.append("BB% washed out")
    if row.get("candle_score", 0) >= 68:
        reasons.append("good candle")
    if row.get("RS_SLOPE_5", 0) < 0:
        reasons.append("RS fading")
    if row.get("RS_SLOPE_5", 0) > 0:
        reasons.append("RS firming")
    if row.get("TSI_25137_slope_3", 0) <= 0.5:
        reasons.append("25,13,7 flattening")
    if row.get("mkt_hot", 0):
        reasons.append("market hot")
    if row.get("mkt_fading", 0):
        reasons.append("market fading")
    if row.get("high_vol_regime", 0):
        reasons.append("high vol regime")
    return ", ".join(reasons[:7]) if reasons else "No stacked trigger yet"


@st.cache_data(ttl=900, show_spinner=False)
def get_option_candidates(symbol: str, option_type: str, max_expirations: int = 3) -> Optional[pd.DataFrame]:
    try:
        ticker = yf.Ticker(symbol)
        exps = list(ticker.options)
        if not exps:
            return None
        hist = ticker.history(period="5d")
        if hist is None or hist.empty:
            return None
        current_price = float(hist["Close"].iloc[-1])
        all_options = []
        now = pd.Timestamp.utcnow().tz_localize(None)
        for exp_date in exps[:max_expirations]:
            chain = ticker.option_chain(exp_date)
            options = chain.calls.copy() if option_type == "CALL" else chain.puts.copy()
            if options is None or options.empty:
                continue
            exp_dt = pd.Timestamp(exp_date)
            dte = int((exp_dt - now.normalize()).days)
            if option_type == "CALL":
                options = options[options["strike"] >= current_price * 0.95]
            else:
                options = options[options["strike"] <= current_price * 1.05]
            if options.empty:
                continue
            options["spread"] = options["ask"] - options["bid"]
            options["mid"] = (options["ask"] + options["bid"]) / 2
            options["expiration"] = exp_date
            options["days_to_expiry"] = dte
            options["option_type"] = option_type
            rel_spread = (options["spread"] / options["mid"].replace(0, np.nan)).clip(upper=1).fillna(1)
            options["liq_score"] = (
                options["volume"].fillna(0).clip(upper=5000) / 5000 * 0.35
                + options["openInterest"].fillna(0).clip(upper=10000) / 10000 * 0.35
                + (1 - rel_spread) * 0.30
            )
            all_options.append(options)
        if not all_options:
            return None
        result = pd.concat(all_options, ignore_index=True)
        result = result.sort_values(["liq_score", "volume", "openInterest"], ascending=False)
        cols = ["contractSymbol", "expiration", "days_to_expiry", "strike", "option_type", "bid", "ask", "mid", "spread", "volume", "openInterest", "impliedVolatility", "liq_score"]
        cols = [c for c in cols if c in result.columns]
        return result[cols].head(12)
    except Exception:
        return None


def expected_move_pct(current_row: pd.Series) -> float:
    atr_val = float(current_row.get("ATR14", np.nan))
    close_val = float(current_row.get("close", np.nan)) if "close" in current_row.index else np.nan
    if pd.isna(atr_val) or pd.isna(close_val) or close_val == 0:
        return np.nan
    return atr_val / close_val


def position_size_pct(confidence: float, atr_pct: float) -> float:
    try:
        base = max(0.1, min(confidence / 100.0, 1.0))
        vol_adj = 1 - float(atr_pct) if not pd.isna(atr_pct) else 0.75
        return float(max(0.1, min(base * vol_adj, 1.0)))
    except Exception:
        return 0.25


def build_live_or_historical_row(symbol: str, benchmark_symbol: str, years: int, analysis_mode: str, analysis_date: Optional[pd.Timestamp]) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame], str]:
    hist_feat = load_or_build_feature_store(symbol, years, benchmark_symbol)
    if hist_feat is None or hist_feat.empty:
        return None, None, "no history returned"
    if analysis_mode == "Historical":
        if analysis_date is None:
            return None, hist_feat, "missing historical date"
        row = get_row_as_of(hist_feat, pd.Timestamp(analysis_date))
        if row is None:
            return None, hist_feat, "no row for chosen date"
        return row, hist_feat, "ok"
    live_df = get_live_daily(symbol)
    if live_df is None or live_df.empty:
        return None, hist_feat, "no live data"
    live_bench = get_benchmark(benchmark_symbol, years=years)
    live_feat = build_features(live_df, live_bench)
    live_feat = add_returns(live_feat)
    if live_feat.empty:
        return None, hist_feat, "no live features"
    return live_feat.iloc[-1].copy(), hist_feat, "ok"


def scan_symbol(symbol: str, current_row: pd.Series, hist_feat: pd.DataFrame, analog_count: int) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    analogs = find_analogs(hist_feat, current_row=current_row, n=analog_count)
    if analogs.empty:
        return None, None
    stats = weighted_stats(analogs)
    bull_score, bull_reasons = calculate_bull_score(current_row)
    bear_score, bear_reasons = calculate_bear_score(current_row)
    signal_type, option_type, _ = get_signal_type(bull_score, bear_score)
    if option_type == "PUT":
        conf = confidence_score(analogs, current_row, stats, side="PUT")
    elif option_type == "CALL":
        conf = confidence_score(analogs, current_row, stats, side="CALL")
    else:
        conf = 40.0
    pos_size = position_size_pct(conf, float(current_row.get("ATR_pct", np.nan)))
    detail = {
        "symbol": symbol,
        "Signal": signal_type,
        "Option Type": option_type,
        "Bull Score": round(bull_score, 2),
        "Bear Score": round(bear_score, 2),
        "DipProb_1d": round(stats["ret1"]["down_prob"] * 100, 1),
        "DipProb_2d": round(stats["ret2"]["down_prob"] * 100, 1),
        "DipProb_5d": round(stats["ret5"]["down_prob"] * 100, 1),
        "RipProb_1d": round(stats["ret1"]["up_prob"] * 100, 1),
        "RipProb_2d": round(stats["ret2"]["up_prob"] * 100, 1),
        "RipProb_5d": round(stats["ret5"]["up_prob"] * 100, 1),
        "ExpRet_1d": round(stats["ret1"]["median"] * 100, 2),
        "ExpRet_2d": round(stats["ret2"]["median"] * 100, 2),
        "ExpRet_5d": round(stats["ret5"]["median"] * 100, 2),
        "Confidence": round(conf, 1),
        "Position Size %": round(pos_size * 100, 1),
        "TSI_424": round(float(current_row.get("TSI_424", np.nan)), 2),
        "TSI_424_pct": round(float(current_row.get("TSI_424_pct", np.nan)) * 100, 1),
        "TSI_747": round(float(current_row.get("TSI_747", np.nan)), 2),
        "TSI_747_pct": round(float(current_row.get("TSI_747_pct", np.nan)) * 100, 1),
        "TSI_1377": round(float(current_row.get("TSI_1377", np.nan)), 2),
        "TSI_25137": round(float(current_row.get("TSI_25137", np.nan)), 2),
        "RSI14": round(float(current_row.get("RSI14", np.nan)), 2),
        "CCI15": round(float(current_row.get("CCI15", np.nan)), 2),
        "CCI20": round(float(current_row.get("CCI20", np.nan)), 2),
        "BBP": round(float(current_row.get("BBP", np.nan)), 3),
        "ATR14": round(float(current_row.get("ATR14", np.nan)), 3),
        "ATR_pct": round(float(current_row.get("ATR_pct", np.nan)) * 100, 1) if pd.notna(current_row.get("ATR_pct", np.nan)) else np.nan,
        "StretchATR": round(float(current_row.get("stretch_ATR", np.nan)), 2),
        "MFI14": round(float(current_row.get("MFI14", np.nan)), 2),
        "VWAP Ext": round(float(current_row.get("EXT_VWAP", np.nan)) * 100, 2) if pd.notna(current_row.get("EXT_VWAP", np.nan)) else np.nan,
        "Candle": round(float(current_row.get("candle_score", np.nan)), 1),
        "Bull Kiss": round(float(current_row.get("bull_kiss_score", 0)), 1),
        "Bear Kiss": round(float(current_row.get("bear_kiss_score", 0)), 1),
        "State": setup_label(current_row),
        "Why": why_text(current_row),
        "Expected Move %": round(expected_move_pct(current_row) * 100, 2) if pd.notna(expected_move_pct(current_row)) else np.nan,
        "Market Regime": regime_label(current_row),
        "Bull Reasons": bull_reasons,
        "Bear Reasons": bear_reasons,
        "Row Date": str(current_row.name.date()) if hasattr(current_row.name, "date") else str(current_row.name),
    }
    return detail, analogs



def regime_label(row: pd.Series) -> str:
    parts = []
    if row.get("mkt_hot", 0):
        parts.append("Market hot")
    elif row.get("mkt_fading", 0):
        parts.append("Market fading")
    else:
        parts.append("Market neutral")
    if row.get("high_vol_regime", 0):
        parts.append("high vol")
    elif row.get("low_vol_regime", 0):
        parts.append("low vol")
    if row.get("leadership_regime", 0):
        parts.append("RS leading")
    return " | ".join(parts)


st.title("🔍 Diamond Scanner Pro")
st.caption("Historical analog ranking from DefeatBeta + live current bar + calls, puts, and date lookback")

with st.sidebar:
    st.header("Universe")
    symbol_source = st.radio("Choose symbol source", ["Upload CSV file", "Paste symbols manually", "Use default symbols"], index=1)
    symbols = []
    if symbol_source == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload CSV with symbol or Symbol column", type=["csv"])
        if uploaded_file is not None:
            df_up = pd.read_csv(uploaded_file)
            col = "symbol" if "symbol" in df_up.columns else ("Symbol" if "Symbol" in df_up.columns else None)
            if col is None:
                st.error("CSV must contain symbol or Symbol column.")
            else:
                symbols = df_up[col].astype(str).str.strip().str.upper().tolist()
                st.success(f"Loaded {len(symbols)} symbols")
    elif symbol_source == "Paste symbols manually":
        symbols_text = st.text_area("Symbols", value="QQQ,SMH,NVDA,MSFT,AMZN,LYFT", height=120)
        if symbols_text.strip():
            if "," in symbols_text:
                symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
            else:
                symbols = [s.strip().upper() for s in symbols_text.splitlines() if s.strip()]
    else:
        symbols = ["QQQ", "SPY", "IWM", "SMH", "NVDA", "MSFT", "AMZN", "META", "TSLA", "LYFT"]
        st.info("Using default symbols")

    st.header("Mode")
    analysis_mode = st.radio("Analysis mode", ["Current", "Historical"], index=0)
    analysis_date = st.date_input("As-of date", value=pd.Timestamp.today().date(), disabled=(analysis_mode == "Current"))

    st.header("Settings")
    years = st.selectbox("Years of historical data", [2, 3, 5], index=1)
    analog_count = st.slider("Number of analogs", 10, 60, 30, 5)
    benchmark_symbol = st.selectbox("Relative strength benchmark", ["SPY", "QQQ", "IWM"], index=0)

    st.header("Cache")
    rebuild_cache = st.checkbox("Rebuild cached feature stores", value=False)

    with st.form("run_scan_form"):
        run_scan = st.form_submit_button("🚀 Run Scan", type="primary")

if not symbols:
    st.warning("Provide at least one symbol.")
    st.stop()

if run_scan:
    if rebuild_cache:
        for symbol in symbols:
            path = feature_store_path(symbol, years, benchmark_symbol)
            if path.exists():
                path.unlink(missing_ok=True)
    rows = []
    debug_rows = []
    detail_rows = {}
    analogs_map = {}
    live_rows_map = {}
    progress = st.progress(0.0)
    for i, symbol in enumerate(symbols):
        progress.progress((i + 1) / max(1, len(symbols)))
        current_row, hist_feat, status = build_live_or_historical_row(symbol, benchmark_symbol, years, analysis_mode, pd.Timestamp(analysis_date))
        if status != "ok" or current_row is None or hist_feat is None:
            debug_rows.append({"symbol": symbol, "status": status})
            continue
        detail, analogs = scan_symbol(symbol, current_row=current_row, hist_feat=hist_feat, analog_count=analog_count)
        if detail is None or analogs is None or analogs.empty:
            debug_rows.append({"symbol": symbol, "status": "no analogs after feature filtering", "hist_rows": len(hist_feat)})
            continue
        rows.append(detail)
        detail_rows[symbol] = detail
        analogs_map[symbol] = analogs
        live_rows_map[symbol] = current_row
    progress.empty()
    if rows:
        results = pd.DataFrame(rows).sort_values(["Confidence", "Bear Score", "Bull Score"], ascending=[False, False, False]).reset_index(drop=True)
        st.session_state.scan_results = results
        st.session_state.debug_rows = debug_rows
        st.session_state.detail_rows = detail_rows
        st.session_state.analogs_map = analogs_map
        st.session_state.live_rows_map = live_rows_map
    else:
        st.session_state.scan_results = None
        st.session_state.debug_rows = debug_rows
        st.session_state.detail_rows = {}
        st.session_state.analogs_map = {}
        st.session_state.live_rows_map = {}

results = st.session_state.scan_results
debug_rows = st.session_state.debug_rows
detail_rows = st.session_state.detail_rows
analogs_map = st.session_state.analogs_map
live_rows_map = st.session_state.live_rows_map

if results is None or results.empty:
    st.info("Run the scan to see ranked setups.")
    if debug_rows:
        st.subheader("Debug")
        st.dataframe(pd.DataFrame(debug_rows), width="stretch")
    st.stop()

st.subheader("🏆 Ranked Setups")
display_cols = ["symbol", "Signal", "Option Type", "Confidence", "Position Size %", "Bull Score", "Bear Score", "RipProb_1d", "RipProb_2d", "RipProb_5d", "DipProb_1d", "DipProb_2d", "DipProb_5d", "TSI_424", "TSI_747", "CCI15", "MFI14", "BBP", "StretchATR", "Expected Move %", "Market Regime", "Bull Kiss", "Bear Kiss", "State", "Why", "Row Date"]
display_cols = [c for c in display_cols if c in results.columns]
st.dataframe(results[display_cols], width="stretch")

if debug_rows:
    with st.expander("Skipped symbols / debug details"):
        st.dataframe(pd.DataFrame(debug_rows), width="stretch")

st.subheader("🔬 Detailed Analysis")
selected = st.selectbox("Select symbol to inspect", results["symbol"].tolist(), key="inspect_symbol")
if selected:
    detail = detail_rows[selected]
    analogs = analogs_map[selected]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal", detail["Signal"])
    c2.metric("Bull Score", f"{detail['Bull Score']:.1f}")
    c3.metric("Bear Score", f"{detail['Bear Score']:.1f}")
    c4.metric("Confidence", f"{detail['Confidence']:.0f}%")
    c5, c6, c7, c8, c9, c10 = st.columns(6)
    c5.metric("TSI 4,2,4", f"{detail['TSI_424']:.2f}")
    c6.metric("TSI 7,4,7", f"{detail['TSI_747']:.2f}")
    c7.metric("CCI15", f"{detail['CCI15']:.1f}")
    c8.metric("BB%", f"{detail['BBP']:.3f}")
    c9.metric("MFI14", f"{detail['MFI14']:.1f}")
    c10.metric("Stretch/ATR", f"{detail['StretchATR']:.2f}")
    left, right = st.columns(2)
    with left:
        st.markdown("**🟢 Bullish reasons**")
        for r in detail["Bull Reasons"]:
            st.write(f"✅ {r}")
    with right:
        st.markdown("**🔴 Bearish reasons**")
        for r in detail["Bear Reasons"]:
            st.write(f"⚠️ {r}")
    st.markdown(f"**Market Regime:** {detail['Market Regime']}")
    st.markdown(f"**Why:** {detail['Why']}")
    st.write(f"Expected move: **{detail.get('Expected Move %', np.nan):.2f}%** | Suggested size: **{detail.get('Position Size %', np.nan):.1f}%** of max risk budget")
    option_type = detail["Option Type"]
    if option_type in {"CALL", "PUT"} and analysis_mode == "Current":
        st.subheader(f"📦 {option_type} option candidates")
        opts = get_option_candidates(selected, option_type=option_type)
        if opts is not None and not opts.empty:
            st.dataframe(opts, width="stretch")
        else:
            st.info(f"No live {option_type.lower()} candidates available right now.")
    elif analysis_mode == "Historical":
        st.info("Historical mode shows what the signal would have been on that date. Historical option chains are not included.")
    st.subheader("📊 Historical analog matches")
    show_cols = ["close", "ret1", "ret2", "ret5", "similarity", "momentum_distance", "stretch_distance", "participation_distance", "state_747", "bull_kiss_score", "bear_kiss_score", "candle_score"]
    show_cols = [c for c in show_cols if c in analogs.columns]
    st.dataframe(analogs[show_cols].sort_values("similarity", ascending=False).head(15), width="stretch")
    st.subheader("🗓 Historical lookback snapshot")
    st.write(f"As-of row date: **{detail['Row Date']}** | Expected 1d: **{detail['ExpRet_1d']:.2f}%** | Expected 2d: **{detail['ExpRet_2d']:.2f}%** | Expected 5d: **{detail['ExpRet_5d']:.2f}%**")
