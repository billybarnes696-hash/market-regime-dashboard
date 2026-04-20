import io
import time
import shutil
import datetime
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

try:
    from defeatbeta_api import Ticker as DefeatTicker
except Exception:
    try:
        from defeatbeta_api.data.ticker import Ticker as DefeatTicker
    except Exception:
        DefeatTicker = None

# -----------------------------
# CONFIG & SETUP
# -----------------------------
st.set_page_config(page_title="Stable Market Engine Final", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store"
DATA_DIR = CACHE_DIR / "market_data"  # Persistent Parquet Storage
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# VECTORIZED UTILS (CORRECTED)
# -----------------------------
def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """
    TRUE Rolling Percentile Rank using Pandas .rank(pct=True).
    This replaces the Min-Max scaler to restore statistical accuracy.
    
    Logic: Returns the percentile rank of the current value within the rolling window.
    Example: A value of 0.95 means the current value is higher than 95% of values in the window.
    """
    # We enforce a minimum period to ensure statistical significance.
    # Calculating a 252-day percentile based on 10 days of data is misleading.
    min_periods = max(10, window // 5)
    
    # vectorized rank calculation
    # pct=True returns values in [0, 1]
    return series.rolling(window, min_periods=min_periods).rank(pct=True)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13, signal_period: int = 7) -> Tuple[pd.Series, pd.Series]:
    delta = series.diff()
    abs_delta = delta.abs()
    double_smoothed = ema(ema(delta, long_period), short_period)
    double_abs = ema(ema(abs_delta, long_period), short_period)
    tsi_line = 100 * double_smoothed / double_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal_period)
    return tsi_line, signal_line

def bollinger_pct_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)

def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()

def slope(series: pd.Series, bars: int = 3) -> pd.Series:
    return series.diff(bars) / bars

def centered_pct(series: pd.Series) -> pd.Series:
    return (series.fillna(0.5) - 0.5) * 2

# -----------------------------
# DATA FETCHING (ROBUST LAYER)
# -----------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    rename = {str(c): str(c).title() for c in out.columns}
    out = out.rename(columns=rename)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    if len(keep) < 4:
        return pd.DataFrame()
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    try:
        out.index = out.index.tz_localize(None)
    except Exception:
        try:
            out.index = out.index.tz_convert(None)
        except Exception:
            pass
    out = out[~out.index.isna()].sort_index()
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    return out

def get_cache_path(symbol: str) -> Path:
    safe_name = "".join(c for c in symbol if c.isalnum() or c in ('_', '-', '.'))
    return DATA_DIR / f"{safe_name}.parquet"

def is_cache_fresh(filepath: Path, max_hours: int = 24) -> bool:
    if not filepath.exists():
        return False
    mod_time = datetime.datetime.fromtimestamp(filepath.stat().st_mtime)
    return (datetime.datetime.now() - mod_time) < datetime.timedelta(hours=max_hours)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_batch(symbols: List[str], years: int = 10) -> Dict[str, pd.DataFrame]:
    """Bulk fetch to avoid rate limits, with persistent disk caching."""
    if not symbols:
        return {}
    
    # 1. Identify fresh cached data
    symbols_to_fetch = []
    for s in symbols:
        if not is_cache_fresh(get_cache_path(s)):
            symbols_to_fetch.append(s)
            
    cache_status = f"{len(symbols) - len(symbols_to_fetch)} Loaded from Disk, {len(symbols_to_fetch)} Fetching..."
    if len(symbols_to_fetch) > 0:
        st.info(f"🔄 Updating Market Data ({cache_status})")
    else:
        st.success(f"✅ Data Fresh ({cache_status})")

    if not symbols_to_fetch:
        # Load all from disk
        data_map = {}
        for s in symbols:
            path = get_cache_path(s)
            if path.exists():
                try:
                    data_map[s] = pd.read_parquet(path)
                except Exception:
                    pass
        return data_map

    # 2. Bulk Download (Anti-Rate Limit)
    try:
        bulk_df = yf.download(
            symbols_to_fetch, 
            period=f"{max(years, 5)}y", 
            interval="1d", 
            group_by='ticker', 
            auto_adjust=True,  # CRITICAL: Use adjusted prices for accuracy
            progress=False,
            threads=True
        )
    except Exception as e:
        st.warning(f"Bulk download warning: {e}")
        bulk_df = {}

    # 3. Save to Disk
    saved_map = {}
    if len(symbols_to_fetch) == 1:
        bulk_df = {symbols_to_fetch[0]: bulk_df}
    else:
        # Handle multi-index response safely
        if isinstance(bulk_df, pd.DataFrame):
            available_cols = [c for c in bulk_df.columns.levels[0] if c in symbols_to_fetch]
            bulk_df = {ticker: bulk_df[ticker].copy() for ticker in available_cols}

    for ticker, df in bulk_df.items():
        df_clean = normalize_ohlcv(df)
        if not df_clean.empty:
            df_clean.to_parquet(get_cache_path(ticker))
            saved_map[ticker] = df_clean

    # 4. Final Assembly (Fresh + Cache)
    final_map = {}
    for s in symbols:
        path = get_cache_path(s)
        if path.exists():
            try:
                final_map[s] = pd.read_parquet(path)
            except Exception:
                pass
    return final_map

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpha_vantage_daily(symbol: str, api_key: str) -> pd.DataFrame:
    api_key = (api_key or "").strip()
    if not api_key:
        return pd.DataFrame()
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
        r = requests.get(url, params=params, timeout=20)
        payload = r.json()
        key = "Time Series (Daily)"
        if key not in payload:
            return pd.DataFrame()
        rows = []
        for ds, item in payload[key].items():
            rows.append(
                {
                    "Date": pd.to_datetime(ds, errors="coerce"),
                    "Open": float(item.get("1. open", np.nan)),
                    "High": float(item.get("2. high", np.nan)),
                    "Low": float(item.get("3. low", np.nan)),
                    "Close": float(item.get("4. close", np.nan)),
                    "Volume": float(item.get("6. volume", np.nan)),
                }
            )
        df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
        # Save to cache manually for AV
        if not df.empty:
            df.to_parquet(get_cache_path(symbol))
        return normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def add_ultimate_oscillator(out: pd.DataFrame, timeframe_name: str) -> pd.DataFrame:
    spans = {
        "proxy_hourly": (5, 13, 5),
        "proxy_2hour": (6, 18, 5),
        "daily": (8, 21, 7),
        "weekly": (5, 13, 5),
    }

    def pct_col(name: str) -> pd.Series:
        if name in out.columns:
            return centered_pct(out[name])
        return pd.Series(0.0, index=out.index, dtype=float)

    fast, slow, sig = spans[timeframe_name]
    stretch = (
        0.18 * pct_col("rsi_14_pctile")
        + 0.18 * pct_col("cci_20_pctile")
        + 0.14 * pct_col("pct_b_pctile")
        + 0.12 * pct_col("atr_stretch_pctile")
        + 0.10 * pct_col("dist_ema20_pctile")
    )
    momentum = 0.20 * pct_col("tsi_pctile") + 0.08 * np.tanh(out["price_slope_3"].fillna(0) * 25)
    rs_part = 0.10 * np.tanh(out["rs_bench_slope_5"].fillna(0) * 25)
    quality = 1 + 0.15 * pct_col("adx_14_pctile")
    if "dist_vwap_pctile" in out.columns:
        stretch = stretch + 0.10 * centered_pct(out["dist_vwap_pctile"].fillna(0.5))
    out["uo_base"] = (stretch + momentum + rs_part) * quality
    out["uo"] = ema(out["uo_base"], fast) - ema(out["uo_base"], slow)
    out["uo_signal"] = ema(out["uo"], sig)
    out["uo_hist"] = out["uo"] - out["uo_signal"]
    out["uo_gap"] = out["uo_hist"]
    out["uo_slope_1"] = out["uo"].diff(1)
    out["uo_slope_3"] = out["uo"].diff(3)
    out["uo_pctile"] = rolling_percentile(out["uo"], 120 if timeframe_name in {"proxy_hourly", "proxy_2hour"} else 252)
    return out

def enrich_price_features(df: pd.DataFrame, timeframe_name: str, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["ema_10"] = ema(out["Close"], 10)
    out["ema_20"] = ema(out["Close"], 20)
    out["sma_50"] = sma(out["Close"], 50)
    out["atr_14"] = atr(out, 14)
    out["atr_stretch"] = (out["Close"] - out["ema_20"]) / out["atr_14"].replace(0, np.nan)
    out["rsi_14"] = rsi(out["Close"], 14)
    out["rsi_slope_3"] = slope(out["rsi_14"], 3)
    out["cci_20"] = cci(out, 20)
    out["cci_slope_3"] = slope(out["cci_20"], 3)
    out["tsi"], out["tsi_signal"] = tsi(out["Close"], 25, 13, 7)
    out["tsi_gap"] = out["tsi"] - out["tsi_signal"]
    out["tsi_slope_3"] = slope(out["tsi"], 3)
    out["pct_b"] = bollinger_pct_b(out["Close"], 20, 2)
    out["adx_14"] = adx(out, 14)
    out["price_slope_3"] = slope(out["Close"], 3)
    out["dist_ema20_pct"] = (out["Close"] / out["ema_20"]) - 1
    out["volume_ma_20"] = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_ma_20"].replace(0, np.nan)
    out["close_in_range"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["High"] - out[["Close", "Open"]].max(axis=1)) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["candle_score"] = 50 + out["upper_wick_pct"].fillna(0) * 30 - (out["close_in_range"].fillna(0.5) - 0.5) * 20
    out["dist_vwap_pct"] = np.nan
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(out.index).ffill()
        out["rs_vs_benchmark"] = out["Close"] / aligned
        out["rs_bench_slope_5"] = slope(out["rs_vs_benchmark"], 5)
    else:
        out["rs_vs_benchmark"] = 1.0
        out["rs_bench_slope_5"] = 0.0
    win = 120 if timeframe_name in {"proxy_hourly", "proxy_2hour"} else 252
    for col in ["rsi_14", "cci_20", "tsi", "pct_b", "atr_stretch", "adx_14", "dist_ema20_pct", "volume_ratio", "dist_vwap_pct"]:
        if col in out.columns:
            out[f"{col}_pctile"] = rolling_percentile(out[col], win)
    out = add_ultimate_oscillator(out, timeframe_name)
    return out

def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("W-FRI").agg(agg).dropna(how="any")

def build_proxy_from_daily(df: pd.DataFrame, proxy_mode: str) -> Tuple[pd.DataFrame, str, str]:
    label = "Hourly Proxy" if proxy_mode == "hourly" else "2-Hour Proxy"
    timeframe_name = "proxy_hourly" if proxy_mode == "hourly" else "proxy_2hour"
    return df.copy(), timeframe_name, label

# -----------------------------
# ANALYSIS LOGIC
# -----------------------------
def compute_distance_to_cross(row: pd.Series, frame: pd.DataFrame) -> Dict[str, float]:
    if row is None or row.empty or frame is None or frame.empty:
        return {"gap": np.nan, "abs_gap": np.nan, "range_pct": np.nan}
    gap = float(row.get("uo", np.nan) - row.get("uo_signal", np.nan))
    abs_gap = abs(gap) if pd.notna(gap) else np.nan
    recent = (frame["uo"] - frame["uo_signal"]).dropna().tail(40)
    if recent.empty:
        range_pct = np.nan
    else:
        denom = max(float(recent.abs().max()), 1e-9)
        range_pct = float(np.clip(abs_gap / denom, 0, 1) * 100) if pd.notna(abs_gap) else np.nan
    return {"gap": gap, "abs_gap": abs_gap, "range_pct": range_pct}

def compute_state_severity(frame: pd.DataFrame, row: pd.Series) -> Dict[str, float]:
    if row is None or row.empty or frame is None or frame.empty:
        return {
            "uo_slope_1": np.nan, "uo_slope_3": np.nan, "bars_above_80": 0, "bars_above_90": 0,
            "bars_below_20": 0, "bars_below_10": 0, "signal_catchup": np.nan, "lower_high_div": 0,
            "higher_low_repair": 0, "late_cycle_flag": 0, "early_repair_flag": 0,
        }
    gap = (frame["uo"] - frame["uo_signal"]).dropna()
    uo = frame["uo"].dropna()
    pct = frame["uo_pctile"].dropna()
    close = frame["Close"].dropna() if "Close" in frame.columns else pd.Series(dtype=float)

    def count_recent(cond: pd.Series) -> int:
        vals = cond.fillna(False).astype(bool).tolist()
        c = 0
        for ok in reversed(vals):
            if ok: c += 1
            else: break
        return c

    bars_above_80 = count_recent(pct > 0.80)
    bars_above_90 = count_recent(pct > 0.90)
    bars_below_20 = count_recent(pct < 0.20)
    bars_below_10 = count_recent(pct < 0.10)

    signal_catchup = np.nan
    if len(gap) >= 4:
        signal_catchup = float(gap.iloc[-1] - gap.iloc[-4])

    lower_high_div = 0
    higher_low_repair = 0
    if len(uo) >= 8 and len(close) >= 8:
        recent_uo = uo.tail(8)
        recent_close = close.reindex(recent_uo.index).dropna()
        if len(recent_close) >= 6:
            uo_now = float(recent_uo.iloc[-1])
            uo_prev_peak = float(recent_uo.iloc[:-2].max())
            px_now = float(recent_close.iloc[-1])
            px_prev_peak = float(recent_close.iloc[:-2].max())
            lower_high_div = int(px_now >= px_prev_peak * 0.998 and uo_now < uo_prev_peak)

            uo_now_low = float(recent_uo.iloc[-1])
            uo_prev_low = float(recent_uo.iloc[:-2].min())
            px_now_low = float(recent_close.iloc[-1])
            px_prev_low = float(recent_close.iloc[:-2].min())
            higher_low_repair = int(px_now_low <= px_prev_low * 1.002 and uo_now_low > uo_prev_low)

    late_cycle_flag = int(bars_above_90 >= 4 and float(row.get("uo_slope_3", 0.0)) <= 0 and (signal_catchup if pd.notna(signal_catchup) else 0) < 0)
    early_repair_flag = int(bars_below_10 >= 3 and float(row.get("uo_slope_3", 0.0)) >= 0 and (signal_catchup if pd.notna(signal_catchup) else 0) > 0)

    return {
        "uo_slope_1": float(row.get("uo_slope_1", np.nan)),
        "uo_slope_3": float(row.get("uo_slope_3", np.nan)),
        "bars_above_80": bars_above_80,
        "bars_above_90": bars_above_90,
        "bars_below_20": bars_below_20,
        "bars_below_10": bars_below_10,
        "signal_catchup": signal_catchup,
        "lower_high_div": lower_high_div,
        "higher_low_repair": higher_low_repair,
        "late_cycle_flag": late_cycle_flag,
        "early_repair_flag": early_repair_flag,
    }

def recommendation_from_state(call: str, severity: Dict[str, float], timeframe: str) -> str:
    if call == "CALL":
        if severity.get("late_cycle_flag", 0) or severity.get("lower_high_div", 0):
            return "WAIT, aging uptrend" if timeframe == "daily" else "CALL, but extended"
        if severity.get("bars_above_90", 0) >= 6:
            return "CALL, but extended"
        return "CALL"
    if call == "PUT":
        if severity.get("bars_above_90", 0) >= 3 or severity.get("lower_high_div", 0):
            return "PUT setup forming" if timeframe != "weekly" else "PUT"
        return "PUT"
    if call == "NEUTRAL":
        if severity.get("late_cycle_flag", 0):
            return "WAIT, aging uptrend"
        if severity.get("early_repair_flag", 0) or severity.get("higher_low_repair", 0):
            return "WAIT, repair forming"
        return "NEUTRAL / mixed"
    if call == "AVOID CHASE":
        return "WAIT, too extended"
    return call

def classify_timeframe_call(row: pd.Series, timeframe: str) -> Tuple[str, float, str]:
    if row is None or row.empty:
        return "NO DATA", 0.0, "No data"
    uo_pct = row.get("uo_pctile", 0.5)
    uo_gap = row.get("uo_gap", 0.0)
    uo_slope = row.get("uo_slope_3", 0.0)
    rsi_val = row.get("rsi_14", 50.0)
    cci_val = row.get("cci_20", 0.0)
    tsi_gap = row.get("tsi_gap", 0.0)
    pct_b = row.get("pct_b", 0.5)
    adx_val = row.get("adx_14", 15.0)
    dist_ema = row.get("dist_ema20_pct", 0.0)

    if uo_pct > 0.80 and (uo_slope < 0 or uo_gap < 0 or tsi_gap < 0) and rsi_val > 70:
        conf = min(95.0, 55 + (uo_pct - 0.80) * 150)
        return "PUT", conf, "Rolling from elevated zone"
    if timeframe in {"proxy_hourly", "proxy_2hour"} and rsi_val > 75 and cci_val > 90 and pct_b > 0.90 and (uo_gap <= 0 or tsi_gap <= 0):
        return "PUT", 78.0, "Proxy overheated and rolling"
    if timeframe in {"proxy_hourly", "proxy_2hour"} and rsi_val > 75 and cci_val
