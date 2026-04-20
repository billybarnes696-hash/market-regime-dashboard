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
    if timeframe in {"proxy_hourly", "proxy_2hour"} and rsi_val > 75 and cci_val > 90 and pct_b > 0.90 and adx_val > 25 and uo_gap > 0:
        return "AVOID CHASE", 72.0, "Pinned continuation risk"
    if uo_pct < 0.20 and (uo_slope > 0 or uo_gap > 0 or tsi_gap > 0) and rsi_val < 35:
        conf = min(95.0, 55 + (0.20 - uo_pct) * 150)
        return "CALL", conf, "Turning up from washed-out zone"
    if uo_gap > 0 and uo_slope > 0 and dist_ema > -0.02:
        return "CALL", 62.0 + max(0.0, min(15.0, (uo_pct - 0.5) * 20)), "Composite rising above signal"
    if uo_gap < 0 and uo_slope < 0 and dist_ema < 0.02:
        return "PUT", 62.0 + max(0.0, min(15.0, (0.5 - uo_pct) * 20)), "Composite below signal and falling"
    if abs(uo_gap) < 0.02:
        return "NEUTRAL", 45.0, "Near signal-line equilibrium"
    return "NEUTRAL", 50.0, "Mixed state"

def combine_calls(proxy_call: str, daily_call: str, weekly_call: str) -> Tuple[str, str]:
    score_map = {"CALL": 1.0, "PUT": -1.0, "NEUTRAL": 0.0, "AVOID CHASE": -0.35, "NO DATA": 0.0}
    score = 0.30 * score_map.get(proxy_call, 0.0) + 0.45 * score_map.get(daily_call, 0.0) + 0.25 * score_map.get(weekly_call, 0.0)
    if proxy_call in {"PUT", "AVOID CHASE"} and daily_call == "CALL" and weekly_call == "CALL":
        return "WAIT / PROXY TOO HOT", "Bullish trend, but short-term timing is poor"
    if score >= 0.55:
        return "CALL", "Timeframes supportive"
    if score <= -0.55:
        return "PUT", "Timeframes lean bearish"
    if daily_call == "CALL" and weekly_call == "CALL":
        return "CALL ON PULLBACK", "Higher timeframe trend constructive, but entry needs reset"
    return "NEUTRAL", "Mixed timeframe signals"

def align_asof(index: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    try:
        if index.tz is not None and ts.tzinfo is None:
            return ts.tz_localize(index.tz)
        if index.tz is None and ts.tzinfo is not None:
            return ts.tz_localize(None)
    except Exception:
        pass
    return ts

def slice_asof(df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    end_ts = align_asof(df.index, pd.Timestamp(analysis_date) + pd.Timedelta(hours=23, minutes=59, seconds=59))
    return df.loc[df.index <= end_ts].copy()

def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [1, 2, 5]:
        out[f"fwd_ret_{n}"] = out["Close"].shift(-n) / out["Close"] - 1
    return out

ANALOG_FEATURES = [
    "uo_pctile", "uo_gap", "uo_slope_1", "uo_slope_3",
    "rsi_14_pctile", "rsi_14", "cci_20_pctile", "cci_20",
    "tsi_pctile", "tsi", "tsi_gap", "pct_b_pctile", "pct_b",
    "atr_stretch_pctile", "atr_stretch", "dist_ema20_pctile", "dist_ema20_pct",
    "rs_bench_slope_5", "adx_14_pctile", "adx_14", "candle_score",
]

ANALOG_WEIGHTS = {
    "uo_pctile": 1.4, "uo_gap": 1.5, "uo_slope_1": 1.2, "uo_slope_3": 1.4,
    "rsi_14_pctile": 0.9, "rsi_14": 0.7, "cci_20_pctile": 1.0, "cci_20": 0.8,
    "tsi_pctile": 1.0, "tsi": 0.8, "tsi_gap": 1.2, "pct_b_pctile": 0.9, "pct_b": 0.8,
    "atr_stretch_pctile": 0.8, "atr_stretch": 0.7, "dist_ema20_pctile": 0.8, "dist_ema20_pct": 0.7,
    "rs_bench_slope_5": 0.8, "adx_14_pctile": 0.7, "adx_14": 0.6, "candle_score": 0.5,
}

def find_analogs(frame: pd.DataFrame, current_ts: pd.Timestamp, n: int = 25, exclusion_bars: int = 10) -> pd.DataFrame:
    if frame is None or frame.empty or current_ts not in frame.index:
        return pd.DataFrame()
    enriched = add_forward_returns(frame)
    use = [c for c in ANALOG_FEATURES if c in enriched.columns]
    if len(use) < 8:
        return pd.DataFrame()
    current_pos = enriched.index.get_loc(current_ts)
    pool = enriched.iloc[:max(0, current_pos - exclusion_bars)].copy()
    pool = pool.dropna(subset=use + ["fwd_ret_1", "fwd_ret_2", "fwd_ret_5"])
    if len(pool) < max(60, n + 20):
        return pd.DataFrame()
    current = enriched.loc[current_ts, use].astype(float)
    X = pool[use].astype(float)
    std = X.std().replace(0, np.nan)
    z = (X - current) / std
    weights = np.array([ANALOG_WEIGHTS.get(c, 1.0) for c in use], dtype=float)
    zw = z.fillna(0.0).to_numpy() * weights
    pool["distance"] = np.sqrt((zw ** 2).sum(axis=1))
    pool["similarity"] = 1 / (1 + pool["distance"])
    # Filter: Only keep matches where distance is reasonably low (tighter quality control)
    pool = pool[pool["distance"] < 4.5] 
    return pool.nsmallest(n, "distance").copy()

def summarize_analogs(analogs: pd.DataFrame) -> Dict[str, float]:
    if analogs is None or analogs.empty:
        return {}
    w = analogs["similarity"].fillna(1.0)
    out: Dict[str, float] = {"n": float(len(analogs))}
    for n in [1, 2, 5]:
        col = f"fwd_ret_{n}"
        vals = analogs[col].fillna(0)
        out[f"ret_{n}_median"] = float(vals.median())
        out[f"ret_{n}_mean_w"] = float(np.average(vals, weights=w))
        out[f"ret_{n}_p_up"] = float(np.average((vals > 0).astype(float), weights=w))
        out[f"ret_{n}_p_down"] = float(np.average((vals < 0).astype(float), weights=w))
    return out

def analog_bias(summary: Dict[str, float]) -> Tuple[str, float]:
    if not summary:
        return "n/a", 0.0
    up2 = summary.get("ret_2_p_up", 0.5)
    dn2 = summary.get("ret_2_p_down", 0.5)
    mean2 = summary.get("ret_2_mean_w", 0.0)
    if up2 >= 0.62 and mean2 > 0:
        return "bullish", min(1.0, (up2 - 0.5) * 3 + max(0.0, mean2 * 30))
    if dn2 >= 0.62 and mean2 < 0:
        return "bearish", min(1.0, (dn2 - 0.5) * 3 + max(0.0, -mean2 * 30))
    return "mixed", abs(up2 - dn2)

def final_recommendation(combined_call: str, proxy_reco: str, daily_reco: str, analog_summary: Dict[str, float]) -> str:
    bias, strength = analog_bias(analog_summary)
    if combined_call in {"WAIT / PROXY TOO HOT", "NEUTRAL"}:
        if bias == "bullish" and strength >= 0.45:
            return "WAIT / bullish analogs"
        if bias == "bearish" and strength >= 0.45:
            return "PUT setup forming"
        return proxy_reco if proxy_reco != "NEUTRAL / mixed" else "NEUTRAL / mixed"
    if combined_call == "CALL":
        if "aging" in daily_reco.lower() and bias == "bearish" and strength >= 0.45:
            return "WAIT, aging uptrend"
        if bias == "bullish":
            return "CALL" if strength < 0.45 else "CALL / analogs supportive"
        if bias == "bearish" and strength >= 0.5:
            return "CALL, but extended"
        return daily_reco if daily_reco != "CALL" else "CALL"
    if combined_call == "PUT":
        if bias == "bullish" and strength >= 0.5:
            return "WAIT, bearish state but bullish analogs"
        return "PUT" if bias != "bearish" else "PUT / analogs confirm"
    return combined_call

def plot_dashboard(symbol: str, proxy_df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, asof_date: pd.Timestamp, proxy_label: str) -> None:
    fig = make_subplots(
        rows=4, cols=1, vertical_spacing=0.06,
        subplot_titles=[f"{symbol} Daily Price (as of {pd.Timestamp(asof_date).date()})", f"{proxy_label} Oscillator", "Daily Ultimate Oscillator", "Weekly Ultimate Oscillator"],
        row_heights=[0.42, 0.19, 0.19, 0.20],
    )
    if not daily_df.empty:
        d = daily_df.tail(220)
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Daily"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["ema_20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["sma_50"], name="SMA50", line=dict(color="blue")), row=1, col=1)
    for row_num, frame in zip([2, 3, 4], [proxy_df.tail(220), daily_df.tail(220), weekly_df.tail(150)]):
        if frame.empty:
            continue
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo"], name=f"UO {row_num}", line=dict(color="red", width=2)), row=row_num, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo_signal"], name=f"Signal {row_num}", line=dict(color="black", width=1)), row=row_num, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row_num, col=1)
    fig.update_layout(height=950, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, width='stretch')

# -----------------------------
# PARALLEL WORKER
# -----------------------------
def process_symbol_task(sym: str, data_map: Dict[str, pd.DataFrame], bench_df: pd.DataFrame, proxy_mode: str, analysis_mode: str, analysis_date_val, alpha_key: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    try:
        if sym not in data_map or data_map[sym].empty:
            # Fallback try for single fetch if bulk missed it
            df_fb = fetch_alpha_vantage_daily(sym, alpha_key)
            if df_fb.empty:
                return None, None
            data_map[sym] = df_fb
        
        daily_raw = data_map[sym]
        proxy_raw, proxy_timeframe, proxy_label = build_proxy_from_daily(daily_raw, proxy_mode)
        weekly_raw = resample_weekly(daily_raw)

        proxy_df = enrich_price_features(proxy_raw, proxy_timeframe, bench_df)
        daily_df = enrich_price_features(daily_raw, "daily", bench_df)
        weekly_df = enrich_price_features(weekly_raw, "weekly", bench_df)

        asof = pd.Timestamp.today().normalize() if analysis_mode == "Current" else pd.Timestamp(analysis_date_val)
        proxy_view = slice_asof(proxy_df, asof)
        daily_view = slice_asof(daily_df, asof)
        weekly_view = slice_asof(weekly_df, asof)

        if daily_view.empty:
            return None, None

        proxy_row = proxy_view.iloc[-1] if not proxy_view.empty else pd.Series(dtype=float)
        daily_row = daily_view.iloc[-1]
        weekly_row = weekly_view.iloc[-1] if not weekly_view.empty else pd.Series(dtype=float)

        proxy_call, proxy_conf, proxy_reason = classify_timeframe_call(proxy_row, proxy_timeframe)
        proxy_cross = compute_distance_to_cross(proxy_row, proxy_view)
        daily_call, daily_conf, daily_reason = classify_timeframe_call(daily_row, "daily")
        weekly_call, weekly_conf, weekly_reason = classify_timeframe_call(weekly_row, "weekly")
        combined_call, combined_reason = combine_calls(proxy_call, daily_call, weekly_call)
        
        proxy_severity = compute_state_severity(proxy_view, proxy_row)
        daily_severity = compute_state_severity(daily_view, daily_row)
        weekly_severity = compute_state_severity(weekly_view, weekly_row)
        
        proxy_reco = recommendation_from_state(proxy_call, proxy_severity, proxy_timeframe)
        daily_reco = recommendation_from_state(daily_call, daily_severity, "daily")
        weekly_reco = recommendation_from_state(weekly_call, weekly_severity, "weekly")
        
        analogs = find_analogs(daily_df, daily_row.name, n=25)
        analog_summary = summarize_analogs(analogs)
        combined_reco = final_recommendation(combined_call, proxy_reco, daily_reco, analog_summary)

        # Row for Table
        row_data = {
            "Symbol": sym,
            "Status": "OK",
            "As Of": str(daily_row.name.date()),
            "Close": round(float(daily_row.get("Close", np.nan)), 2),
            "Proxy Mode": proxy_label,
            "Proxy Call": proxy_call,
            "Daily Call": daily_call,
            "Weekly Call": weekly_call,
            "Combined": combined_call,
            "Proxy Recommendation": proxy_reco,
            "Daily Recommendation": daily_reco,
            "Weekly Recommendation": weekly_reco,
            "Combined Recommendation": combined_reco,
            "Proxy %ile": round(float(proxy_row.get("uo_pctile", np.nan)) * 100, 1) if not proxy_row.empty else np.nan,
            "Cross Dist %": round(float(proxy_cross.get("range_pct", np.nan)), 1) if proxy_cross else np.nan,
            "Daily %ile": round(float(daily_row.get("uo_pctile", np.nan)) * 100, 1),
            "Weekly %ile": round(float(weekly_row.get("uo_pctile", np.nan)) * 100, 1) if not weekly_row.empty else np.nan,
            "Candle Score": round(float(daily_row.get("candle_score", np.nan)), 1),
            "RSI14": round(float(daily_row.get("rsi_14", np.nan)), 1),
            "CCI20": round(float(daily_row.get("cci_20", np.nan)), 1),
            "Analog N": int(analog_summary.get("n", 0)) if analog_summary else 0,
            "Analog 2d Med": round(float(analog_summary.get("ret_2_median", np.nan)) * 100, 2) if analog_summary else np.nan,
            "Analog 2d Up %": round(float(analog_summary.get("ret_2_p_up", np.nan)) * 100, 1) if analog_summary else np.nan,
        }

        # Detail Dict
        detail_data = {
            "proxy": proxy_view,
            "daily": daily_view,
            "weekly": weekly_view,
            "proxy_call": (proxy_call, proxy_conf, proxy_reason),
            "proxy_label": proxy_label,
            "proxy_cross": proxy_cross,
            "daily_call": (daily_call, daily_conf, daily_reason),
            "weekly_call": (weekly_call, weekly_conf, weekly_reason),
            "combined": (combined_call, combined_reason),
            "severity": {"proxy": proxy_severity, "daily": daily_severity, "weekly": weekly_severity},
            "recommendation": {"proxy": proxy_reco, "daily": daily_reco, "weekly": weekly_reco, "combined": combined_reco},
            "analogs": analogs,
            "analog_summary": analog_summary,
            "asof": asof,
        }
        return row_data, detail_data

    except Exception as e:
        return {"Symbol": sym, "Status": f"Error: {str(e)[:20]}", "Combined Recommendation": "ERROR"}, None

# -----------------------------
# APP MAIN
# -----------------------------
st.title("📈 Stable Market Engine Final")
st.caption("True Percentile Rank | Batch Fetching | Disk Cache | Parallel Processing")

with st.sidebar:
    st.header("Inputs")
    manual_symbols = st.text_area("Paste tickers (comma or line separated)", value="SMH, QQQ, INTC, NVDA, AMD, TSLA, META", height=110)
    alpha_vantage_key = st.text_input("Alpha Vantage API key (optional fallback)", type="password")

    st.header("Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=0)
    history_years = st.selectbox("Historical years", [3, 5, 10], index=1)
    analysis_mode = st.radio("Analysis mode", ["Current", "Historical"], index=0)
    default_date = pd.Timestamp.today().date()
    analysis_date = st.date_input("Calendar lookback", value=default_date, disabled=(analysis_mode == "Current"))
    proxy_mode = st.radio("Tactical proxy", ["hourly", "2hour"], index=0)
    run_analysis = st.button("Run Analysis", type="primary", width='stretch')

if not run_analysis:
    st.stop()

symbols = [s.strip().upper() for s in manual_symbols.replace("\n", ",").split(",") if s.strip()]
symbols = list(dict.fromkeys(symbols))
if not symbols:
    st.error("Provide at least one symbol.")
    st.stop()

# 1. Bulk Fetch Data
# Combine symbols + benchmark to ensure we have everything in one go
all_fetch_symbols = list(set(symbols + [benchmark]))
all_data_map = fetch_yahoo_batch(all_fetch_symbols, history_years)

if benchmark not in all_data_map or all_data_map[benchmark].empty:
    st.error(f"Could not fetch benchmark {benchmark}. Check cache or network.")
    st.stop()

benchmark_df = all_data_map[benchmark]

# 2. Parallel Processing
rows: List[Dict] = []
detail: Dict[str, Dict] = {}
progress = st.progress(0.0)
status_text = st.empty()

# Use ThreadPoolExecutor to process indicators in parallel
# max_workers=4 is safe; indicator calculation is CPU bound but Pandas releases GIL
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(process_symbol_task, s, all_data_map, benchmark_df, proxy_mode, analysis_mode, analysis_date, alpha_vantage_key): s 
        for s in symbols
    }
    
    for i, future in enumerate(as_completed(futures)):
        sym = futures[future]
        status_text.text(f"Processing {sym} ({i+1}/{len(symbols)})...")
        progress.progress((i + 1) / len(symbols))
        
        try:
            row, det = future.result()
            if row:
                rows.append(row)
            if det:
                detail[sym] = det
        except Exception as e:
            rows.append({"Symbol": sym, "Status": "Crash", "Combined Recommendation": "ERROR"})

progress.empty()
status_text.empty()

# 3. Display Results
results_df = pd.DataFrame(rows)
st.subheader("Ranked Results")
if results_df.empty:
    st.warning("No results generated.")
    st.stop()

# Sort by Daily Percentile (highest momentum/extension first)
if "Daily %ile" in results_df.columns:
    results_df = results_df.sort_values("Daily %ile", ascending=False)

st.dataframe(results_df, width='stretch', hide_index=True)
st.download_button("Download results CSV", results_df.to_csv(index=False).encode("utf-8"), "stable_market_engine_final.csv", "text/csv")

valid_symbols = results_df.loc[results_df["Status"] == "OK", "Symbol"].tolist()
if not valid_symbols:
    st.stop()

st.subheader("Detailed Analysis")
selected = st.selectbox("Select symbol", valid_symbols)
item = detail[selected]

# Metric Columns
proxy_call, _, proxy_reason = item["proxy_call"]
proxy_label = item.get("proxy_label", "Proxy")
proxy_cross = item.get("proxy_cross", {})
daily_call, _, daily_reason = item["daily_call"]
weekly_call, _, weekly_reason = item["weekly_call"]
combined_call, combined_reason = item["combined"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(proxy_label, proxy_call)
c2.metric("Daily", daily_call)
c3.metric("Weekly", weekly_call)
c4.metric("Combined", combined_call)
st.markdown(f"**{proxy_label} reason:** {proxy_reason}")
st.markdown(f"**Daily reason:** {daily_reason}")
st.markdown(f"**Weekly reason:** {weekly_reason}")
st.markdown(f"**Combined read:** {combined_reason}")

reco = item.get("recommendation", {})
sev = item.get("severity", {})
st.markdown(f"**Recommendation:** {reco.get('combined', combined_call)}")

# Cross Stats
cd1, cd2, cd3 = st.columns(3)
cd1.metric("Distance to cross", f"{proxy_cross.get('abs_gap', np.nan):.4f}" if pd.notna(proxy_cross.get('abs_gap', np.nan)) else "n/a")
gap_label = "Above signal" if proxy_cross.get("gap", np.nan) >= 0 else "Below signal" if pd.notna(proxy_cross.get("gap", np.nan)) else "n/a"
cd2.metric("Cross gap sign", gap_label)
cd3.metric("Cross distance %", f"{proxy_cross.get('range_pct', np.nan):.1f}%" if pd.notna(proxy_cross.get("range_pct", np.nan)) else "n/a")

# Severity Stats
sv1, sv2, sv3, sv4 = st.columns(4)
proxy_sev = sev.get("proxy", {})
sv1.metric("Bars > 90", str(proxy_sev.get("bars_above_90", 0)))
sv2.metric("Bars < 10", str(proxy_sev.get("bars_below_10", 0)))
sv3.metric("Late-cycle", "Yes" if proxy_sev.get("late_cycle_flag", 0) else "No")
sv4.metric("Divergence", "Yes" if proxy_sev.get("lower_high_div", 0) else ("Repair" if proxy_sev.get("higher_low_repair", 0) else "No"))

# As-Of Table
proxy_last = item["proxy"].iloc[-1] if not item["proxy"].empty else pd.Series(dtype=float)
daily_last = item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float)
weekly_last = item["weekly"].iloc[-1] if not item["weekly"].empty else pd.Series(dtype=float)

st.markdown("### As-of values")
asof_table = pd.DataFrame(
    [
        {
            "Frame": proxy_label,
            "Date": str(proxy_last.name.date()) if not proxy_last.empty else "n/a",
            "Price": round(float(proxy_last.get("Close", np.nan)), 2) if not proxy_last.empty else np.nan,
            "UO": round(float(proxy_last.get("uo", np.nan)), 4) if not proxy_last.empty else np.nan,
            "Signal": round(float(proxy_last.get("uo_signal", np.nan)), 4) if not proxy_last.empty else np.nan,
            "Percentile": round(float(proxy_last.get("uo_pctile", np.nan)) * 100, 1) if not proxy_last.empty else np.nan,
            "Candle Score": round(float(proxy_last.get("candle_score", np.nan)), 1) if not proxy_last.empty else np.nan,
            "Cross Gap": round(float(proxy_cross.get("gap", np.nan)), 4) if pd.notna(proxy_cross.get("gap", np.nan)) else np.nan,
            "Cross Dist %": round(float(proxy_cross.get("range_pct", np.nan)), 1) if pd.notna(proxy_cross.get("range_pct", np.nan)) else np.nan,
            "Recommendation": item.get("recommendation", {}).get("proxy", proxy_call),
        },
        {
            "Frame": "Daily",
            "Date": str(daily_last.name.date()) if not daily_last.empty else "n/a",
            "Price": round(float(daily_last.get("Close", np.nan)), 2) if not daily_last.empty else np.nan,
            "UO": round(float(daily_last.get("uo", np.nan)), 4) if not daily_last.empty else np.nan,
            "Signal": round(float(daily_last.get("uo_signal", np.nan)), 4) if not daily_last.empty else np.nan,
            "Percentile": round(float(daily_last.get("uo_pctile", np.nan)) * 100, 1) if not daily_last.empty else np.nan,
            "Candle Score": round(float(daily_last.get("candle_score", np.nan)), 1) if not daily_last.empty else np.nan,
            "Recommendation": item.get("recommendation", {}).get("daily", daily_call),
        },
        {
            "Frame": "Weekly",
            "Date": str(weekly_last.name.date()) if not weekly_last.empty else "n/a",
            "Price": round(float(weekly_last.get("Close", np.nan)), 2) if not weekly_last.empty else np.nan,
            "UO": round(float(weekly_last.get("uo", np.nan)), 4) if not weekly_last.empty else np.nan,
            "Signal": round(float(weekly_last.get("uo_signal", np.nan)), 4) if not weekly_last.empty else np.nan,
            "Percentile": round(float(weekly_last.get("uo_pctile", np.nan)) * 100, 1) if not weekly_last.empty else np.nan,
            "Candle Score": round(float(weekly_last.get("candle_score", np.nan)), 1) if not weekly_last.empty else np.nan,
            "Recommendation": item.get("recommendation", {}).get("weekly", weekly_call),
        },
    ]
)
st.dataframe(asof_table, width='stretch', hide_index=True)

# Analogs
st.markdown("### Historical analogs")
analog_summary = item.get("analog_summary", {})
if analog_summary:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Analog count", str(int(analog_summary.get("n", 0))))
    a2.metric("2d median", f"{analog_summary.get('ret_2_median', np.nan) * 100:.2f}%" if pd.notna(analog_summary.get('ret_2_median', np.nan)) else "n/a")
    a3.metric("2d up %", f"{analog_summary.get('ret_2_p_up', np.nan) * 100:.1f}%" if pd.notna(analog_summary.get('ret_2_p_up', np.nan)) else "n/a")
    a4.metric("5d median", f"{analog_summary.get('ret_5_median', np.nan) * 100:.2f}%" if pd.notna(analog_summary.get('ret_5_median', np.nan)) else "n/a")
    analogs = item.get("analogs", pd.DataFrame())
    if analogs is not None and not analogs.empty:
        show_cols = [c for c in ["Close", "uo", "uo_signal", "distance", "similarity", "fwd_ret_1", "fwd_ret_2", "fwd_ret_5"] if c in analogs.columns]
        show = analogs[show_cols].head(12).copy()
        try:
            show.index = show.index.strftime("%Y-%m-%d")
        except Exception:
            pass
        st.dataframe(show, width='stretch')
else:
    st.caption("No analog set available for this as-of date.")

plot_dashboard(selected, item["proxy"], item["daily"], item["weekly"], item["asof"], proxy_label)
