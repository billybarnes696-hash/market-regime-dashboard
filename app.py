#!/usr/bin/env python3
"""
Stable Market Engine v12 — TSI Cross Dashboard (Multi-Symbol)
✅ TSI Logic: Cross, Heat, Regime, Exhaustion, Divergence
✅ Dashboard: Mega View, Range Tabs, Traffic Lights, Diagnostics
✅ Data: Alpaca (Real 1H/2H + Daily) with Smooth Visualization
✅ Multi-Symbol: Batch processing, ranking, detailed drill-down
"""

from __future__ import annotations
import io
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# -----------------------------
# CONFIG & SETUP
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store_alpaca"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NY_TZ = "America/New_York"

st.set_page_config(
    page_title="Stable Market Engine v12 — TSI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# UTILITY HELPERS
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def smooth_component(series: pd.Series, span: int) -> pd.Series:
    return ema(series, span) if span and span > 1 else series

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def slope(series: pd.Series, bars: int = 3) -> pd.Series:
    return series.diff(bars) / bars

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

def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0.0)
    pv = typical * vol
    pv_sum = pv.rolling(window, min_periods=max(5, window // 4)).sum()
    v_sum = vol.rolling(window, min_periods=max(5, window // 4)).sum()
    return pv_sum / v_sum.replace(0, np.nan)

def anchored_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    local_idx = pd.to_datetime(df.index)
    if getattr(local_idx, "tz", None) is None:
        local_idx = local_idx.tz_localize(NY_TZ)
    else:
        local_idx = local_idx.tz_convert(NY_TZ)
    day_keys = pd.Series(local_idx.date, index=df.index)
    for _, part in df.groupby(day_keys):
        tp = (part["High"] + part["Low"] + part["Close"]) / 3.0
        pv = tp * part["Volume"].fillna(0)
        cum_vol = part["Volume"].fillna(0).cumsum().replace(0, np.nan)
        out.loc[part.index] = pv.cumsum() / cum_vol
    return out

def normalize_rolling(series: pd.Series, window: int) -> pd.Series:
    lo = series.rolling(window, min_periods=max(20, window // 5)).min()
    hi = series.rolling(window, min_periods=max(20, window // 5)).max()
    out = 100 * (series - lo) / (hi - lo).replace(0, np.nan)
    return out.clip(0, 100)

def recent_divergence(price: pd.Series, osc: pd.Series, lookback: int = 10) -> pd.Series:
    hh_price = price == price.rolling(lookback, min_periods=max(5, lookback // 2)).max()
    hh_osc = osc == osc.rolling(lookback, min_periods=max(5, lookback // 2)).max()
    ll_price = price == price.rolling(lookback, min_periods=max(5, lookback // 2)).min()
    ll_osc = osc == osc.rolling(lookback, min_periods=max(5, lookback // 2)).min()
    bear = hh_price & (~hh_osc) & (price > price.shift(max(2, lookback // 2))) & (osc < osc.shift(max(2, lookback // 2)))
    bull = ll_price & (~ll_osc) & (price < price.shift(max(2, lookback // 2))) & (osc > osc.shift(max(2, lookback // 2)))
    out = pd.Series("None", index=price.index)
    out.loc[bear] = "Bearish"
    out.loc[bull] = "Bullish"
    return out

# -----------------------------
# DATA FETCHING
# -----------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out.columns = [str(c).title() for c in out.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    if len(keep) < 4:
        return pd.DataFrame()
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(NY_TZ).tz_localize(None)
    out = out.sort_index()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    return out.dropna(subset=["Open", "High", "Low", "Close"])

def parse_symbol_csv(uploaded) -> List[str]:
    if uploaded is None:
        return []
    raw = uploaded.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return []
    cols = {str(c).strip().lower(): c for c in df.columns}
    sym_col = cols.get("symbol") or cols.get("ticker") or next(iter(df.columns), None)
    if sym_col is None:
        return []
    return [str(x).strip().upper() for x in df[sym_col].dropna().tolist() if str(x).strip()]

def clean_symbols(text: str, uploaded_symbols: List[str]) -> List[str]:
    symbols: List[str] = []
    if text.strip():
        symbols.extend([s.strip().upper() for s in text.replace("\n", ",").split(",") if s.strip()])
    symbols.extend(uploaded_symbols)
    out = []
    for s in symbols:
        if s and s not in out:
            out.append(s)
    return out

def cache_path(symbol: str, kind: str) -> Path:
    safe = "".join(c for c in symbol if c.isalnum() or c in ".-")
    return CACHE_DIR / f"{safe}_{kind}.parquet"

def is_fresh(path: Path, max_hours: int = 18) -> bool:
    if not path.exists():
        return False
    age = pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
    return age < pd.Timedelta(hours=max_hours)

def alpaca_client(key: str, secret: str) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(key, secret)

def _bars_df_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        if "symbol" in df.index.names:
            try:
                sub = df.xs(symbol, level="symbol").copy()
            except Exception:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    else:
        sub = df.copy()
    if "timestamp" in sub.columns:
        sub = sub.set_index("timestamp")
    return normalize_ohlcv(sub)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpaca_daily_batch(symbols: List[str], years: int, key: str, secret: str, feed: str) -> Dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    data_map: Dict[str, pd.DataFrame] = {}
    missing = []
    for s in symbols:
        p = cache_path(s, "daily")
        if is_fresh(p):
            try:
                data_map[s] = pd.read_parquet(p)
                continue
            except Exception:
                pass
        missing.append(s)
    if not missing:
        return data_map
    client = alpaca_client(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(years=max(years, 5))).normalize().tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=missing, timeframe=TimeFrame.Day, start=start, end=end, adjustment="raw", feed=feed)
    try:
        raw = client.get_stock_bars(req).df
    except Exception:
        return data_map
    for s in missing:
        sub = _bars_df_for_symbol(raw, s)
        if not sub.empty:
            sub.to_parquet(cache_path(s, "daily"))
            data_map[s] = sub
    return data_map

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_alpaca_2hour(symbol: str, months: int, key: str, secret: str, feed: str) -> pd.DataFrame:
    p = cache_path(symbol, "2hour")
    if is_fresh(p, max_hours=6):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    client = alpaca_client(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(months=max(months, 3))).tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(30, TimeFrameUnit.Minute), start=start, end=end, adjustment="raw", feed=feed)
    raw = None
    for attempt in range(3):
        try:
            raw = client.get_stock_bars(req).df
            break
        except Exception:
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(1.5 * (attempt + 1))
    sub = _bars_df_for_symbol(raw, symbol)
    if sub.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(sub.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    sub.index = idx
    sub = sub.between_time("09:30", "16:00")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    pieces = []
    for _, day_df in sub.groupby(sub.index.date):
        if day_df.empty:
            continue
        anchor = pd.Timestamp(day_df.index[0].date()) + pd.Timedelta(hours=9, minutes=30)
        ranges = [
            (anchor, anchor + pd.Timedelta(hours=2)),
            (anchor + pd.Timedelta(hours=2), anchor + pd.Timedelta(hours=4)),
            (anchor + pd.Timedelta(hours=4), anchor + pd.Timedelta(hours=6)),
            (anchor + pd.Timedelta(hours=6), anchor + pd.Timedelta(hours=6, minutes=30)),
        ]
        bins = []
        for start_ts, end_ts in ranges:
            chunk = day_df[(day_df.index >= start_ts) & (day_df.index < end_ts)]
            if chunk.empty:
                continue
            row = pd.DataFrame(
                {"Open": [chunk["Open"].iloc[0]], "High": [chunk["High"].max()], "Low": [chunk["Low"].min()], "Close": [chunk["Close"].iloc[-1]], "Volume": [chunk["Volume"].sum()]},
                index=[start_ts]
            )
            bins.append(row)
        if bins:
            pieces.append(pd.concat(bins))
    out = pd.concat(pieces).sort_index() if pieces else pd.DataFrame()
    out = normalize_ohlcv(out)
    if not out.empty:
        out.to_parquet(p)
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_alpaca_1hour(symbol: str, months: int, key: str, secret: str, feed: str) -> pd.DataFrame:
    p = cache_path(symbol, "1hour")
    if is_fresh(p, max_hours=6):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    client = alpaca_client(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(months=max(months, 3))).tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=start, end=end, adjustment="raw", feed=feed)
    raw = None
    for attempt in range(3):
        try:
            raw = client.get_stock_bars(req).df
            break
        except Exception:
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(1.5 * (attempt + 1))
    sub = _bars_df_for_symbol(raw, symbol)
    if sub.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(sub.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    sub.index = idx
    sub = sub.between_time("09:30", "16:00")
    sub = normalize_ohlcv(sub)
    if not sub.empty:
        sub.to_parquet(p)
    return sub

def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("W-FRI").agg(agg).dropna(how="any")

def time_compressed_proxy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, str, str]:
    out = df.copy()
    if target == "2hour_proxy":
        label = "2-Hour Proxy"
        timeframe_name = "2hour_proxy"
    else:
        label = "Hourly Proxy"
        timeframe_name = "hourly_proxy"
    return out, timeframe_name, label

# -----------------------------
# FEATURE ENGINEERING (TSI LOGIC)
# -----------------------------
def regime_bucket(tsi_heat: float, stretch_heat: float) -> str:
    if pd.isna(tsi_heat):
        return "Neutral"
    if tsi_heat >= 70 and stretch_heat >= 70:
        return "Strong & Extended"
    if tsi_heat >= 60:
        return "Strong"
    if tsi_heat <= 35 and stretch_heat <= 45:
        return "Weak"
    if tsi_heat < 50:
        return "Fading"
    return "Neutral"

def classify_state(row: pd.Series) -> str:
    tsi_val, tsi_sig = row["TSI"], row["TSI_signal"]
    tsi_slope, gap = row["TSI_slope"], row["TSI_gap"]
    heat, price_chg = row["Exhaustion_score"], row["Price_lookback_ret"]
    if tsi_val < tsi_sig:
        if heat >= 72:
            return "PUT · Exhausted, No Price Damage" if abs(price_chg) < 0.004 else "PUT · Exhausted"
        return "PUT · Bearish" if (tsi_slope < 0 or gap < 0) else "NEUTRAL · Transition"
    if tsi_val > tsi_sig:
        if heat <= 28 and tsi_slope > 0:
            return "CALL · Oversold Bull Turn"
        return "CALL · Bullish" if (tsi_slope > 0 or gap > 0) else "NEUTRAL · Transition"
    return "NEUTRAL · Transition"

def enrich_price_features_tsi(df: pd.DataFrame, timeframe_name: str, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    x = df.copy()
    x["EMA10"] = ema(x["Close"], 10)
    x["EMA20"] = ema(x["Close"], 20)
    x["SMA50"] = x["Close"].rolling(50, min_periods=10).mean()
    x["TSI"], x["TSI_signal"] = tsi(x["Close"], 25, 13, 7)
    x["TSI_gap"] = x["TSI"] - x["TSI_signal"]
    x["RSI"] = rsi(x["Close"], 14)
    x["CCI"] = cci(x, 20)
    x["BBPct"] = bollinger_pct_b(x["Close"], 20, 2.0)
    
    # VWAP logic
    if timeframe_name in {"1H", "1H_PROXY", "hourly_proxy", "proxy_smooth_1hour"}:
        x["VWAP"] = anchored_intraday_vwap(x)
        roll_window, div_lb, price_lb = 140, 10, 4
    else:
        x["VWAP"] = rolling_vwap(x, 20)
        roll_window, div_lb, price_lb = 252, 6, 3

    x["Dist_EMA10"] = 100 * (x["Close"] / x["EMA10"] - 1)
    x["Dist_VWAP"] = 100 * (x["Close"] / x["VWAP"] - 1)
    x["Price_lookback_ret"] = x["Close"] / x["Close"].shift(price_lb) - 1

    # Heat & Exhaustion
    x["TSI_heat"] = normalize_rolling(x["TSI"], roll_window)
    x["CCI_heat"] = normalize_rolling(x["CCI"], roll_window)
    x["RSI_heat"] = x["RSI"].clip(0, 100)
    x["BB_heat"] = (x["BBPct"] * 100).clip(0, 100)
    x["Stretch_heat"] = normalize_rolling(x["Dist_EMA10"] + 0.5 * x["Dist_VWAP"].fillna(0), roll_window)
    raw_exhaust = 0.25*x["TSI_heat"] + 0.25*x["CCI_heat"] + 0.15*x["RSI_heat"] + 0.20*x["BB_heat"] + 0.15*x["Stretch_heat"]
    x["Exhaustion_score"] = ema(raw_exhaust, 4)

    x["TSI_slope"] = slope(x["TSI"], 3)
    x["Exhaustion_slope"] = slope(x["Exhaustion_score"], 3)
    x["Divergence"] = recent_divergence(x["Close"], x["TSI"], div_lb)
    x["Regime_bucket"] = [regime_bucket(a, b) for a, b in zip(x["TSI_heat"], x["Stretch_heat"])]
    x["State"] = x.apply(classify_state, axis=1)

    # Map to UO columns for backward compatibility with v11 UI
    x["uo"] = x["TSI"]
    x["uo_signal"] = x["TSI_signal"]
    x["uo_gap"] = x["TSI_gap"]
    x["uo_slope_1"] = slope(x["TSI"], 1)
    x["uo_slope_3"] = x["TSI_slope"]
    x["uo_pctile"] = x["TSI_heat"] / 100.0
    x["uo_decision"] = x["TSI"]  # Decision line matches TSI
    x["uo_signal_decision"] = x["TSI_signal"]
    x["uo_viz"] = ema(x["TSI"], 5)  # Light smoothing for viz
    x["uo_signal_viz"] = ema(x["TSI_signal"], 5)
    x["uo_above_signal_2"] = (x["uo"] > x["uo_signal"]).astype(int).rolling(2, min_periods=2).sum() == 2
    x["uo_below_signal_2"] = (x["uo"] < x["uo_signal"]).astype(int).rolling(2, min_periods=2).sum() == 2
    gap_scale = x["uo_gap"].abs().rolling(40, min_periods=10).max().replace(0, np.nan)
    x["uo_gap_strength"] = (x["uo_gap"].abs() / gap_scale).clip(0, 1)

    # Benchmarks
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(x.index).ffill()
        x["rs_bench_slope_5"] = slope(x["Close"] / aligned, 5)
    else:
        x["rs_bench_slope_5"] = 0.0

    # Pinning flags
    x["tsi_slope_3"] = x["TSI_slope"]
    x["cci_slope_3"] = slope(x["CCI"], 3)
    tsi_flat_thresh = x["tsi_slope_3"].abs().rolling(50, min_periods=10).median().fillna(0.02)
    x["price_change_while_tsi_flat"] = np.where(
        x["tsi_slope_3"].abs() <= tsi_flat_thresh,
        x["Close"].pct_change(3),
        0.0,
    )
    x["pinning_up_flag"] = ((x["price_change_while_tsi_flat"] > 0.012) & (x["cci_slope_3"] < 0) & (x["BBPct"] > 0.75)).astype(float)
    x["pinning_down_flag"] = ((x["price_change_while_tsi_flat"] < -0.012) & (x["cci_slope_3"] > 0) & (x["BBPct"] < 0.25)).astype(float)

    # OB/OS Scores
    x["ob_internal_score"] = (0.45 * (x["RSI"]/100) + 0.25 * (x["TSI_heat"]/100) + 0.15 * (x["CCI_heat"]/100) + 0.15 * (x["BBPct"])).clip(0, 1)
    x["os_internal_score"] = 1 - x["ob_internal_score"]
    x["ob_price_score"] = (0.35 * x["BBPct"] + 0.25 * ((x["Dist_EMA10"] + 1).clip(0.5, 1.5) - 0.5) + 0.2 * ((x["Dist_VWAP"] + 1).clip(0.5, 1.5) - 0.5) + 0.2 * ((x["RSI"]/100))).clip(0, 1)
    x["os_price_score"] = 1 - x["ob_price_score"]
    x["candle_score"] = 50
    
    # Additional v11 expectations
    x["close_in_range"] = (x["Close"] - x["Low"]) / (x["High"] - x["Low"]).replace(0, np.nan)
    x["upper_wick_pct"] = (x["High"] - x[["Close", "Open"]].max(axis=1)) / (x["High"] - x["Low"]).replace(0, np.nan)
    x["atr_14"] = atr(x, 14)
    x["atr_stretch"] = (x["Close"] - x["EMA20"]) / x["atr_14"].replace(0, np.nan)
    x["adx_14"] = adx(x, 14)
    x["price_slope_3"] = slope(x["Close"], 3)
    x["dist_ema20_pct"] = (x["Close"] / x["EMA20"]) - 1
    x["dist_vwap_pct"] = (x["Close"] / x["VWAP"]) - 1
    
    return x

def merge_higher_state(lower_df: pd.DataFrame, higher_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["State", "Regime_bucket", "TSI", "TSI_signal", "TSI_gap", "TSI_slope"]
    if higher_df.empty or lower_df.empty:
        return lower_df.copy()
    lower = lower_df.copy()
    higher = higher_df[cols].copy()
    aligned = higher.reindex(lower.index, method="ffill")
    aligned = aligned.rename(columns={
        "State": "Higher_State",
        "Regime_bucket": "Higher_Regime",
        "TSI": "Higher_TSI",
        "TSI_signal": "Higher_TSI_signal",
        "TSI_gap": "Higher_TSI_gap",
        "TSI_slope": "Higher_TSI_slope",
    })
    return lower.join(aligned)

# -----------------------------
# ANALYSIS & ANALOGS
# -----------------------------
FWD_BARS = {"1H": [1, 3, 6, 12], "2H": [1, 2, 4, 8], "Daily": [1, 2, 5, 10], "Weekly": [1, 2, 4, 8]}
ANALOG_FEATURES_TSI = ["TSI", "TSI_gap", "TSI_slope", "TSI_heat", "Exhaustion_score", "Divergence", "Regime_bucket", "RSI", "CCI", "BBPct", "Dist_EMA10", "Dist_VWAP"]

def analog_summary_tsi(df: pd.DataFrame, tf_key: str) -> Tuple[dict, pd.DataFrame]:
    max_fwd = max(FWD_BARS.get(tf_key, [5]))
    if len(df) <= max_fwd + 20:
        return {}, pd.DataFrame()
    base = df.iloc[:-max_fwd].copy()
    cur = df.iloc[-1]
    mask = base["State"].eq(cur["State"]) & base["Higher_Regime"].eq(cur["Higher_Regime"])
    if cur["Divergence"] in ["Bearish", "Bullish"]:
        mask &= base["Divergence"].eq(cur["Divergence"])
    matches = base.loc[mask].copy()
    if len(matches) < 15:
        matches = base.loc[base["State"].eq(cur["State"]) & base["Higher_Regime"].eq(cur["Higher_Regime"])].copy()
    if matches.empty:
        return {}, pd.DataFrame()
    fwd = pd.DataFrame(index=matches.index)
    for h in FWD_BARS.get(tf_key, [1, 2, 5]):
        fwd[f"ret_{h}"] = df["Close"].shift(-h).reindex(matches.index) / df["Close"].reindex(matches.index) - 1
    return {"sample": int(len(matches))}, fwd

def classify_timeframe_call_tsi(row: pd.Series, timeframe: str) -> Tuple[str, str]:
    # Wrapper to map TSI state to the UI's CALL/PUT/NEUTRAL expectations
    if row is None or row.empty:
        return "NO DATA", "No data"
    state = str(row.get("State", ""))
    if state.startswith("PUT"):
        return "PUT", state
    if state.startswith("CALL"):
        return "CALL", state
    return "NEUTRAL", state

def extreme_state(row: pd.Series) -> Dict[str, float]:
    if row is None or row.empty:
        return {"label": "No data", "ob_internal": np.nan, "ob_price": np.nan, "os_internal": np.nan, "os_price": np.nan}
    return {
        "label": "Neutral",
        "ob_internal": float(row.get("ob_internal_score", 0.5)),
        "ob_price": float(row.get("ob_price_score", 0.5)),
        "os_internal": float(row.get("os_internal_score", 0.5)),
        "os_price": float(row.get("os_price_score", 0.5)),
    }

def render_extreme_table(frame_items: List[Tuple[str, pd.Series]]) -> None:
    rows = []
    for label, row in frame_items:
        stt = extreme_state(row)
        rows.append({
            "Frame": label,
            "State": str(row.get("State", "N/A")),
            "Regime": str(row.get("Regime_bucket", "N/A")),
            "OB internal": round(float(stt["ob_internal"]) * 100, 1),
            "OB price": round(float(stt["ob_price"]) * 100, 1),
            "OS internal": round(float(stt["os_internal"]) * 100, 1),
            "OS price": round(float(stt["os_price"]) * 100, 1),
        })
    st.markdown("### Diagnostics (TSI State & Regime)")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

def render_traffic_lights(frame_items: List[Tuple[str, str, pd.Series]]) -> None:
    st.markdown("### Traffic lights")
    cols = st.columns(len(frame_items))
    for col, (label, call, row) in zip(cols, frame_items):
        if call.startswith("PUT"):
            emoji, sub = "🔴", "Bearish"
        elif call.startswith("CALL"):
            emoji, sub = "🟢", "Bullish"
        else:
            emoji, sub = "🟡", "Neutral"
        
        if row is not None and not row.empty:
            sub = str(row.get("Regime_bucket", sub))
            
        col.markdown(f"<div style='text-align:center;font-size:40px;line-height:1.1'>{emoji}</div>", unsafe_allow_html=True)
        col.markdown(f"**{label}**")
        col.caption(f"{call} · {sub}")

def traffic_state(call: str, row: pd.Series) -> Tuple[str, str]:
    if row is None or row.empty:
        return "⚪", "No data"
    if call == "CALL":
        return "🟢", "Bullish"
    if call == "PUT":
        return "🔴", "Bearish"
    return "🟡", "Neutral"

def frame_warning_message(row: pd.Series, timeframe_label: str, structural_bias: str = "") -> str:
    if row is None or row.empty:
        return f"{timeframe_label}: no data."
    state = str(row.get("State", ""))
    regime = str(row.get("Regime_bucket", ""))
    divergence = str(row.get("Divergence", "None"))
    
    msg = f"{timeframe_label}: {state}."
    if "Exhausted" in state:
        msg += " High reversal risk."
    elif "Oversold" in state and "CALL" in state:
        msg += " Potential bounce."
    
    if structural_bias:
        msg += f" Aligns with {structural_bias.lower()} bias."
    if divergence != "None":
        msg += f" Divergence detected: {divergence}."
    return msg

def distance_to_cross(row: pd.Series, frame: pd.DataFrame) -> Dict[str, float]:
    if row is None or row.empty or frame.empty:
        return {"gap": np.nan, "abs_gap": np.nan, "range_pct": np.nan}
    gap = float(row.get("TSI_gap", 0.0))
    abs_gap = abs(gap)
    recent = frame["TSI_gap"].dropna().tail(40)
    denom = max(float(recent.abs().max()), 1e-9) if not recent.empty else np.nan
    range_pct = (abs_gap / denom * 100) if pd.notna(denom) else np.nan
    return {"gap": gap, "abs_gap": abs_gap, "range_pct": range_pct}

# -----------------------------
# PLOTTING & DASHBOARD
# -----------------------------
def trim_to_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty or not range_key or range_key == "MAX":
        return df.copy()
    end_ts = pd.Timestamp(df.index.max())
    mapping = {
        "2W": pd.DateOffset(weeks=2), "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2),
        "5Y": pd.DateOffset(years=5), "10Y": pd.DateOffset(years=10),
    }
    offset = mapping.get(range_key)
    if offset is None:
        return df.copy()
    start_ts = end_ts - offset
    return df.loc[df.index >= start_ts].copy()

def _price_panel_trace(fig, frame: pd.DataFrame, row: int, title: str) -> None:
    if frame.empty:
        return
    d = frame.copy()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=title), row=row, col=1)
    if "EMA20" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["EMA20"], name=f"{title} EMA20", line=dict(color="orange")), row=row, col=1)
    if "SMA50" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], name=f"{title} SMA50", line=dict(color="blue")), row=row, col=1)

def _osc_panel_trace_tsi(fig, frame: pd.DataFrame, row: int, nm: str) -> None:
    if frame.empty:
        return
    fig.add_trace(go.Scatter(x=frame.index, y=frame.get("TSI", frame.get("uo", [])), name=f"{nm} TSI", line=dict(color="red", width=2.3)), row=row, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame.get("TSI_signal", frame.get("uo_signal", [])), name=f"{nm} Signal", line=dict(color="black", width=1.3)), row=row, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)

def plot_mega_view(symbol: str, hourly_df: pd.DataFrame, tactical_df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, asof_date: pd.Timestamp, tactical_label: str, mega_ranges: Dict[str, str]) -> None:
    fig = make_subplots(
        rows=5, cols=1, vertical_spacing=0.04,
        subplot_titles=[f"{symbol} Price (as of {pd.Timestamp(asof_date).date()})", "1-Hour TSI", f"{tactical_label} TSI", "Daily TSI", "Weekly TSI"],
        row_heights=[0.30, 0.18, 0.18, 0.18, 0.16],
    )
    hourly_plot = trim_to_range(hourly_df, mega_ranges.get("1h", "3M"))
    tactical_plot = trim_to_range(tactical_df, mega_ranges.get("2h", "6M"))
    daily_plot = trim_to_range(daily_df, mega_ranges.get("daily", "1Y"))
    weekly_plot = trim_to_range(weekly_df, mega_ranges.get("weekly", "5Y"))
    base_price = daily_plot if not daily_plot.empty else hourly_plot
    _price_panel_trace(fig, base_price, 1, symbol)
    _osc_panel_trace_tsi(fig, hourly_plot, 2, "Hourly")
    _osc_panel_trace_tsi(fig, tactical_plot, 3, "Tactical")
    _osc_panel_trace_tsi(fig, daily_plot, 4, "Daily")
    _osc_panel_trace_tsi(fig, weekly_plot, 5, "Weekly")
    fig.update_layout(height=1380, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

def plot_single_frame(symbol: str, price_df: pd.DataFrame, osc_df: pd.DataFrame, asof_date: pd.Timestamp, frame_title: str, range_key: str) -> None:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.06, subplot_titles=[f"{symbol} Price (as of {pd.Timestamp(asof_date).date()})", frame_title], row_heights=[0.56, 0.44])
    price_plot = trim_to_range(price_df, range_key)
    osc_plot = trim_to_range(osc_df, range_key)
    _price_panel_trace(fig, price_plot, 1, symbol)
    _osc_panel_trace_tsi(fig, osc_plot, 2, frame_title)
    fig.update_layout(height=760, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

# -----------------------------
# MAIN APP
# -----------------------------
st.title("📈 Stable Market Engine v12 — TSI Dashboard")
st.caption("TSI Cross Logic | Real Alpaca Data | Mega View | Multi-Symbol Ranking")

with st.sidebar:
    st.header("Credentials")
    alpaca_key = st.text_input("Alpaca API key", type="password")
    alpaca_secret = st.text_input("Alpaca API secret", type="password")
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], index=0)
    
    st.header("Input")
    symbols_text = st.text_area("Paste tickers (comma or line separated)", value="QQQ, SMH, INTC, NVDA, AMD, XLF", height=110)
    upload_watchlist = st.file_uploader("Upload results/watchlist CSV", type=["csv"])

    st.header("Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=0)
    history_years = st.selectbox("Historical years", [3, 5, 10], index=1)
    analysis_mode = st.radio("Analysis mode", ["Current", "Historical"], index=0)
    analysis_date = st.date_input("Calendar lookback", value=pd.Timestamp.today().date(), disabled=(analysis_mode == "Current"))
    intraday_source = st.radio("Intraday source", ["proxy", "real"], index=0, format_func=lambda x: {"proxy": "Proxy Smooth (default)", "real": "Real Alpaca"}[x])
    force_refresh = st.checkbox("Force refresh data (clear cache)", value=False)
    run_analysis = st.button("Run Analysis", type="primary", width="stretch")

if not run_analysis:
    st.stop()
if not alpaca_key or not alpaca_secret:
    st.error("Enter Alpaca API key and secret.")
    st.stop()

uploaded_symbols = parse_symbol_csv(upload_watchlist)
symbols = clean_symbols(symbols_text, uploaded_symbols)
if not symbols:
    st.error("Provide at least one symbol.")
    st.stop()

all_syms = list(dict.fromkeys(symbols + [benchmark]))
if force_refresh:
    for kind in ["daily", "1hour", "2hour", "hourly_proxy", "2hour_proxy"]:
        for sym in all_syms:
            p = cache_path(sym, kind)
            if p.exists():
                p.unlink(missing_ok=True)
    fetch_alpaca_daily_batch.clear()
    fetch_alpaca_2hour.clear()
    fetch_alpaca_1hour.clear()

st.info("Loading Alpaca daily data...")
daily_map = fetch_alpaca_daily_batch(all_syms, history_years, alpaca_key, alpaca_secret, feed)
if benchmark not in daily_map or daily_map[benchmark].empty:
    st.error(f"Could not load benchmark {benchmark} from Alpaca.")
    st.stop()
benchmark_daily = daily_map[benchmark]

rows: List[Dict[str, object]] = []
detail: Dict[str, Dict[str, object]] = {}
progress = st.progress(0.0)

for i, sym in enumerate(symbols):
    progress.progress((i + 1) / max(1, len(symbols)))
    daily_raw = daily_map.get(sym, pd.DataFrame())
    if daily_raw.empty:
        rows.append({"Symbol": sym, "Status": "No data"})
        continue
        
    # Fetch/Proxy Logic
    if intraday_source == "real":
        tactical_raw = fetch_alpaca_2hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
        tactical_timeframe = "2H"
        tactical_label = "2-Hour (Real)"
        if tactical_raw.empty:
            tactical_raw, tactical_timeframe, _ = time_compressed_proxy(daily_raw, "2hour_proxy")
            tactical_label = "2-Hour (Proxy Fallback)"
        hourly_raw = fetch_alpaca_1hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
        hourly_label = "1-Hour (Real)"
        if hourly_raw.empty:
            hourly_raw, _, _ = time_compressed_proxy(daily_raw, "hourly_proxy")
            hourly_label = "1-Hour (Proxy Fallback)"
    else:
        tactical_raw = fetch_alpaca_2hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
        tactical_timeframe = "2H_PROXY"
        tactical_label = "2-Hour (Proxy Smooth)"
        if tactical_raw.empty:
            tactical_raw, _, _ = time_compressed_proxy(daily_raw, "2hour_proxy")
            tactical_label = "2-Hour (Proxy Fallback)"
        hourly_raw = fetch_alpaca_1hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
        hourly_label = "1-Hour (Proxy Smooth)"
        if hourly_raw.empty:
            hourly_raw, _, _ = time_compressed_proxy(daily_raw, "hourly_proxy")
            hourly_label = "1-Hour (Proxy Fallback)"

    weekly_raw = resample_weekly(daily_raw)
    
    # Feature Engineering
    hourly_df = enrich_price_features_tsi(hourly_raw, "1H_PROXY", benchmark_daily)
    tactical_df = enrich_price_features_tsi(tactical_raw, "2H_PROXY", benchmark_daily)
    daily_df = enrich_price_features_tsi(daily_raw, "Daily", benchmark_daily)
    weekly_df = enrich_price_features_tsi(weekly_raw, "Weekly", benchmark_daily)
    
    # Merge States
    hourly_df = merge_higher_state(hourly_df, tactical_df)
    tactical_df = merge_higher_state(tactical_df, daily_df)
    daily_df = merge_higher_state(daily_df, weekly_df)
    weekly_df["Higher_State"] = weekly_df["State"]
    weekly_df["Higher_Regime"] = weekly_df["Regime_bucket"]

    asof = pd.Timestamp.today().normalize() if analysis_mode == "Current" else pd.Timestamp(analysis_date)
    
    # Slicing
    def safe_slice(df, asof):
        if df.empty: return df.copy()
        end_ts = asof + pd.Timedelta(hours=23, minutes=59, seconds=59)
        return df.loc[df.index <= end_ts].copy()

    hourly_view = safe_slice(hourly_df, asof)
    tactical_view = safe_slice(tactical_df, asof)
    daily_view = safe_slice(daily_df, asof)
    weekly_view = safe_slice(weekly_df, asof)
    
    if daily_view.empty:
        rows.append({"Symbol": sym, "Status": "No data on date"})
        continue

    hourly_row = hourly_view.iloc[-1] if not hourly_view.empty else pd.Series(dtype=float)
    tactical_row = tactical_view.iloc[-1] if not tactical_view.empty else pd.Series(dtype=float)
    daily_row = daily_view.iloc[-1]
    weekly_row = weekly_view.iloc[-1] if not weekly_view.empty else pd.Series(dtype=float)

    # Classification
    hourly_call, hourly_reason = classify_timeframe_call_tsi(hourly_row, "1H")
    tactical_call, tactical_reason = classify_timeframe_call_tsi(tactical_row, "2H")
    daily_call, daily_reason = classify_timeframe_call_tsi(daily_row, "Daily")
    weekly_call, weekly_reason = classify_timeframe_call_tsi(weekly_row, "Weekly")
    
    # Analogs
    analogs = analog_summary_tsi(daily_df, "Daily")
    analog_summary_dict = analogs[0]
    overheat_score = (
        (float(tactical_row.get("TSI_heat", 50)) * 0.30)
        + (float(daily_row.get("TSI_heat", 50)) * 0.35)
        + (min(max(float(daily_row.get("RSI", 50)) / 100, 0), 1) * 0.15)
        + (min(max((float(daily_row.get("CCI", 0)) + 200) / 400, 0), 1) * 0.20)
    ) * 100

    # Recommendation Logic
    reco = tactical_call if tactical_call != "NEUTRAL" else daily_call
    if "Extended" in str(tactical_row.get("Regime_bucket", "")): reco += ", Extended"
    elif "Exhausted" in tactical_reason: reco += ", Exhausted"

    # Combine Score
    score = {"CALL": 1, "PUT": -1, "NEUTRAL": 0, "NO DATA": 0}
    net = 0.30 * score.get(hourly_call, 0) + 0.45 * score.get(tactical_call, 0) + 0.25 * score.get(daily_call, 0)
    combined = "CALL" if net >= 0.55 else ("PUT" if net <= -0.55 else ("CALL ON PULLBACK" if daily_call == "CALL" and weekly_call == "CALL" else "NEUTRAL"))

    rows.append({
        "Symbol": sym, "Status": "OK", "As Of": str(daily_row.name.date()),
        "Hourly": hourly_call, "Tactical": tactical_call, "Daily": daily_call, "Weekly": weekly_call,
        "Combined": combined, "Recommendation": reco,
        "Price": round(float(daily_row.get("Close", np.nan)), 2),
        "Overheat Score": round(overheat_score, 1),
        "Tactical Heat": round(float(tactical_row.get("TSI_heat", np.nan)), 1),
        "Daily Heat": round(float(daily_row.get("TSI_heat", np.nan)), 1),
        "RSI14": round(float(daily_row.get("RSI", np.nan)), 1),
        "CCI20": round(float(daily_row.get("CCI", np.nan)), 1),
        "Analog N": int(analog_summary_dict.get("n", 0)),
        "Analog 2d Med": round(float(analog_summary_dict.get("ret_2_median", np.nan)) * 100, 2),
        "Analog 2d Up %": round(float(analog_summary_dict.get("ret_2_p_up", np.nan)) * 100, 1),
    })
    
    detail[sym] = {
        "hourly": hourly_view, "hourly_label": hourly_label, "hourly_call": (hourly_call, hourly_reason),
        "tactical": tactical_view, "tactical_label": tactical_label, "tactical_call": (tactical_call, tactical_reason),
        "daily": daily_view, "daily_call": (daily_call, daily_reason),
        "weekly": weekly_view, "weekly_call": (weekly_call, weekly_reason),
        "combined": combined, "recommendation": reco,
        "analog_summary": analog_summary_dict,
        "cross": distance_to_cross(tactical_row, tactical_view),
        "asof": asof,
    }

progress.empty()

# Results Table
results_df = pd.DataFrame(rows)
st.subheader("Ranked Results")
if not results_df.empty:
    results_df = results_df.sort_values("Overheat Score", ascending=False)
    st.dataframe(results_df, width="stretch", hide_index=True)
    st.download_button("Download results CSV", results_df.to_csv(index=False).encode("utf-8"), "stable_market_engine_v12.csv", "text/csv")
else:
    st.warning("No results generated.")
    st.stop()

valid_symbols = results_df.loc[results_df["Status"] == "OK", "Symbol"].tolist()
if not valid_symbols:
    st.stop()

# Detailed Analysis
if "selected_symbol" not in st.session_state or st.session_state.selected_symbol not in valid_symbols:
    st.session_state.selected_symbol = valid_symbols[0]
selected = st.selectbox("Select symbol", valid_symbols, index=valid_symbols.index(st.session_state.selected_symbol))
st.session_state.selected_symbol = selected
item = detail[selected]

st.subheader("Detailed Analysis")
hourly_call, hourly_reason = item["hourly_call"]
tactical_call, tactical_reason = item["tactical_call"]
daily_call, daily_reason = item["daily_call"]
weekly_call, weekly_reason = item["weekly_call"]
tactical_label = item["tactical_label"]
hourly_label = item["hourly_label"]

render_traffic_lights([
    (hourly_label, hourly_call, item["hourly"].iloc[-1] if not item["hourly"].empty else None),
    (tactical_label, tactical_call, item["tactical"].iloc[-1] if not item["tactical"].empty else None),
    ("Daily", daily_call, item["daily"].iloc[-1] if not item["daily"].empty else None),
    ("Weekly", weekly_call, item["weekly"].iloc[-1] if not item["weekly"].empty else None),
])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(hourly_label, hourly_call)
c2.metric(tactical_label, tactical_call)
c3.metric("Daily", daily_call)
c4.metric("Weekly", weekly_call)
c5.metric("Combined", item["combined"])

st.markdown(f"**{hourly_label} reason:** {hourly_reason}")
st.markdown(f"**{tactical_label} reason:** {tactical_reason}")
st.markdown(f"**Daily reason:** {daily_reason}")
st.markdown(f"**Weekly reason:** {weekly_reason}")
st.markdown(f"**Recommendation:** {item['recommendation']}")

structural_bias = item["combined"] if item["combined"] != "NEUTRAL" else daily_call
st.markdown("### Frame warnings")
st.write(frame_warning_message(item["hourly"].iloc[-1] if not item["hourly"].empty else None, hourly_label, structural_bias))
st.write(frame_warning_message(item["tactical"].iloc[-1] if not item["tactical"].empty else None, tactical_label, structural_bias))
st.write(frame_warning_message(item["daily"].iloc[-1] if not item["daily"].empty else None, "Daily", structural_bias))
st.write(frame_warning_message(item["weekly"].iloc[-1] if not item["weekly"].empty else None, "Weekly", structural_bias))

cross = item["cross"]
cd1, cd2, cd3 = st.columns(3)
cd1.metric("Distance to cross", f"{cross.get('abs_gap', np.nan):.4f}" if pd.notna(cross.get("abs_gap", np.nan)) else "n/a")
gap_label = "Above signal" if cross.get("gap", np.nan) >= 0 else ("Below signal" if pd.notna(cross.get("gap", np.nan)) else "n/a")
cd2.metric("Cross gap sign", gap_label)
cd3.metric("Cross distance %", f"{cross.get('range_pct', np.nan):.1f}%" if pd.notna(cross.get("range_pct", np.nan)) else "n/a")

# Diagnostics
render_extreme_table([
    (hourly_label, item["hourly"].iloc[-1] if not item["hourly"].empty else pd.Series(dtype=float)),
    (tactical_label, item["tactical"].iloc[-1] if not item["tactical"].empty else pd.Series(dtype=float)),
    ("Daily", item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float)),
])

# As-of Table
last_h = item["hourly"].iloc[-1] if not item["hourly"].empty else pd.Series(dtype=float)
last_t = item["tactical"].iloc[-1] if not item["tactical"].empty else pd.Series(dtype=float)
last_d = item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float)
last_w = item["weekly"].iloc[-1] if not item["weekly"].empty else pd.Series(dtype=float)

asof_table = pd.DataFrame([
    {"Frame": hourly_label, "Date": str(last_h.name.date()) if not last_h.empty else "n/a", "TSI": round(float(last_h.get("TSI", np.nan)), 2), "Signal": round(float(last_h.get("TSI_signal", np.nan)), 2), "Heat": round(float(last_h.get("TSI_heat", np.nan)), 1), "State": str(last_h.get("State", ""))},
    {"Frame": tactical_label, "Date": str(last_t.name.date()) if not last_t.empty else "n/a", "TSI": round(float(last_t.get("TSI", np.nan)), 2), "Signal": round(float(last_t.get("TSI_signal", np.nan)), 2), "Heat": round(float(last_t.get("TSI_heat", np.nan)), 1), "State": str(last_t.get("State", ""))},
    {"Frame": "Daily", "Date": str(last_d.name.date()) if not last_d.empty else "n/a", "TSI": round(float(last_d.get("TSI", np.nan)), 2), "Signal": round(float(last_d.get("TSI_signal", np.nan)), 2), "Heat": round(float(last_d.get("TSI_heat", np.nan)), 1), "State": str(last_d.get("State", ""))},
    {"Frame": "Weekly", "Date": str(last_w.name.date()) if not last_w.empty else "n/a", "TSI": round(float(last_w.get("TSI", np.nan)), 2), "Signal": round(float(last_w.get("TSI_signal", np.nan)), 2), "Heat": round(float(last_w.get("TSI_heat", np.nan)), 1), "State": str(last_w.get("State", ""))},
])
st.markdown("### As-of values")
st.dataframe(asof_table, width="stretch", hide_index=True)

# Analogs
analog_summary_dict = item["analog_summary"]
st.markdown("### Historical analogs")
if analog_summary_dict:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Analog count", str(int(analog_summary_dict.get("n", 0))))
    a2.metric("2d median", f"{analog_summary_dict.get('ret_2_median', np.nan)*100:.2f}%" if pd.notna(analog_summary_dict.get("ret_2_median", np.nan)) else "n/a")
    a3.metric("2d up %", f"{analog_summary_dict.get('ret_2_p_up', np.nan)*100:.1f}%" if pd.notna(analog_summary_dict.get("ret_2_p_up", np.nan)) else "n/a")
    a4.metric("5d median", f"{analog_summary_dict.get('ret_5_median', np.nan)*100:.2f}%" if pd.notna(analog_summary_dict.get("ret_5_median", np.nan)) else "n/a")
else:
    st.caption("No analog set available.")

# Mega View & Tabs
range_options = ["2W", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "MAX"]
st.markdown("### Mega view")
mc1, mc2, mc3, mc4 = st.columns(4)
mega_ranges = {
    "1h": mc1.selectbox("Mega 1H Range", range_options, index=2, key="mega_range_1h"),
    "2h": mc2.selectbox("Mega 2H Range", range_options, index=3, key="mega_range_2h"),
    "daily": mc3.selectbox("Mega Daily Range", range_options, index=4, key="mega_range_daily"),
    "weekly": mc4.selectbox("Mega Weekly Range", range_options, index=6, key="mega_range_weekly"),
}
plot_mega_view(selected, item["hourly"], item["tactical"], item["daily"], item["weekly"], item["asof"], tactical_label, mega_ranges)

st.markdown("### Frame tabs")
tab1, tab2, tab3, tab4 = st.tabs([hourly_label, tactical_label, "Daily", "Weekly"])
with tab1:
    range_1h = st.selectbox("1H Range", range_options, index=2, key="tab_range_1h")
    plot_single_frame(selected, item["hourly"] if not item["hourly"].empty else item["daily"], item["hourly"], item["asof"], hourly_label, range_1h)
with tab2:
    range_2h = st.selectbox("2H Range", range_options, index=3, key="tab_range_2h")
    plot_single_frame(selected, item["tactical"] if not item["tactical"].empty else item["daily"], item["tactical"], item["asof"], tactical_label, range_2h)
with tab3:
    range_daily = st.selectbox("Daily Range", range_options, index=4, key="tab_range_daily")
    plot_single_frame(selected, item["daily"], item["daily"], item["asof"], "Daily TSI", range_daily)
with tab4:
    range_weekly = st.selectbox("Weekly Range", range_options, index=6, key="tab_range_weekly")
    plot_single_frame(selected, item["weekly"], item["weekly"], item["asof"], "Weekly TSI", range_weekly)
