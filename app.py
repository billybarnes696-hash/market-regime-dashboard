#!/usr/bin/env python3
"""
Stable Market Engine v13.1 — Probabilistic Ranker & MC Predictor
✅ FIXED: Added CSV Upload Widget
✅ FIXED: Pandas resample offset syntax ("9H30T")
✅ Bulk CSV/Paste Scanning | Monte Carlo Predictor | Smooth Oscillator
"""

from __future__ import annotations
import io
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple
from pathlib import Path
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

st.set_page_config(page_title="Stable Market Engine v13.1", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# UTILITY HELPERS
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

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
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13, signal_period: int = 7) -> Tuple[pd.Series, pd.Series]:
    delta = series.diff()
    double_smoothed = ema(ema(delta, long_period), short_period)
    double_abs = ema(ema(delta.abs(), long_period), short_period)
    tsi_line = 100 * double_smoothed / double_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal_period)
    return tsi_line, signal_line

def bollinger_pct_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)

def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = np.where((high.diff() > -low.diff()) & (high.diff() > 0), high.diff(), 0.0)
    minus_dm = np.where((-low.diff() > high.diff()) & (-low.diff() > 0), -low.diff(), 0.0)
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()

def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0.0)
    return (typical * vol).rolling(window).sum() / vol.rolling(window).sum().replace(0, np.nan)

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out.columns = [str(c).title() for c in out.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    if len(keep) < 4: return pd.DataFrame()
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(NY_TZ).tz_localize(None)
    out = out.sort_index()
    for c in keep: out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Volume" not in out.columns: out["Volume"] = np.nan
    return out.dropna(subset=["Open", "High", "Low", "Close"])

def parse_symbol_csv(uploaded) -> List[str]:
    if uploaded is None: return []
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        cols = {str(c).strip().lower(): c for c in df.columns}
        sym_col = cols.get("symbol") or cols.get("ticker") or next(iter(df.columns), None)
        return [str(x).strip().upper() for x in df[sym_col].dropna().tolist() if str(x).strip()]
    except: return []

def cache_path(symbol: str, kind: str) -> Path:
    safe = "".join(c for c in symbol if c.isalnum() or c in ".-")
    return CACHE_DIR / f"{safe}_{kind}.parquet"

def is_fresh(path: Path, max_hours: int = 18) -> bool:
    if not path.exists(): return False
    return (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")) < pd.Timedelta(hours=max_hours)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpaca_daily_batch(symbols: List[str], years: int, key: str, secret: str, feed: str) -> Dict[str, pd.DataFrame]:
    if not symbols or not key or not secret: return {}
    client = StockHistoricalDataClient(key, secret)
    data_map, missing = {}, []
    for s in symbols:
        p = cache_path(s, "daily")
        if is_fresh(p):
            try: data_map[s] = pd.read_parquet(p); continue
            except: pass
        missing.append(s)
    if not missing: return data_map
    
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(years=max(years, 5))).normalize().tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=missing, timeframe=TimeFrame.Day, start=start, end=end, adjustment="raw", feed=feed)
    try: raw = client.get_stock_bars(req).df
    except: return data_map
    
    for s in missing:
        try:
            sub = raw.xs(s, level=0) if isinstance(raw.index, pd.MultiIndex) else raw
            clean = normalize_ohlcv(sub)
            if not clean.empty: clean.to_parquet(cache_path(s, "daily")); data_map[s] = clean
        except: pass
    return data_map

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_alpaca_2hour(symbol: str, months: int, key: str, secret: str, feed: str) -> pd.DataFrame:
    p = cache_path(symbol, "2hour_real")
    if is_fresh(p, max_hours=4):
        try: return pd.read_parquet(p)
        except: pass
    
    client = StockHistoricalDataClient(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(months=max(months, 3))).tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(30, TimeFrameUnit.Minute), start=start, end=end, adjustment="raw", feed=feed)
    try: raw = client.get_stock_bars(req).df
    except: return pd.DataFrame()
    
    if symbol in raw.index.get_level_values(0):
        sub = raw.xs(symbol, level=0)
        clean = normalize_ohlcv(sub)
    else:
        clean = normalize_ohlcv(raw)
        
    if clean.empty: return pd.DataFrame()
    clean = clean.between_time("09:30", "16:00")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    
    # ✅ FIXED: Use "9H30T" for pandas offset (9 hours 30 minutes)
    resampled = clean.resample("2H", offset="9H30T").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    
    if not resampled.empty: resampled.to_parquet(p)
    return resampled

# -----------------------------
# FEATURE ENGINEERING & OSCILLATOR
# -----------------------------
def add_ultimate_oscillator(out: pd.DataFrame, timeframe_name: str) -> pd.DataFrame:
    spans = {"2hour_proxy": (10, 26, 8), "real_2hour": (10, 26, 8), "daily": (16, 42, 10), "weekly": (8, 21, 7)}
    pre_smooth_map = {"2hour_proxy": 5, "real_2hour": 5, "daily": 5, "weekly": 3}
    decision_smooth_map = {"2hour_proxy": 2, "real_2hour": 2, "daily": 1, "weekly": 1}
    viz_smooth_map = {"2hour_proxy": 12, "real_2hour": 12, "daily": 5, "weekly": 3}

    fast, slow, sig = spans.get(timeframe_name, (16, 42, 10))
    tsi_n = np.tanh(out["tsi"].fillna(0.0) / 35.0)
    cci_n = np.tanh(out["cci_20"].fillna(0.0) / 180.0)
    bb_n = ((out["pct_b"].fillna(0.5) - 0.5) * 2.0).clip(-1.25, 1.25)
    vwap_n = np.tanh(out["dist_vwap_pct"].fillna(0.0) * 18.0)
    z_n = np.tanh(out["close_zscore"].fillna(0.0) / 2.5)
    w = dict(tsi=0.30, cci=0.20, bb=0.15, vwap=0.15, adx=0.10, z=0.10)
    
    out["uo_base"] = w["tsi"]*tsi_n + w["cci"]*cci_n + w["bb"]*bb_n + w["vwap"]*vwap_n + w["adx"]*out["adx_14_pctile"].fillna(0.5) + w["z"]*z_n
    
    pre = pre_smooth_map.get(timeframe_name, 3)
    out["uo_base_sm"] = ema(out["uo_base"], pre) if pre > 1 else out["uo_base"]
    out["uo_raw"] = ema(out["uo_base_sm"], fast) - ema(out["uo_base_sm"], slow)

    dec = decision_smooth_map.get(timeframe_name, 1)
    viz = viz_smooth_map.get(timeframe_name, 5)
    
    out["uo_decision"] = ema(out["uo_raw"], dec) if dec > 1 else out["uo_raw"]
    out["uo_signal_decision"] = ema(out["uo_decision"], sig)
    out["uo_viz"] = ema(out["uo_decision"], viz) if viz > 1 else out["uo_decision"]
    out["uo_signal_viz"] = ema(out["uo_viz"], sig)

    out["uo"] = out["uo_viz"]
    out["uo_signal"] = out["uo_signal_viz"]
    out["uo_gap"] = out["uo_decision"] - out["uo_signal_decision"]
    out["uo_slope_3"] = out["uo_decision"].diff(3)
    
    lookback = 120 if "2hour" in timeframe_name else 252
    out["uo_pctile"] = ((out["uo_decision"] - out["uo_decision"].rolling(lookback, min_periods=20).min()) / 
                        (out["uo_decision"].rolling(lookback, min_periods=20).max() - out["uo_decision"].rolling(lookback, min_periods=20).min()).replace(0, np.nan)).clip(0, 1)
    return out

def enrich_price_features(df: pd.DataFrame, timeframe_name: str, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df.empty: return df.copy()
    x = df.copy()
    x["ema_10"], x["ema_20"], x["sma_50"] = ema(x["Close"], 10), ema(x["Close"], 20), sma(x["Close"], 50)
    x["rsi_14"] = rsi(x["Close"], 14)
    x["cci_20"] = cci(x, 20)
    x["tsi"], x["tsi_signal"] = tsi(x["Close"], 25, 13, 7)
    x["pct_b"] = bollinger_pct_b(x["Close"], 20, 2)
    x["adx_14"] = adx(x, 14)
    x["dist_ema20_pct"] = (x["Close"] / x["ema_20"]) - 1
    x["vwap"] = rolling_vwap(x, 12) if "2hour" in timeframe_name else rolling_vwap(x, 20)
    x["dist_vwap_pct"] = (x["Close"] / x["vwap"]) - 1
    x["close_zscore"] = (x["Close"] - x["Close"].rolling(252).mean()) / x["Close"].rolling(252).std()
    
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(x.index).ffill()
        x["rs_bench_slope_5"] = slope(x["Close"] / aligned, 5)
    else: x["rs_bench_slope_5"] = 0.0

    for col in ["rsi_14", "cci_20", "tsi", "pct_b", "adx_14", "dist_ema20_pct", "dist_vwap_pct", "close_zscore"]:
        if col in x.columns:
            lo = x[col].rolling(252, min_periods=20).min()
            hi = x[col].rolling(252, min_periods=20).max()
            x[f"{col}_pctile"] = ((x[col] - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)
            
    return add_ultimate_oscillator(x, timeframe_name)

# -----------------------------
# GRADING & PREDICTION LOGIC
# -----------------------------
def compute_grade(row: pd.Series, analog_summary: Dict) -> Tuple[str, str, float]:
    uo_pct = float(row.get("uo_pctile", 0.5))
    uo_gap = float(row.get("uo_gap", 0.0))
    uo_slope = float(row.get("uo_slope_3", 0.0))
    ob_int = float(row.get("rsi_14_pctile", 0.5))
    score = 0.0
    
    if uo_gap > 0.01 and uo_slope > 0: score += 0.8
    elif uo_gap < -0.01 and uo_slope < 0: score -= 0.8
    elif uo_gap > 0: score += 0.2
    else: score -= 0.2
    
    if analog_summary:
        p_up = analog_summary.get("p_up", 50.0)
        p_down = analog_summary.get("p_down", 50.0)
        if p_up > 0.60: score += 0.3
        if p_down > 0.60: score -= 0.3
        
    if ob_int > 0.90: score -= 0.5; risk = "EXTREME RISK / Overheated"
    elif ob_int > 0.80: score -= 0.3; risk = "High Risk / Extended"
    elif ob_int < 0.10: score += 0.3; risk = "Oversold / Bounce Risk"
    else: risk = "Healthy"

    if score >= 0.8 and risk == "Healthy": return "A+", "Strong Bullish / Healthy", score
    if score >= 0.5 and risk == "Healthy": return "A", "Bullish / Healthy", score
    if score >= 0.5: return "B-", "Bullish / Extended", score
    if score >= 0.2: return "B", "Moderate Bullish", score
    if score <= -0.5 and ob_int < 0.2: return "C+", "Bearish / Oversold Bounce?", score
    if score <= -0.5: return "C-", "Strong Bearish", score
    return "C", "Neutral / Mixed", score

def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [1, 2, 5, 10]: out[f"fwd_ret_{n}"] = out["Close"].shift(-n) / out["Close"] - 1
    return out

def find_analogs(frame: pd.DataFrame, current_ts: pd.Timestamp, n: int = 30) -> pd.DataFrame:
    enriched = add_forward_returns(frame)
    use = ["uo_pctile", "uo_gap", "uo_slope_3", "tsi", "tsi_gap", "cci_20", "pct_b", "dist_ema20_pct"]
    use = [c for c in use if c in enriched.columns]
    if len(use) < 6 or current_ts not in enriched.index: return pd.DataFrame()
    
    cur_pos = enriched.index.get_loc(current_ts)
    pool = enriched.iloc[:max(0, cur_pos-10)].dropna(subset=use+[f"fwd_ret_{n}" for n in [1,2,5,10]]).copy()
    if len(pool) < 40: return pd.DataFrame()

    current = enriched.loc[current_ts, use].astype(float)
    std = pool[use].std().replace(0, np.nan)
    z = ((pool[use] - current) / std).fillna(0.0)
    pool["distance"] = np.sqrt((z.to_numpy()**2).sum(axis=1))
    pool["similarity"] = 1 / (1 + pool["distance"])
    return pool.nsmallest(n, "distance").copy()

def monte_carlo_from_analogs(analogs: pd.DataFrame, horizons: List[int] = [1, 2, 5, 10], n_sims: int = 3000) -> Dict[str, Dict]:
    if analogs.empty: return {}
    results = {}
    for h in horizons:
        col = f"fwd_ret_{h}"
        if col not in analogs.columns: continue
        vals = analogs[col].dropna().values
        if len(vals) < 5: continue
        rng = np.random.default_rng(42)
        sims = rng.choice(vals, size=n_sims)
        results[h] = {
            "median": float(np.median(sims)), "mean": float(np.mean(sims)),
            "p_up": float(np.mean(sims > 0) * 100), "p_down": float(np.mean(sims < 0) * 100),
            "sample": len(vals)
        }
    return results

def classify_timeframe_call(row: pd.Series, timeframe: str) -> Tuple[str, str]:
    if row is None or row.empty: return "NO DATA", "No data"
    uo_pct = float(row.get("uo_pctile", 0.5))
    uo_gap = float(row.get("uo_gap", 0.0))
    slope3 = float(row.get("uo_slope_3", 0.0))
    is_structural = timeframe in {"daily", "weekly"}
    
    if uo_pct > 0.88 and uo_gap < 0 and slope3 < 0: return "PUT", "Rollover"
    if uo_pct < 0.20 and uo_gap > 0 and slope3 > 0: return "CALL", "Washed-out turn"
    if uo_gap > 0 and slope3 > 0: return "CALL", "Rising"
    if uo_gap < 0 and slope3 < 0: return "PUT", "Falling"
    return "NEUTRAL", "Mixed"

# -----------------------------
# DASHBOARD & PLOTTING
# -----------------------------
def plot_dashboard(symbol: str, hourly_df, tactical_df, daily_df, weekly_df, asof_date, tactical_label):
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.04,
        subplot_titles=[f"{symbol} Price", f"{tactical_label} Oscillator", "Daily Oscillator", "Weekly Oscillator", "Overbought/Oversold"],
        row_heights=[0.25, 0.18, 0.18, 0.18, 0.21])
    
    if not daily_df.empty:
        d = daily_df.tail(260)
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"), row=1, col=1)
        
    for rn, frame, nm in zip([2,3,4], [tactical_df.tail(260), daily_df.tail(260), weekly_df.tail(160)], [tactical_label, "Daily", "Weekly"]):
        if frame.empty: continue
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo"], name=f"{nm} UO", line=dict(color="red", width=2.3)), row=rn, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo_signal"], name=f"{nm} Signal", line=dict(color="black", width=1.3)), row=rn, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=rn, col=1)
        
    fig.update_layout(height=1300, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# MAIN APP
# -----------------------------
st.title("📈 Stable Market Engine v13.1 — Probabilistic Ranker")
st.caption("Upload CSV / Paste Symbols | 2H Tactical Call | MC Predictor | Real 2H Data")

with st.sidebar:
    st.header("Credentials")
    key = st.text_input("Alpaca API Key", type="password")
    secret = st.text_input("Secret Key", type="password")
    feed = st.selectbox("Feed", ["iex", "sip"], index=0)
    
    st.header("Input")
    symbols_text = st.text_area("Paste Tickers", value="QQQ, SMH, NVDA, XLF", height=80)
    csv_file = st.file_uploader("Or Upload CSV Watchlist", type=["csv"]) # ✅ CSV UPLOAD ADDED HERE
    
    st.header("Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"], index=0)
    history_years = st.selectbox("History (Yrs)", [3, 5, 10], index=1)
    run = st.button("Run Bulk Scan & Rank", type="primary", use_container_width=True)

if not run or not key: st.stop()

symbols = []
if symbols_text.strip(): symbols.extend([s.strip().upper() for s in symbols_text.replace("\n", ",").split(",") if s.strip()])
if csv_file is not None: symbols.extend(parse_symbol_csv(csv_file))
symbols = list(dict.fromkeys(symbols))
if not symbols: st.error("Provide symbols via paste or CSV."); st.stop()

st.info(f"Fetching data for {len(symbols)} symbols...")
daily_map = fetch_alpaca_daily_batch(symbols + ["SPY"], 5, key, secret, feed)
if "SPY" not in daily_map: st.error("Failed to fetch benchmark."); st.stop()

rows, detail = [], {}
for sym in symbols:
    daily_raw = daily_map.get(sym, pd.DataFrame())
    if daily_raw.empty: continue
    
    tactical_raw = fetch_alpaca_2hour(sym, 3, key, secret, feed)
    tactical_label = "2-Hour (Real Alpaca)" if not tactical_raw.empty else "2-Hour (Proxy Fallback)"
    if tactical_raw.empty: tactical_raw = daily_raw.copy()
        
    weekly_raw = daily_raw.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    
    tactical_df = enrich_price_features(tactical_raw, "real_2hour" if not tactical_raw.equals(daily_raw) else "2hour_proxy", daily_map["SPY"])
    daily_df = enrich_price_features(daily_raw, "daily", daily_map["SPY"])
    
    row_2h = tactical_df.iloc[-1]
    row_d = daily_df.iloc[-1]
    
    analogs = find_analogs(daily_df, row_d.name, n=30)
    mc = monte_carlo_from_analogs(analogs, horizons=[1, 2, 5, 10])
    mc2 = mc.get(2, {})
    
    grade, grade_reason, grade_score = compute_grade(row_d, mc2)
    
    rows.append({
        "Symbol": sym, "Grade": grade, "Grade Reason": grade_reason,
        "2H Tactical": "CALL" if row_2h["uo_gap"] > 0 else "PUT",
        "Daily Call": "CALL" if row_d["uo_gap"] > 0 else "PUT",
        "Prob Adv % (2D)": round(mc2.get("p_up", 50.0), 1),
        "Prob Dec % (2D)": round(mc2.get("p_down", 50.0), 1),
        "Net Bias": round(mc2.get("p_up", 50.0) - 50.0, 1),
        "MC 2D Med %": round(mc2.get("median", 0)*100, 2),
        "MC 5D Med %": round(mc.get(5, {}).get("median", 0)*100, 2),
        "Liq Options": "Y" if mc2.get("sample", 0) > 20 else "N",
        "RSI14": round(float(row_d.get("rsi_14", 0)), 1),
        "Status": "OK"
    })
    detail[sym] = {"tactical": tactical_df, "daily": daily_df, "weekly": weekly_raw, "tactical_label": tactical_label, "mc": mc, "analogs": analogs, "grade_reason": grade_reason, "grade_score": grade_score}

results_df = pd.DataFrame(rows).sort_values("Net Bias", ascending=False).reset_index(drop=True)
st.subheader("📊 Ranked Results")
st.dataframe(results_df, use_container_width=True, hide_index=True)
st.download_button("Download Ranked CSV", results_df.to_csv(index=False).encode("utf-8"), "ranked_scan.csv", "text/csv")

valid = results_df.loc[results_df["Status"]=="OK", "Symbol"].tolist()
if not valid: st.stop()

default_idx = valid.index(st.session_state.get("selected_symbol", valid[0])) if st.session_state.get("selected_symbol") in valid else 0
selected = st.selectbox("Select Symbol for Detail View", valid, index=default_idx, key="symbol_dropdown")
st.session_state.selected_symbol = selected

item = detail[selected]
mc = item["mc"]

st.markdown(f"### {selected} — Grade: {results_df.loc[results_df['Symbol']==selected, 'Grade'].values[0]} ({item['grade_reason']})")
st.progress((item["grade_score"] + 1) / 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prob Advance (2D)", f"{mc.get(2,{}).get('p_up',50):.1f}%")
c2.metric("Prob Decline (2D)", f"{mc.get(2,{}).get('p_down',50):.1f}%")
c3.metric("MC 2D Median", f"{mc.get(2,{}).get('median',0)*100:.2f}%")
c4.metric("MC 5D Median", f"{mc.get(5,{}).get('median',0)*100:.2f}%")

plot_dashboard(selected, item["tactical"], item["tactical"], item["daily"], item["weekly"], pd.Timestamp.today(), item["tactical_label"])

if not item["analogs"].empty:
    st.markdown("### 🎲 Monte Carlo Forward Distribution (2-Day)")
    fig = go.Figure()
    rng = np.random.default_rng(42)
    sims = rng.choice(item["analogs"]["fwd_ret_2"].dropna().values, size=2000)
    fig.add_trace(go.Histogram(x=sims*100, nbinsx=40, marker_color="rgba(75, 192, 192, 0.6)", name="2D Return %"))
    fig.update_layout(barmode="overlay", height=300, xaxis_title="2-Day Return %", margin=dict(l=40,r=40,t=20,b=40))
    st.plotly_chart(fig, use_container_width=True)
