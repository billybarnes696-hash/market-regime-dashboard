#!/usr/bin/env python3
"""
Stable Market Engine v13 — Probabilistic Ranker & MC Predictor
✅ Bulk CSV Upload / Paste
✅ Rank by Advance/Decline Probability
✅ Monte Carlo Forward-Return Predictor (1D, 2D, 5D, 10D)
✅ Smooth TSI Oscillator + Real Alpaca Data + Analog Engine
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

st.set_page_config(
    page_title="Stable Market Engine v13 — Ranker",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
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

def hybrid_normalize(series: pd.Series, window: int) -> pd.Series:
    lo = series.rolling(window, min_periods=max(20, window//5)).min()
    hi = series.rolling(window, min_periods=max(20, window//5)).max()
    return ((series - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)

def centered_pct(series: pd.Series) -> pd.Series:
    return (series.fillna(0.5) - 0.5) * 2

# -----------------------------
# DATA FETCHING & CACHING
# -----------------------------
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
    data_map = {}
    missing = [s for s in symbols if not is_fresh(cache_path(s, "daily"))]
    if not missing:
        for s in symbols:
            try: data_map[s] = pd.read_parquet(cache_path(s, "daily"))
            except: pass
        return data_map

    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(years=max(years, 5))).normalize().tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=missing, timeframe=TimeFrame.Day, start=start, end=end, adjustment="raw", feed=feed)
    try:
        raw = client.get_stock_bars(req).df
    except: return data_map

    for s in missing:
        try:
            sub = raw.xs(s, level=0) if isinstance(raw.index, pd.MultiIndex) else raw
            clean = normalize_ohlcv(sub)
            if not clean.empty:
                clean.to_parquet(cache_path(s, "daily"))
                data_map[s] = clean
        except: pass
    return data_map

def time_compressed_proxy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, str, str]:
    label = "2-Hour Proxy" if target == "2hour_proxy" else "Hourly Proxy"
    tf = "2hour_proxy" if target == "2hour_proxy" else "hourly_proxy"
    return df.copy(), tf, label

# -----------------------------
# FEATURE ENGINEERING (SMOOTH TSI)
# -----------------------------
def add_ultimate_oscillator(out: pd.DataFrame, timeframe_name: str) -> pd.DataFrame:
    spans = {
        "proxy_smooth_1hour": (8, 21, 7), "real_1hour": (8, 21, 7),
        "proxy_smooth_2hour": (10, 26, 8), "real_2hour": (10, 26, 8),
        "daily": (16, 42, 10), "weekly": (8, 21, 7)
    }
    pre_smooth_map = {
        "proxy_smooth_1hour": 4, "real_1hour": 3,
        "proxy_smooth_2hour": 5, "real_2hour": 4,
        "daily": 5, "weekly": 3
    }
    decision_smooth_map = {
        "proxy_smooth_1hour": 2, "real_1hour": 1,
        "proxy_smooth_2hour": 2, "real_2hour": 1,
        "daily": 1, "weekly": 1
    }
    viz_smooth_map = {
        "proxy_smooth_1hour": 21, "real_1hour": 13,
        "proxy_smooth_2hour": 18, "real_2hour": 10,
        "daily": 5, "weekly": 3
    }

    fast, slow, sig = spans[timeframe_name]
    tsi_n = np.tanh(out["tsi"].fillna(0.0) / 35.0)
    cci_n = np.tanh(out["cci_20"].fillna(0.0) / 180.0)
    bb_n = ((out["pct_b"].fillna(0.5) - 0.5) * 2.0).clip(-1.25, 1.25)
    vwap_n = np.tanh(out["dist_vwap_pct"].fillna(0.0) * 18.0)
    z_n = np.tanh(out["close_zscore"].fillna(0.0) / 2.5)
    adx_dir = np.sign(out["tsi_gap"].fillna(0.0) + out["uo_seed_dir"].fillna(0.0))
    adx_n = (((out["adx_14"].fillna(18.0) - 18.0) / 22.0).clip(-1.0, 1.0)) * adx_dir.replace(0, 1)

    w = dict(tsi=0.31, cci=0.22, bb=0.14, vwap=0.15, adx=0.10, z=0.08) if "hour" in timeframe_name or "2hour" in timeframe_name else dict(tsi=0.34, cci=0.18, bb=0.16, vwap=0.06, adx=0.14, z=0.12)
    
    out["uo_base"] = w["tsi"]*tsi_n + w["cci"]*cci_n + w["bb"]*bb_n + w["vwap"]*vwap_n + w["adx"]*adx_n + w["z"]*z_n
    out["uo_base_sm"] = ema(out["uo_base"], pre_smooth_map[timeframe_name]) if pre_smooth_map[timeframe_name] > 1 else out["uo_base"]
    out["uo_raw"] = ema(out["uo_base_sm"], fast) - ema(out["uo_base_sm"], slow)

    out["uo_decision"] = ema(out["uo_raw"], decision_smooth_map[timeframe_name]) if decision_smooth_map[timeframe_name] > 1 else out["uo_raw"]
    out["uo_signal_decision"] = ema(out["uo_decision"], sig)
    out["uo_viz"] = ema(out["uo_decision"], viz_smooth_map[timeframe_name])
    out["uo_signal_viz"] = ema(out["uo_viz"], sig)

    out["uo"] = out["uo_viz"]
    out["uo_signal"] = out["uo_signal_viz"]
    out["uo_gap"] = out["uo_decision"] - out["uo_signal_decision"]
    out["uo_slope_1"] = out["uo_decision"].diff(1)
    out["uo_slope_3"] = out["uo_decision"].diff(3)
    out["uo_pctile"] = hybrid_normalize(out["uo_decision"], 120 if "hour" in timeframe_name or "2hour" in timeframe_name else 252)
    out["uo_above_signal_2"] = (out["uo_decision"] > out["uo_signal_decision"]).rolling(2, min_periods=2).sum() == 2
    out["uo_below_signal_2"] = (out["uo_decision"] < out["uo_signal_decision"]).rolling(2, min_periods=2).sum() == 2
    return out

def enrich_price_features(df: pd.DataFrame, timeframe_name: str, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df.empty: return df.copy()
    x = df.copy()
    x["ema_10"], x["ema_20"], x["sma_50"] = ema(x["Close"], 10), ema(x["Close"], 20), sma(x["Close"], 50)
    x["atr_14"] = atr(x, 14)
    x["rsi_14"] = rsi(x["Close"], 14)
    x["cci_20"] = cci(x, 20)
    x["tsi"], x["tsi_signal"] = tsi(x["Close"], 25, 13, 7)
    x["tsi_gap"] = x["tsi"] - x["tsi_signal"]
    x["pct_b"] = bollinger_pct_b(x["Close"], 20, 2)
    x["adx_14"] = adx(x, 14)
    x["vwap"] = rolling_vwap(x, 20)
    x["dist_ema20_pct"] = (x["Close"] / x["ema_20"]) - 1
    x["dist_vwap_pct"] = (x["Close"] / x["vwap"]) - 1
    x["close_zscore"] = (x["Close"] - x["Close"].rolling(252).mean()) / x["Close"].rolling(252).std()
    
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(x.index).ffill()
        x["rs_bench_slope_5"] = slope(x["Close"] / aligned, 5)
    else:
        x["rs_bench_slope_5"] = 0.0

    x["uo_seed_dir"] = np.tanh(slope(x["Close"], 3).fillna(0.0) * 20.0)
    for col in ["rsi_14", "cci_20", "tsi", "pct_b", "adx_14", "dist_ema20_pct", "dist_vwap_pct", "close_zscore"]:
        if col in x.columns:
            x[f"{col}_pctile"] = hybrid_normalize(x[col], 252)
    return add_ultimate_oscillator(x, timeframe_name)

# -----------------------------
# ANALOGS & MONTE CARLO
# -----------------------------
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
            "median": float(np.median(sims)),
            "mean": float(np.mean(sims)),
            "p_up": float(np.mean(sims > 0) * 100),
            "p_down": float(np.mean(sims < 0) * 100),
            "pct_10": float(np.percentile(sims, 10)),
            "pct_90": float(np.percentile(sims, 90)),
            "sample": len(vals)
        }
    return results

# -----------------------------
# CLASSIFICATION & RANKING
# -----------------------------
def classify_timeframe_call(row: pd.Series, timeframe: str) -> Tuple[str, str]:
    if row is None or row.empty: return "NO DATA", "No data"
    uo_pct = float(row.get("uo_pctile", 0.5))
    uo_gap = float(row.get("uo_gap", 0.0))
    slope3 = float(row.get("uo_slope_3", 0.0))
    is_tactical = "hour" in timeframe or "2hour" in timeframe
    is_structural = timeframe in {"daily", "weekly"}

    if is_tactical and uo_pct > 0.88 and uo_gap < 0 and slope3 < 0:
        return "PUT", "Tactical rollover"
    if is_tactical and uo_pct > 0.75 and uo_gap > 0 and slope3 >= 0:
        return "CALL", "Tactical momentum"
    if is_tactical and uo_pct < 0.20 and uo_gap > 0 and slope3 > 0:
        return "CALL", "Washed-out turn"
    if is_tactical and uo_gap < 0 and slope3 < 0:
        return "PUT", "Falling"
    if is_structural:
        if uo_pct > 0.85 and int(row.get("uo_below_signal_2", 0))==1 and slope3 < 0:
            return "PUT", "Confirmed rollover"
        if uo_pct < 0.18 and int(row.get("uo_above_signal_2", 0))==1 and slope3 > 0:
            return "CALL", "Confirmed turn"
    if uo_gap > 0 and slope3 > 0: return "CALL", "Rising"
    if uo_gap < 0 and slope3 < 0: return "PUT", "Falling"
    return "NEUTRAL", "Mixed"

# -----------------------------
# PLOTTING
# -----------------------------
def plot_dashboard(symbol: str, proxy_df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, asof_date: pd.Timestamp, tactical_label: str) -> None:
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.05,
        subplot_titles=[f"{symbol} Daily Price", f"{tactical_label} Oscillator", "Daily Oscillator", "Weekly Oscillator"],
        row_heights=[0.38, 0.21, 0.21, 0.20])
    if not daily_df.empty:
        d = daily_df.tail(260)
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["ema_20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
    for rn, frame, nm in zip([2,3,4], [proxy_df.tail(260), daily_df.tail(260), weekly_df.tail(160)], [tactical_label, "Daily", "Weekly"]):
        if frame.empty: continue
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo"], name=f"{nm} UO", line=dict(color="red", width=2.3)), row=rn, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo_signal"], name=f"{nm} Signal", line=dict(color="black", width=1.3)), row=rn, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=rn, col=1)
    fig.update_layout(height=1200, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# MAIN APP
# -----------------------------
st.title("📈 Stable Market Engine v13 — Probabilistic Ranker")
st.caption("Upload CSV / Paste Symbols | Rank by Advance/Decline Probability | Monte Carlo Predictor")

with st.sidebar:
    st.header("Credentials & Input")
    alpaca_key = st.text_input("Alpaca API Key", type="password")
    alpaca_secret = st.text_input("Alpaca Secret", type="password")
    feed = st.selectbox("Feed", ["iex", "sip"], index=0)
    symbols_text = st.text_area("Paste Tickers", value="QQQ, SMH, NVDA, XLF, AMD", height=80)
    csv_file = st.file_uploader("Or Upload CSV (Symbol/Ticker column)", type=["csv"])
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"], index=0)
    history_years = st.selectbox("History (Yrs)", [3, 5, 10], index=1)
    run_analysis = st.button("Run Bulk Scan & Rank", type="primary", use_container_width=True)

if not run_analysis or not alpaca_key or not alpaca_secret:
    if not alpaca_key: st.warning("Enter Alpaca credentials.")
    st.stop()

# Parse symbols
symbols = []
if symbols_text.strip():
    symbols.extend([s.strip().upper() for s in symbols_text.replace("\n", ",").split(",") if s.strip()])
if csv_file:
    try:
        df = pd.read_csv(csv_file)
        col = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), df.columns[0])
        symbols.extend([str(x).strip().upper() for x in df[col].dropna() if str(x).strip()])
    except: st.error("Failed to parse CSV.")
symbols = list(dict.fromkeys(symbols))
if not symbols: st.error("Provide symbols via paste or CSV."); st.stop()

# Fetch Data
st.info(f"Fetching daily data for {len(symbols)} symbols...")
daily_map = fetch_alpaca_daily_batch(list(set(symbols + [benchmark])), history_years, alpaca_key, alpaca_secret, feed)
if benchmark not in daily_map or daily_map[benchmark].empty:
    st.error(f"Failed to fetch {benchmark}."); st.stop()
bench = daily_map[benchmark]

rows = []
detail = {}
progress = st.progress(0.0)

for i, sym in enumerate(symbols):
    progress.progress((i+1)/len(symbols))
    daily_raw = daily_map.get(sym, pd.DataFrame())
    if daily_raw.empty:
        rows.append({"Symbol": sym, "Status": "No Data"})
        continue
        
    # Create tactical proxy (smooth) & real daily
    tactical_raw, tactical_tf, tactical_label = time_compressed_proxy(daily_raw, "2hour_proxy")
    weekly_raw = daily_raw.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    
    hourly_df = enrich_price_features(tactical_raw, tactical_tf, bench)
    daily_df = enrich_price_features(daily_raw, "daily", bench)
    weekly_df = enrich_price_features(weekly_raw, "weekly", bench)
    
    daily_row = daily_df.iloc[-1]
    tactical_row = hourly_df.iloc[-1]
    weekly_row = weekly_df.iloc[-1] if not weekly_df.empty else pd.Series(dtype=float)
    
    t_call, _ = classify_timeframe_call(tactical_row, tactical_tf)
    d_call, _ = classify_timeframe_call(daily_row, "daily")
    w_call, _ = classify_timeframe_call(weekly_row, "weekly")
    
    # Analogs & MC
    analogs = find_analogs(daily_df, daily_row.name, n=30)
    mc = monte_carlo_from_analogs(analogs, horizons=[1, 2, 5, 10], n_sims=3000)
    mc2 = mc.get(2, {})
    mc5 = mc.get(5, {})
    
    # Probabilities & Ranking Score
    adv_prob = mc2.get("p_up", 50.0)
    dec_prob = mc2.get("p_down", 50.0)
    net_bias = adv_prob - 50.0
    
    rows.append({
        "Symbol": sym,
        "Price": round(float(daily_row.get("Close", np.nan)), 2),
        "Daily Call": d_call,
        "Tactical Call": t_call,
        "Weekly Call": w_call,
        "Prob Advance % (2D)": round(adv_prob, 1),
        "Prob Decline % (2D)": round(dec_prob, 1),
        "Net Bias Score": round(net_bias, 1),
        "MC 2D Median %": round(mc2.get("median", 0)*100, 2),
        "MC 5D Median %": round(mc5.get("median", 0)*100, 2),
        "MC Sample": mc2.get("sample", 0),
        "RSI14": round(float(daily_row.get("rsi_14", np.nan)), 1),
        "Status": "OK"
    })
    detail[sym] = {
        "hourly": hourly_df, "daily": daily_df, "weekly": weekly_df,
        "tactical_label": tactical_label, "mc": mc, "analogs": analogs,
        "daily_row": daily_row, "tactical_row": tactical_row
    }

progress.empty()
results_df = pd.DataFrame(rows)
results_df = results_df[results_df["Status"]=="OK"].sort_values("Net Bias Score", ascending=False).reset_index(drop=True)

# RESULTS TABLE
st.subheader("📊 Ranked Results (Sorted by Net Bias)")
col_sort = st.selectbox("Sort By", ["Net Bias Score", "Prob Advance % (2D)", "Prob Decline % (2D)", "RSI14"], index=0)
if col_sort in results_df.columns:
    asc = False if "Decline" in col_sort else True
    results_df = results_df.sort_values(col_sort, ascending=asc)

st.dataframe(results_df, use_container_width=True, hide_index=True)
st.download_button("Download Ranked CSV", results_df.to_csv(index=False).encode("utf-8"), "ranked_scan.csv", "text/csv")

# DETAILED VIEW
valid = results_df["Symbol"].tolist()
if not valid: st.stop()
selected = st.selectbox("Select Symbol", valid)
item = detail[selected]
mc = item["mc"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prob Advance (2D)", f"{mc.get(2,{}).get('p_up',50):.1f}%")
c2.metric("Prob Decline (2D)", f"{mc.get(2,{}).get('p_down',50):.1f}%")
c3.metric("MC 2D Median", f"{mc.get(2,{}).get('median',0)*100:.2f}%")
c4.metric("MC 5D Median", f"{mc.get(5,{}).get('median',0)*100:.2f}%")

plot_dashboard(selected, item["hourly"], item["daily"], item["weekly"], pd.Timestamp.today(), item["tactical_label"])

# MC PATH VISUALIZATION
if not item["analogs"].empty:
    st.markdown("### 🎲 Monte Carlo Forward Distribution (2-Day)")
    fig = go.Figure()
    rng = np.random.default_rng(42)
    sims = rng.choice(item["analogs"]["fwd_ret_2"].dropna().values, size=2000)
    fig.add_trace(go.Histogram(x=sims*100, nbinsx=40, marker_color="rgba(75, 192, 192, 0.6)", name="2D Return %"))
    fig.update_layout(barmode="overlay", height=300, xaxis_title="2-Day Return %", margin=dict(l=40,r=40,t=20,b=40))
    st.plotly_chart(fig, use_container_width=True)
