# app.py
# ------------------------------------------------------------
# Optionable ETF EOD Screener (Daily) — Consolidated + Robust
# ------------------------------------------------------------
# What you asked for:
# - Keep the sidebar “toggle/inputs” style, but allow you to SET thresholds:
#     * TSI must be > (you choose 90/95/etc)
#     * CCI must be > (you choose 125/etc)
#     * Optional: Require CCI regressing (down N days)
# - Remove the extra “options table / extra tables”; focus on screener.
# - After results, show Diamonds + a clear “WHY” breakdown (TSI>thr, CCI regressing, etc).
# - Multi-panel chart like StockCharts (optional; if plotly missing, app still runs).
# - Bulletproof yfinance fetch to avoid KeyError on missing OHLC columns.
# - pandas_ta optional: if missing, fallback to local indicator implementations.
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

from __future__ import annotations

import importlib
from datetime import date, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Optional dependencies
# -----------------------------
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    PLOTLY_AVAILABLE = False

PANDAS_TA_AVAILABLE = True
try:
    ta = importlib.import_module("pandas_ta")
except Exception:
    PANDAS_TA_AVAILABLE = False
    ta = None

YFINANCE_AVAILABLE = True
try:
    yf = importlib.import_module("yfinance")
except Exception:
    YFINANCE_AVAILABLE = False
    yf = None

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Optionable ETF EOD Screener (Daily)", layout="wide")

# -----------------------------
# Hard requirements check
# -----------------------------
if not YFINANCE_AVAILABLE:
    st.error(
        "Missing required package: yfinance\n\n"
        "Add this to requirements.txt and redeploy:\n"
        "- yfinance>=0.2.43"
    )
    st.stop()

# -----------------------------
# Fallback ETF list (if scrape fails)
# -----------------------------
FALLBACK_ETFS = [
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLI","XLY","XLU","XLV","XLP","XLB",
    "SMH","SOXX","ARKK","TLT","IEF","SHY","HYG","LQD","GLD","SLV","GDX",
    "USO","UNG","VNQ","EEM","VWO","FXI","EWJ","EWZ","EFA","VEA","VTI","VOO"
]

# -----------------------------
# Universe: pull ETF tickers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_all_etf_tickers() -> List[str]:
    """
    Best effort:
    - Try StockAnalysis ETF directory (scrape /etf/XXX/ links)
    - If it fails, return a fallback list
    """
    url = "https://stockanalysis.com/etf/"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        html = r.text
    except Exception:
        return FALLBACK_ETFS

    # Prefer lxml if present; else html.parser
    parser = "lxml"
    try:
        importlib.import_module("lxml")
    except Exception:
        parser = "html.parser"

    soup = BeautifulSoup(html, parser)
    tickers = set()

    for a in soup.select('a[href^="/etf/"]'):
        href = a.get("href", "")
        parts = href.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "etf":
            t = parts[1].upper()
            if 1 <= len(t) <= 6 and t.isalnum():
                tickers.add(t)

    return sorted(tickers) if tickers else FALLBACK_ETFS

# -----------------------------
# Data fetch (daily bars) — BULLETPROOF
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_daily_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Bulletproof daily OHLCV fetch for a single ticker from yfinance.
    Handles:
      - empty / partial responses
      - MultiIndex columns
      - missing OHLC columns
      - weird column casing
    Returns empty df if unusable.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if present (sometimes happens)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c and c[0] else c[-1] for c in df.columns]

    # Normalize names
    df.columns = [str(c).strip().title() for c in df.columns]

    # Ensure datetime index
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    required_ohlc = {"Open", "High", "Low", "Close"}
    if not required_ohlc.issubset(set(df.columns)):
        return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    return df

# -----------------------------
# Optionable check (proxy): options expirations exist
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def is_optionable_yf(ticker: str) -> bool:
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        return bool(exps and len(exps) > 0)
    except Exception:
        return False

# -----------------------------
# Indicator implementations (fallback if pandas_ta missing)
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = ema(gain, length)
    avg_loss = ema(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(length).mean()
    mad = (tp - sma_tp).abs().rolling(length).mean()
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    denom = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / denom

def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * volume
    return mfv.rolling(length).sum() / volume.rolling(length).sum()

def tsi(close: pd.Series, fast: int = 6, slow: int = 3, signal: int = 6) -> pd.Series:
    # Common TSI form: 100 * EMA(EMA(mom, slow), fast) / EMA(EMA(|mom|, slow), fast)
    mom = close.diff()
    num = ema(ema(mom, slow), fast)
    den = ema(ema(mom.abs(), slow), fast)
    return 100 * (num / den.replace(0, np.nan))

# -----------------------------
# Features + scoring
# -----------------------------
def compute_features(
    df: pd.DataFrame,
    tsi_fast=6, tsi_slow=3, tsi_signal=6,
    rsi_len=14, cci_len=20, willr_len=14,
    cmf_len=20,
    vwap_len=20,
    vwap_1=0.01, vwap_2=0.02,
    wick_thresh=0.50,
    min_score_trigger=6,
    # NEW thresholds
    tsi_thr=95.0,
    rsi_thr=70.0,
    cci_thr=125.0,
    willr_thr=-20.0,
    cmf_thr=0.0,
    # NEW regression rule
    use_cci_regress=True,
    cci_regress_days=2,
) -> pd.DataFrame:
    out = df.copy()

    # Indicators via pandas_ta when possible; fallback otherwise
    if PANDAS_TA_AVAILABLE:
        try:
            tsi_df = ta.tsi(out["Close"], fast=tsi_fast, slow=tsi_slow, signal=tsi_signal)
            if tsi_df is not None and not tsi_df.empty:
                out = out.join(tsi_df)

            out["RSI"] = ta.rsi(out["Close"], length=rsi_len)
            out["CCI"] = ta.cci(out["High"], out["Low"], out["Close"], length=cci_len)
            out["WILLR"] = ta.willr(out["High"], out["Low"], out["Close"], length=willr_len)
            out["CMF"] = ta.cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=cmf_len)

            # Grab a usable TSI column
            tsi_main = [c for c in out.columns if c.lower().startswith("tsi_") and not c.lower().startswith("tsis_")]
            tsi_sig = [c for c in out.columns if c.lower().startswith("tsis_")]
            tsi_col = tsi_main[0] if tsi_main else (tsi_sig[0] if tsi_sig else None)
            out["TSI"] = out[tsi_col] if tsi_col else np.nan
        except Exception:
            out["TSI"] = tsi(out["Close"], fast=tsi_fast, slow=tsi_slow, signal=tsi_signal)
            out["RSI"] = rsi(out["Close"], length=rsi_len)
            out["CCI"] = cci(out["High"], out["Low"], out["Close"], length=cci_len)
            out["WILLR"] = willr(out["High"], out["Low"], out["Close"], length=willr_len)
            out["CMF"] = cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=cmf_len)
    else:
        out["TSI"] = tsi(out["Close"], fast=tsi_fast, slow=tsi_slow, signal=tsi_signal)
        out["RSI"] = rsi(out["Close"], length=rsi_len)
        out["CCI"] = cci(out["High"], out["Low"], out["Close"], length=cci_len)
        out["WILLR"] = willr(out["High"], out["Low"], out["Close"], length=willr_len)
        out["CMF"] = cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=cmf_len)

    # VWAP proxy (daily): sum(close*vol)/sum(vol) over vwap_len
    pv = out["Close"] * out["Volume"]
    out["VWAP_PROXY"] = pv.rolling(vwap_len).sum() / out["Volume"].rolling(vwap_len).sum()

    # VWAP stretch points (0/1/2)
    out["VW_STRETCH"] = 0
    out.loc[out["Close"] > out["VWAP_PROXY"] * (1 + vwap_2), "VW_STRETCH"] = 2
    out.loc[
        (out["Close"] > out["VWAP_PROXY"] * (1 + vwap_1)) &
        (out["Close"] <= out["VWAP_PROXY"] * (1 + vwap_2)),
        "VW_STRETCH"
    ] = 1

    # Candle exhaustion: upper wick / range
    upper_wick = out["High"] - np.maximum(out["Open"], out["Close"])
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    out["UPPER_WICK_PCT"] = (upper_wick / rng).clip(lower=0, upper=1)
    out["CANDLE_EXHAUST"] = (out["UPPER_WICK_PCT"] >= wick_thresh).astype(int)

    # Simple bearish divergence (1-bar): higher close, RSI lower
    out["BEAR_DIV"] = ((out["Close"] > out["Close"].shift(1)) & (out["RSI"] < out["RSI"].shift(1))).astype(int)

    # Threshold signals
    out["S_TSI"] = (out["TSI"] > float(tsi_thr)).astype(int)
    out["S_RSI"] = (out["RSI"] > float(rsi_thr)).astype(int)
    out["S_CCI"] = (out["CCI"] > float(cci_thr)).astype(int)
    out["S_WILLR"] = (out["WILLR"] > float(willr_thr)).astype(int)  # W%R in [-100,0]
    out["S_CMF"] = (out["CMF"] < float(cmf_thr)).astype(int)

    # CCI regressing: down N consecutive days
    cci_diff = out["CCI"].diff()
    out["CCI_REGRESS"] = ((cci_diff < 0).rolling(int(cci_regress_days)).sum() == int(cci_regress_days))
    out["CCI_REGRESS"] = out["CCI_REGRESS"].fillna(False).astype(int)
    out["S_CCI_REGRESS"] = out["CCI_REGRESS"].astype(int) if use_cci_regress else 0

    # Score
    out["SCORE"] = (
        out["S_TSI"] + out["S_RSI"] + out["S_CCI"] + out["S_WILLR"] + out["S_CMF"] +
        out["VW_STRETCH"] + out["CANDLE_EXHAUST"] + out["BEAR_DIV"] +
        (out["S_CCI_REGRESS"] if use_cci_regress else 0)
    )

    out["MAX_SCORE"] = 10 if use_cci_regress else 9
    out["PROB_PCT"] = (out["SCORE"] / out["MAX_SCORE"] * 100).clip(0, 100)
    out["SIGNAL"] = (out["SCORE"] >= min_score_trigger)

    return out

def backtest_next_day_drop(out: pd.DataFrame, min_score: int, drop_pct: float) -> dict:
    df = out.copy()
    df["NEXT_CLOSE"] = df["Close"].shift(-1)
    df["NEXT_RET_PCT"] = (df["NEXT_CLOSE"] / df["Close"] - 1) * 100
    df["SIGNAL"] = df["SCORE"] >= min_score
    df["HIT"] = df["SIGNAL"] & (df["NEXT_RET_PCT"] <= -drop_pct)

    signals = int(df["SIGNAL"].sum(skipna=True))
    wins = int(df["HIT"].sum(skipna=True))
    acc = (wins / signals * 100) if signals > 0 else np.nan
    latest = df.iloc[-1].copy()
    return {"df": df, "signals": signals, "wins": wins, "accuracy": acc, "latest": latest}

# -----------------------------
# Diamond reasons helper
# -----------------------------
def diamond_reasons(
    latest: pd.Series,
    *,
    tsi_thr: float, rsi_thr: float, cci_thr: float, willr_thr: float, cmf_thr: float,
    use_cci_regress: bool
) -> List[str]:
    reasons: List[str] = []

    def add_if(cond: bool, text: str):
        if bool(cond):
            reasons.append(text)

    tsi_v = latest.get("TSI", np.nan)
    rsi_v = latest.get("RSI", np.nan)
    cci_v = latest.get("CCI", np.nan)
    willr_v = latest.get("WILLR", np.nan)
    cmf_v = latest.get("CMF", np.nan)

    add_if(np.isfinite(tsi_v) and tsi_v > tsi_thr, f"TSI > {tsi_thr:.0f} ({tsi_v:.1f})")
    add_if(np.isfinite(rsi_v) and rsi_v > rsi_thr, f"RSI > {rsi_thr:.0f} ({rsi_v:.1f})")
    add_if(np.isfinite(cci_v) and cci_v > cci_thr, f"CCI > {cci_thr:.0f} ({cci_v:.0f})")
    add_if(np.isfinite(willr_v) and willr_v > willr_thr, f"W%R > {willr_thr:.0f} ({willr_v:.1f})")
    add_if(np.isfinite(cmf_v) and cmf_v < cmf_thr, f"CMF < {cmf_thr:.2f} ({cmf_v:.3f})")

    vw_pts = int(latest.get("VW_STRETCH", 0) if pd.notna(latest.get("VW_STRETCH", np.nan)) else 0)
    add_if(vw_pts >= 1, f"VWAP stretch pts = {vw_pts}")

    exhaust = int(latest.get("CANDLE_EXHAUST", 0) if pd.notna(latest.get("CANDLE_EXHAUST", np.nan)) else 0)
    wick_pct = latest.get("UPPER_WICK_PCT", np.nan)
    add_if(exhaust == 1 and np.isfinite(wick_pct), f"Upper-wick exhaustion ({wick_pct*100:.0f}%)")

    div = int(latest.get("BEAR_DIV", 0) if pd.notna(latest.get("BEAR_DIV", np.nan)) else 0)
    add_if(div == 1, "Bearish divergence (1-bar)")

    if use_cci_regress:
        reg = int(latest.get("CCI_REGRESS", 0) if pd.notna(latest.get("CCI_REGRESS", np.nan)) else 0)
        add_if(reg == 1, "CCI regressing (down N days)")

    return reasons

# -----------------------------
# Chart (optional Plotly)
# -----------------------------
def stockcharts_style_figure(df: pd.DataFrame, title: str, signal_score: int):
    dfp = df.dropna(subset=["Open", "High", "Low", "Close"]).tail(220).copy()
    has_prob = "PROB_PCT" in dfp.columns

    rows = 7 + (1 if has_prob else 0)
    row_heights = [0.38, 0.10, 0.10, 0.08, 0.08, 0.08, 0.10] + ([0.08] if has_prob else [])
    row_titles = [
        "Price (Candles) + VWAP Proxy",
        "Volume",
        "TSI",
        "RSI",
        "CCI",
        "Williams %R",
        "CMF",
    ] + (["Probability %"] if has_prob else [])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=row_titles,
    )

    fig.add_trace(
        go.Candlestick(
            x=dfp.index, open=dfp["Open"], high=dfp["High"], low=dfp["Low"], close=dfp["Close"],
            name="Price",
        ),
        row=1, col=1
    )

    if "VWAP_PROXY" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["VWAP_PROXY"], mode="lines", name="VWAP Proxy"), row=1, col=1)

    # Signal markers
    if "SCORE" in dfp.columns:
        sig = dfp["SCORE"] >= signal_score
        fig.add_trace(
            go.Scatter(
                x=dfp.index[sig],
                y=dfp.loc[sig, "High"] * 1.01,
                mode="markers",
                name="Signal",
                marker=dict(symbol="triangle-down", size=9),
            ),
            row=1, col=1
        )

    fig.add_trace(go.Bar(x=dfp.index, y=dfp["Volume"], name="Volume"), row=2, col=1)

    if "TSI" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["TSI"], mode="lines", name="TSI"), row=3, col=1)

    if "RSI" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["RSI"], mode="lines", name="RSI"), row=4, col=1)

    if "CCI" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["CCI"], mode="lines", name="CCI"), row=5, col=1)

    if "WILLR" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["WILLR"], mode="lines", name="Williams %R"), row=6, col=1)

    if "CMF" in dfp.columns:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["CMF"], mode="lines", name="CMF"), row=7, col=1)

    if has_prob:
        fig.add_trace(go.Scatter(x=dfp.index, y=dfp["PROB_PCT"], mode="lines", name="Prob%"), row=8, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=1100 if has_prob else 980,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="left",
        legend_x=0,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    for r in range(1, rows + 1):
        fig.update_yaxes(showgrid=True, row=r, col=1)

    return fig

# -----------------------------
# UI
# -----------------------------
st.title("Optionable ETFs — EOD Screener (Daily)")
st.caption("Focused on screener + diamonds + reasons. Optional chart. No options chain table.")

with st.sidebar:
    st.header("Universe")
    use_auto_universe = st.checkbox("Auto-load ETF tickers", value=True)
    max_etfs = st.slider("Max ETFs to scan", 25, 1200, 250, 25)
    manual_tickers = st.text_area("Manual tickers (overrides auto if filled)", value="", height=70)

    st.header("Backtest target")
    years = st.slider("Years of daily history", 3, 20, 10)
    min_score = st.slider("Min score to trigger", 3, 12, 6)  # allows headroom if you add more signals later
    drop_pct = st.slider("Next-day CLOSE drop threshold (%)", 0.5, 3.0, 1.0, 0.1)

    st.header("Indicator params (daily)")
    tsi_fast = st.number_input("TSI fast", 1, 50, 6)
    tsi_slow = st.number_input("TSI slow", 1, 50, 3)
    tsi_signal = st.number_input("TSI signal", 1, 50, 6)
    rsi_len = st.number_input("RSI length", 2, 50, 14)
    cci_len = st.number_input("CCI length", 5, 50, 20)
    willr_len = st.number_input("Williams %R length", 5, 50, 14)
    cmf_len = st.number_input("CMF length", 5, 50, 20)

    st.header("Signal thresholds (editable)")
    tsi_thr = st.number_input("TSI must be >", 0.0, 100.0, 95.0, 1.0)
    rsi_thr = st.number_input("RSI must be >", 0.0, 100.0, 70.0, 1.0)
    cci_thr = st.number_input("CCI must be >", -300.0, 300.0, 125.0, 5.0)

    use_cci_regress = st.checkbox("Require CCI regressing (down N days)", value=True)
    cci_regress_days = st.slider("CCI regress days", 1, 5, 2)

    willr_thr = st.number_input("Williams %R must be >", -100.0, 0.0, -20.0, 1.0)
    cmf_thr = st.number_input("CMF must be <", -0.50, 0.50, 0.0, 0.01)

    st.header("VWAP proxy + candle")
    vwap_len = st.number_input("VWAP proxy length", 5, 100, 20)
    vwap_1 = st.number_input("VWAP stretch 1 (%)", 0.1, 10.0, 1.0, 0.1) / 100
    vwap_2 = st.number_input("VWAP stretch 2 (%)", 0.1, 10.0, 2.0, 0.1) / 100
    wick_thresh = st.number_input("Upper-wick exhaustion threshold", 0.1, 0.9, 0.50, 0.05)

    st.header("Screener filters")
    diamond_thr = st.slider("Diamond probability threshold (%)", 50, 100, 90)
    show_only_diamonds = st.checkbox("Show only Diamonds", value=False)
    min_hist_acc = st.slider("Min historical accuracy %", 0, 100, 0)

    st.header("Runtime")
    optionable_check_limit = st.slider("Optionable check max tickers", 25, 1200, 250, 25)

    run_btn = st.button("Run Screener", type="primary")

# Helpful info about optional deps
with st.expander("Environment notes (optional)", expanded=False):
    st.write(f"- Plotly charting available: **{PLOTLY_AVAILABLE}**")
    st.write(f"- pandas_ta available: **{PANDAS_TA_AVAILABLE}** (fallback indicators used if False)")
    if not PLOTLY_AVAILABLE:
        st.info("To enable charts, add: plotly>=5.22 to requirements.txt and redeploy.")

if not run_btn:
    st.info("Set your parameters in the sidebar and click **Run Screener**.")
    st.stop()

# -----------------------------
# Universe selection
# -----------------------------
manual = [t.strip().upper() for t in manual_tickers.replace(",", " ").split() if t.strip()]
manual = list(dict.fromkeys(manual))

if manual:
    tickers = manual
else:
    if use_auto_universe:
        tickers = fetch_all_etf_tickers()[:max_etfs]
    else:
        tickers = FALLBACK_ETFS

# -----------------------------
# Date window
# -----------------------------
end_dt = date.today()
start_dt = end_dt - timedelta(days=365 * years + 30)

st.write(f"Universe: **{len(tickers)} ETFs**. Filtering to **optionable** (options chain exists).")

# -----------------------------
# Optionable filter
# -----------------------------
tickers_for_check = tickers[:optionable_check_limit]
if len(tickers) > optionable_check_limit:
    st.warning(f"Optionable check limited to first {optionable_check_limit} tickers for runtime. Increase if desired.")

optionable: List[str] = []
p = st.progress(0)
for i, tkr in enumerate(tickers_for_check, start=1):
    p.progress(i / len(tickers_for_check))
    if is_optionable_yf(tkr):
        optionable.append(tkr)
p.empty()

st.write(f"Found **{len(optionable)} optionable ETFs** in this run.")

# -----------------------------
# Compute features + backtests
# -----------------------------
rows: List[Dict] = []
details: Dict[str, Dict] = {}

p = st.progress(0)
for i, tkr in enumerate(optionable, start=1):
    p.progress(i / max(1, len(optionable)))

    df = fetch_daily_ohlcv(tkr, start_dt.isoformat(), (end_dt + timedelta(days=1)).isoformat())
    if df.empty or len(df) < 160:
        continue

    feat = compute_features(
        df,
        tsi_fast=tsi_fast, tsi_slow=tsi_slow, tsi_signal=tsi_signal,
        rsi_len=rsi_len, cci_len=cci_len, willr_len=willr_len,
        cmf_len=cmf_len,
        vwap_len=vwap_len,
        vwap_1=vwap_1, vwap_2=vwap_2,
        wick_thresh=wick_thresh,
        min_score_trigger=min_score,
        tsi_thr=tsi_thr,
        rsi_thr=rsi_thr,
        cci_thr=cci_thr,
        willr_thr=willr_thr,
        cmf_thr=cmf_thr,
        use_cci_regress=use_cci_regress,
        cci_regress_days=cci_regress_days,
    )

    bt = backtest_next_day_drop(feat, min_score=min_score, drop_pct=drop_pct)
    latest = bt["latest"]

    rows.append({
        "Ticker": tkr,
        "Prob% (today)": float(latest.get("PROB_PCT", np.nan)),
        "Score (today)": int(latest.get("SCORE", 0)),
        "Signals": bt["signals"],
        "Wins": bt["wins"],
        "Accuracy%": float(bt["accuracy"]) if bt["signals"] > 0 else np.nan,
        "Close": float(latest.get("Close", np.nan)),
        # quick diagnostics (optional to show in screener)
        "TSI": float(latest.get("TSI", np.nan)),
        "RSI": float(latest.get("RSI", np.nan)),
        "CCI": float(latest.get("CCI", np.nan)),
        "W%R": float(latest.get("WILLR", np.nan)),
        "CMF": float(latest.get("CMF", np.nan)),
        "VW_pts": int(latest.get("VW_STRETCH", 0) if pd.notna(latest.get("VW_STRETCH", np.nan)) else 0),
        "Exhaust": int(latest.get("CANDLE_EXHAUST", 0) if pd.notna(latest.get("CANDLE_EXHAUST", np.nan)) else 0),
        "BearDiv": int(latest.get("BEAR_DIV", 0) if pd.notna(latest.get("BEAR_DIV", np.nan)) else 0),
        "CCI_Reg": int(latest.get("CCI_REGRESS", 0) if pd.notna(latest.get("CCI_REGRESS", np.nan)) else 0),
    })
    details[tkr] = bt

p.empty()

if not rows:
    st.warning("No results. Try increasing scan size or loosening filters.")
    st.stop()

res = pd.DataFrame(rows).sort_values(
    ["Prob% (today)", "Score (today)", "Accuracy%"],
    ascending=[False, False, False],
    na_position="last"
).reset_index(drop=True)

# Screener filters
if show_only_diamonds:
    res = res[res["Prob% (today)"] >= float(diamond_thr)]
if min_hist_acc > 0:
    res = res[res["Accuracy%"].notna() & (res["Accuracy%"] >= float(min_hist_acc))]
res = res.reset_index(drop=True)

if res.empty:
    st.warning("No ETFs match your current filters. Lower diamond threshold or min accuracy.")
    st.stop()

# Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Optionable scanned", f"{len(optionable)}")
c2.metric("Shown", f"{len(res)}")
c3.metric(f"Diamonds ≥{diamond_thr}%", f"{int((res['Prob% (today)'] >= float(diamond_thr)).sum())}")
c4.metric("Top Prob% today", f"{float(res['Prob% (today)'].max()):.1f}%")

# Screener table
st.subheader("Screener Results")

def highlight(row):
    if pd.isna(row["Prob% (today)"]):
        return [""] * len(row)
    if row["Prob% (today)"] >= float(diamond_thr):
        return ["background-color: #ffd6d6"] * len(row)  # diamond
    if row["Score (today)"] >= min_score:
        return ["background-color: #e7f7e7"] * len(row)  # signal
    return [""] * len(row)

styled = res.style.apply(highlight, axis=1).format({
    "Prob% (today)": "{:.1f}",
    "Accuracy%": "{:.1f}",
    "Close": "{:.2f}",
    "TSI": "{:.2f}",
    "RSI": "{:.2f}",
    "W%R": "{:.2f}",
    "CMF": "{:.3f}",
})

st.dataframe(styled, use_container_width=True, height=520)

diamonds = res.loc[res["Prob% (today)"] >= float(diamond_thr), "Ticker"].tolist()
st.write("**Diamonds (action list):** " + (", ".join(diamonds) if diamonds else "None"))

# Why it's a diamond (selected)
st.subheader("Why it’s a diamond (selected ETF)")
pick_reason = st.selectbox("Select ETF", options=res["Ticker"].tolist(), index=0, key="pick_reason")

bt_reason = details.get(pick_reason)
if bt_reason:
    latest = bt_reason["latest"]
    reasons = diamond_reasons(
        latest,
        tsi_thr=float(tsi_thr),
        rsi_thr=float(rsi_thr),
        cci_thr=float(cci_thr),
        willr_thr=float(willr_thr),
        cmf_thr=float(cmf_thr),
        use_cci_regress=bool(use_cci_regress),
    )

    st.write(
        f"**{pick_reason}** — Score: **{int(latest.get('SCORE', 0))} / {int(latest.get('MAX_SCORE', 0))}**"
        f" | Prob: **{float(latest.get('PROB_PCT', np.nan)):.1f}%**"
        f" | Close: **{float(latest.get('Close', np.nan)):.2f}**"
    )

    if reasons:
        st.markdown("\n".join([f"- {r}" for r in reasons]))
    else:
        st.write("- (No contributing reasons found — check your thresholds/settings.)")

# Optional chart (kept, but not required)
st.subheader("Chart (optional)")
pick_chart = st.selectbox("Select ETF for chart", options=res["Ticker"].tolist(), index=0, key="pick_chart")

bt_chart = details.get(pick_chart)
if not bt_chart:
    st.info("No chart data available for that symbol.")
else:
    if not PLOTLY_AVAILABLE:
        st.warning(
            "Plotly is not installed in this environment, so charts are disabled.\n\n"
            "Fix: add this line to requirements.txt and redeploy:\n"
            "- plotly>=5.22"
        )
    else:
        df_bt = bt_chart["df"].copy()
        fig = stockcharts_style_figure(df_bt, title=f"{pick_chart} — StockCharts-style panels", signal_score=int(min_score))
        st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Notes: Daily OHLCV. 'Optionable' is proxied by the presence of a Yahoo Finance options chain. "
    "VWAP is a rolling volume-weighted proxy (not intraday VWAP). Backtest target is next-day CLOSE drop."
)
