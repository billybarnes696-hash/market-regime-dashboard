
# app.py
# ------------------------------------------------------------
# Optionable ETF Mean-Reversion Optimizer & Screener
# Parallel Processing + Checkpointing + Progress Tracking
# ------------------------------------------------------------

from __future__ import annotations
import importlib
import random
import time
import json
import os
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Optional Dependencies
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

if not YFINANCE_AVAILABLE:
    st.error("Missing required package: yfinance. Please install it.")
    st.stop()

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_FILE = "optimization_checkpoint.json"
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# Fallback ETF List
# -----------------------------
FALLBACK_ETFS = [
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLI","XLY","XLU","XLV","XLP","XLB",
    "SMH","SOXX","ARKK","TLT","IEF","SHY","HYG","LQD","GLD","SLV","GDX",
    "USO","UNG","VNQ","EEM","VWO","FXI","EWJ","EWZ","EFA","VEA","VTI","VOO"
]

# -----------------------------
# Data Fetching (Cached)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_daily_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, threads=False)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c and c[0] else c[-1] for c in df.columns]
    
    df.columns = [str(c).strip().title() for c in df.columns]
    
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
        
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 250:
        return pd.DataFrame()
        
    return df

@st.cache_data(show_spinner=False, ttl=60 * 60)
def is_optionable_yf(ticker: str) -> bool:
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        return bool(exps and len(exps) > 0)
    except Exception:
        return False

# -----------------------------
# Indicators
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
    mom = close.diff()
    num = ema(ema(mom, slow), fast)
    den = ema(ema(mom.abs(), slow), fast)
    return 100 * (num / den.replace(0, np.nan))

# -----------------------------
# Feature Computation
# -----------------------------
def compute_features(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    out = df.copy()
    
    if PANDAS_TA_AVAILABLE:
        try:
            out["RSI"] = ta.rsi(out["Close"], length=params.get('rsi_len', 14))
            out["CCI"] = ta.cci(out["High"], out["Low"], out["Close"], length=params.get('cci_len', 20))
            out["WILLR"] = ta.willr(out["High"], out["Low"], out["Close"], length=params.get('willr_len', 14))
            out["CMF"] = ta.cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=params.get('cmf_len', 20))
            out["TSI"] = tsi(out["Close"], fast=params.get('tsi_fast', 6), slow=params.get('tsi_slow', 3))
        except Exception:
            out["TSI"] = tsi(out["Close"], fast=params.get('tsi_fast', 6), slow=params.get('tsi_slow', 3))
            out["RSI"] = rsi(out["Close"], length=params.get('rsi_len', 14))
            out["CCI"] = cci(out["High"], out["Low"], out["Close"], length=params.get('cci_len', 20))
            out["WILLR"] = willr(out["High"], out["Low"], out["Close"], length=params.get('willr_len', 14))
            out["CMF"] = cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=params.get('cmf_len', 20))
    else:
        out["TSI"] = tsi(out["Close"], fast=params.get('tsi_fast', 6), slow=params.get('tsi_slow', 3))
        out["RSI"] = rsi(out["Close"], length=params.get('rsi_len', 14))
        out["CCI"] = cci(out["High"], out["Low"], out["Close"], length=params.get('cci_len', 20))
        out["WILLR"] = willr(out["High"], out["Low"], out["Close"], length=params.get('willr_len', 14))
        out["CMF"] = cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=params.get('cmf_len', 20))

    pv = out["Close"] * out["Volume"]
    out["VWAP_PROXY"] = pv.rolling(params.get('vwap_len', 20)).sum() / out["Volume"].rolling(params.get('vwap_len', 20)).sum()
    
    upper_wick = out["High"] - np.maximum(out["Open"], out["Close"])
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    out["UPPER_WICK_PCT"] = (upper_wick / rng).clip(lower=0, upper=1)
    
    out["BEAR_DIV"] = ((out["Close"] > out["Close"].shift(1)) & (out["RSI"] < out["RSI"].shift(1))).astype(int)
    
    return out

def apply_thresholds(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    out = df.copy()
    
    vwap_1 = params.get('vwap_1', 0.01)
    vwap_2 = params.get('vwap_2', 0.02)
    out["VW_STRETCH"] = 0
    out.loc[out["Close"] > out["VWAP_PROXY"] * (1 + vwap_2), "VW_STRETCH"] = 2
    out.loc[(out["Close"] > out["VWAP_PROXY"] * (1 + vwap_1)) & (out["Close"] <= out["VWAP_PROXY"] * (1 + vwap_2)), "VW_STRETCH"] = 1
    
    wick_thresh = params.get('wick_thresh', 0.50)
    out["CANDLE_EXHAUST"] = (out["UPPER_WICK_PCT"] >= wick_thresh).astype(int)
    
    out["S_TSI"] = (out["TSI"] > float(params['tsi_thr'])).astype(int)
    out["S_RSI"] = (out["RSI"] > float(params['rsi_thr'])).astype(int)
    out["S_CCI"] = (out["CCI"] > float(params['cci_thr'])).astype(int)
    out["S_WILLR"] = (out["WILLR"] > float(params['willr_thr'])).astype(int)
    out["S_CMF"] = (out["CMF"] < float(params['cmf_thr'])).astype(int)
    
    if params.get('use_cci_regress', True):
        days = params.get('cci_regress_days', 2)
        cci_diff = out["CCI"].diff()
        out["CCI_REGRESS"] = ((cci_diff < 0).rolling(int(days)).sum() == int(days)).fillna(False).astype(int)
        out["S_CCI_REGRESS"] = out["CCI_REGRESS"]
    else:
        out["S_CCI_REGRESS"] = 0
        
    out["SCORE"] = (
        out["S_TSI"] + out["S_RSI"] + out["S_CCI"] + out["S_WILLR"] + out["S_CMF"] +
        out["VW_STRETCH"] + out["CANDLE_EXHAUST"] + out["BEAR_DIV"] + out["S_CCI_REGRESS"]
    )
    
    out["MAX_SCORE"] = 9 if params.get('use_cci_regress', True) else 8
    out["PROB_PCT"] = (out["SCORE"] / out["MAX_SCORE"] * 100).clip(0, 100)
    
    return out

def backtest_drop_window(df: pd.DataFrame, min_score: int, drop_pct: float, exit_window: int = 2) -> Dict:
    df = df.sort_index().copy()
    
    df["RET_D1"] = df["Close"].shift(-1) / df["Close"] - 1
    df["RET_D2"] = df["Close"].shift(-2) / df["Close"] - 1
    
    df["SIGNAL"] = df["SCORE"] >= min_score
    
    target = -abs(drop_pct) / 100.0
    df["HIT_D1"] = df["SIGNAL"] & (df["RET_D1"] <= target)
    df["HIT_D2"] = df["SIGNAL"] & (df["RET_D2"] <= target)
    df["HIT_EITHER"] = df["SIGNAL"] & ((df["RET_D1"] <= target) | (df["RET_D2"] <= target))
    
    signals = int(df["SIGNAL"].sum())
    if signals == 0:
        return {"signals": 0, "prob_either_pct": 0, "expectancy": 0, "latest": df.iloc[-1] if not df.empty else None}
    
    prob_d1 = df["HIT_D1"].sum() / signals * 100
    prob_d2 = df["HIT_D2"].sum() / signals * 100
    prob_either = df["HIT_EITHER"].sum() / signals * 100
    
    hit_returns = df.loc[df["HIT_EITHER"], ["RET_D1", "RET_D2"]].min(axis=1)
    avg_drop = hit_returns.mean() * 100 if not hit_returns.empty else 0
    
    miss_mask = df["SIGNAL"] & ~df["HIT_EITHER"]
    miss_returns = df.loc[miss_mask, ["RET_D1", "RET_D2"]].max(axis=1)
    avg_gain = miss_returns.mean() * 100 if not miss_returns.empty else 0
    
    hit_rate = prob_either / 100
    expectancy = (hit_rate * avg_drop) + ((1 - hit_rate) * avg_gain)
    
    return {
        "signals": signals,
        "prob_day1_pct": round(prob_d1, 1),
        "prob_day2_pct": round(prob_d2, 1),
        "prob_either_pct": round(prob_either, 1),
        "avg_drop_when_hit": round(avg_drop, 2),
        "avg_gain_when_miss": round(avg_gain, 2),
        "expectancy": round(expectancy, 3),
        "latest": df.iloc[-1]
    }

# -----------------------------
# Single Symbol Optimization
# -----------------------------
def optimize_single_symbol(ticker: str, df: pd.DataFrame, n_combos: int, drop_pct: float, min_signals: int) -> Dict:
    param_space = {
        'tsi_thr': list(range(85, 100, 1)),
        'cci_thr': list(range(100, 201, 5)),
        'rsi_thr': list(range(60, 86, 1)),
        'willr_thr': list(range(-30, -9, 1)),
        'cmf_thr': [round(x, 2) for x in np.arange(-0.10, 0.11, 0.02)],
        'cci_regress_days': [1, 2, 3],
        'min_score_trigger': list(range(5, 9)),
        'use_cci_regress': [True, False]
    }
    
    results = []
    for _ in range(n_combos):
        params = {k: random.choice(v) for k, v in param_space.items()}
        params.update({'rsi_len': 14, 'cci_len': 20, 'willr_len': 14, 'cmf_len': 20, 
                       'tsi_fast': 6, 'tsi_slow': 3, 'vwap_len': 20, 'vwap_1': 0.01, 
                       'vwap_2': 0.02, 'wick_thresh': 0.50})
        
        feat = apply_thresholds(df, params)
        bt = backtest_drop_window(feat, params['min_score_trigger'], drop_pct, exit_window=2)
        
        if bt['signals'] >= min_signals:
            score = (0.5 * bt['prob_either_pct']) + (0.5 * (-bt['expectancy'] * 10))
            results.append({**params, **bt, 'score': score, 'ticker': ticker})
            
    top = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    if not top:
        return {'ticker': ticker, 'personal_best': [], 'avg_prob': 0, 'avg_expectancy': 0, 'status': 'no_edge'}
        
    return {
        'ticker': ticker,
        'personal_best': top,
        'avg_prob': round(np.mean([r['prob_either_pct'] for r in top]), 1),
        'avg_expectancy': round(np.mean([r['expectancy'] for r in top]), 3),
        'status': 'complete'
    }

# -----------------------------
# Checkpoint Functions
# -----------------------------
def save_checkpoint(results: List[Dict], completed_tickers: List[str], checkpoint_file: str = CHECKPOINT_FILE):
    """Save progress every N tickers"""
    checkpoint = {
        'completed_tickers': completed_tickers,
        'results': results,
        'timestamp': date.today().isoformat()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, default=str)

def load_checkpoint(checkpoint_file: str = CHECKPOINT_FILE) -> Optional[Dict]:
    """Load previous progress if exists"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def clear_checkpoint(checkpoint_file: str = CHECKPOINT_FILE):
    """Clear checkpoint after successful completion"""
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

# -----------------------------
# Parallel Optimization Engine
# -----------------------------
def run_optimization_parallel(tickers: List[str], years: int, n_combos: int, drop_pct: float, 
                               min_signals: int, max_workers: int = 8, checkpoint_interval: int = 50) -> Dict:
    """Run optimization with parallel processing + checkpointing"""
    
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365 * years)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        completed_tickers = checkpoint.get('completed_tickers', [])
        existing_results = checkpoint.get('results', [])
        tickers_remaining = [t for t in tickers if t not in completed_tickers]
        st.info(f"📁 Resuming from checkpoint: {len(completed_tickers)} completed, {len(tickers_remaining)} remaining")
    else:
        completed_tickers = []
        existing_results = []
        tickers_remaining = tickers
    
    # Fetch data for remaining tickers
    symbol_data = {}
    fetch_progress = st.progress(0)
    
    for i, tkr in enumerate(tickers_remaining):
        df = fetch_daily_ohlcv(tkr, start_dt.isoformat(), (end_dt + timedelta(1)).isoformat())
        if not df.empty:
            feat = compute_features(df, {})
            symbol_data[tkr] = feat
        
        if (i + 1) % 10 == 0:
            fetch_progress.progress(min((i + 1) / len(tickers_remaining) * 0.3, 1.0))
    fetch_progress.empty()
    
    # Optimize in parallel
    all_results = existing_results.copy()
    optimize_progress = st.progress(0)
    status_text = st.empty()
    
    batch_results = []
    batch_tickers = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(optimize_single_symbol, tkr, df, n_combos, drop_pct, min_signals): tkr 
                   for tkr, df in symbol_data.items()}
        
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            tkr = futures[future]
            try:
                result = future.result()
                batch_results.append(result)
                batch_tickers.append(tkr)
                completed += 1
                
                # Update progress
                progress = 0.3 + (completed / total * 0.7)
                optimize_progress.progress(min(progress, 1.0))
                status_text.text(f"⚡ Optimizing: {completed}/{total} ETFs | Last: {tkr}")
                
                # Checkpoint every N tickers
                if len(batch_tickers) >= checkpoint_interval:
                    all_results.extend(batch_results)
                    completed_tickers.extend(batch_tickers)
                    save_checkpoint(all_results, completed_tickers)
                    st.success(f"💾 Checkpoint saved: {len(completed_tickers)} ETFs complete")
                    batch_results = []
                    batch_tickers = []
                    
            except Exception as e:
                st.error(f"❌ Error optimizing {tkr}: {str(e)}")
                completed += 1
    
    # Final save
    all_results.extend(batch_results)
    completed_tickers.extend(batch_tickers)
    save_checkpoint(all_results, completed_tickers)
    
    optimize_progress.empty()
    status_text.empty()
    
    # Clear checkpoint on successful completion
    if len(completed_tickers) >= len(tickers):
        clear_checkpoint()
        st.success("✅ Optimization complete! Checkpoint cleared.")
    
    # Aggregate results
    consensus = []
    if all_results:
        all_winners = []
        for res in all_results:
            for p in res.get('personal_best', []):
                sig = f"TSI>{p['tsi_thr']}_CCI>{p['cci_thr']}_RSI>{p['rsi_thr']}_Reg={p['use_cci_regress']}"
                all_winners.append({'sig': sig, 'params': p, 'prob': p['prob_either_pct']})
        
        from collections import Counter
        counts = Counter(w['sig'] for w in all_winners)
        for sig, count in counts.most_common(5):
            example = next((w['params'] for w in all_winners if f"TSI>{w['params']['tsi_thr']}" in sig), None)
            if example:
                avg_prob = np.mean([w['prob'] for w in all_winners if w['sig'] == sig])
                consensus.append({'signature': sig, 'count': count, 'avg_prob': round(avg_prob, 1), 'params': example})
    
    high_edge = sorted([r for r in all_results if r.get('avg_prob', 0) > 0], 
                       key=lambda x: x['avg_prob'], reverse=True)[:20]
    
    return {
        'symbol_results': all_results,
        'consensus': consensus,
        'high_edge': high_edge,
        'completed_count': len(completed_tickers),
        'total_count': len(tickers)
    }

# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(page_title="ETF Mean-Reversion Optimizer", layout="wide")
st.title("📉 ETF Mean-Reversion Optimizer & Screener")

# Initialize Session State
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = None
if 'live_params' not in st.session_state:
    st.session_state.live_params = None
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False

tab1, tab2 = st.tabs(["🔍 Optimize Parameters", "💎 Live Screener"])

# -----------------------------
# TAB 1: OPTIMIZE
# -----------------------------
with tab1:
    st.header("Batch Optimization Engine")
    st.markdown("""
    **Bearish Mode:** Finds overheated ETFs that historically drop ≥ X% within 1-2 days.
    
    - ✅ Parallel processing (8 workers)
    - ✅ Checkpoint every 50 ETFs (resume if interrupted)
    - ✅ Per-symbol sweet spots + aggregate consensus
    """)
    
    # Input Section
    col1, col2 = st.columns(2)
    with col1:
        paste_tickers = st.text_area(
            "Paste ETF/Stock Tickers (up to 500, one per line)",
            height=200,
            placeholder="SPY\nQQQ\nIWM\nTLT\nGLD\n..."
        )
        use_fallback = st.checkbox("Use fallback list (35 ETFs) for testing", value=False)
    
    with col2:
        opt_years = st.slider("Historical Lookback (Years)", 5, 25, 20)
        drop_pct = st.number_input("Target Drop % (Bearish)", 0.5, 5.0, 1.0, 0.1)
        n_combos = st.slider("Parameter Combos per ETF", 50, 500, 150)
        min_signals = st.slider("Min Signals Required", 5, 50, 15)
        max_workers = st.slider("Parallel Workers", 1, 16, 8)
    
    # Parse Tickers
    if use_fallback:
        tickers = FALLBACK_ETFS
    else:
        tickers = [t.strip().upper() for t in paste_tickers.replace(",", "\n").split("\n") if t.strip()]
        tickers = list(dict.fromkeys(tickers))[:500]
    
    st.write(f"📊 **{len(tickers)} tickers** queued for optimization")
    
    # Check for existing checkpoint
    existing_checkpoint = load_checkpoint()
    if existing_checkpoint:
        st.warning(f"💾 Found checkpoint from {existing_checkpoint.get('timestamp', 'unknown')}: {len(existing_checkpoint.get('completed_tickers', []))} ETFs already processed")
        if st.button("🗑️ Clear Checkpoint & Start Fresh"):
            clear_checkpoint()
            st.rerun()
    
    # Run Button
    if st.button("🚀 Run Optimization", type="primary", disabled=len(tickers) == 0):
        if len(tickers) == 0:
            st.error("Please enter tickers or use fallback list")
        else:
            with st.spinner("Starting optimization..."):
                try:
                    results = run_optimization_parallel(
                        tickers=tickers,
                        years=opt_years,
                        n_combos=n_combos,
                        drop_pct=drop_pct,
                        min_signals=min_signals,
                        max_workers=max_workers,
                        checkpoint_interval=50
                    )
                    
                    st.session_state.opt_results = results
                    st.session_state.optimization_complete = True
                    st.session_state.drop_pct = drop_pct
                    
                    st.success(f"✅ Optimization Complete! {results['completed_count']}/{results['total_count']} ETFs processed")
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    st.info("Checkpoint saved. Refresh and click 'Run Optimization' to resume.")
    
    # Display Results
    if st.session_state.opt_results:
        results = st.session_state.opt_results
        
        st.divider()
        st.subheader("📊 Aggregate Results")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ETFs Processed", f"{results['completed_count']}/{results['total_count']}")
        c2.metric("Avg Probability (1-2 Day Drop)", f"{np.mean([r['avg_prob'] for r in results['symbol_results'] if r['avg_prob'] > 0]):.1f}%")
        c3.metric("High-Edge ETFs (>65% Prob)", f"{len([r for r in results['symbol_results'] if r['avg_prob'] >= 65])}")
        c4.metric("Target Drop", f"-{st.session_state.get('drop_pct', 1.0)}%")
        
        # Consensus Params
        if results['consensus']:
            st.subheader("🌐 Top Consensus Parameter Sets")
            for i, cons in enumerate(results['consensus'][:3], 1):
                with st.expander(f"#{i} | Frequency: {cons['count']} ETFs | Avg Prob: {cons['avg_prob']}%"):
                    p = cons['params']
                    st.write(f"""
                    - **TSI >** {p['tsi_thr']}
                    - **CCI >** {p['cci_thr']}
                    - **RSI >** {p['rsi_thr']}
                    - **W%R >** {p['willr_thr']}
                    - **CMF <** {p['cmf_thr']}
                    - **CCI Regressing:** {p['use_cci_regress']} ({p['cci_regress_days']} days)
                    - **Min Score:** {p['min_score_trigger']}
                    """)
                    if st.button(f"Apply # {i} to Screener", key=f"apply_cons_{i}"):
                        st.session_state.live_params = p
                        st.session_state.param_source = "consensus"
                        st.success(f"✅ Consensus #{i} loaded! Switch to Live Screener tab.")
        
        # High-Edge Symbols
        if results['high_edge']:
            st.subheader("🎯 High-Edge ETFs (Top 20)")
            high_edge_df = pd.DataFrame([
                {
                    'Ticker': r['ticker'],
                    'Avg Prob (1-2d)': r['avg_prob'],
                    'Avg Expectancy': r['avg_expectancy'],
                    'Status': r['status']
                }
                for r in results['high_edge']
            ])
            st.dataframe(high_edge_df.style.format({
                'Avg Prob (1-2d)': '{:.1f}%',
                'Avg Expectancy': '{:.3f}%'
            }), use_container_width=True, height=400)
            
            # Per-symbol detail
            st.subheader("🔍 View Individual ETF Sweet Spots")
            selected_ticker = st.selectbox("Select ETF", options=[r['ticker'] for r in results['symbol_results']])
            selected_result = next((r for r in results['symbol_results'] if r['ticker'] == selected_ticker), None)
            
            if selected_result and selected_result.get('personal_best'):
                st.write(f"**{selected_ticker}** — Personal Best Parameters")
                best = selected_result['personal_best'][0]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Probability (1-2 Day Drop)", f"{best['prob_either_pct']}%")
                    st.metric("Expectancy", f"{best['expectancy']}%")
                    st.metric("Total Signals", best['signals'])
                with c2:
                    st.write(f"""
                    - TSI > {best['tsi_thr']}
                    - CCI > {best['cci_thr']}
                    - RSI > {best['rsi_thr']}
                    - Min Score: {best['min_score_trigger']}
                    """)
                if st.button(f"Apply {selected_ticker}'s Params to Screener", key=f"apply_{selected_ticker}"):
                    st.session_state.live_params = best
                    st.session_state.param_source = "personal"
                    st.session_state.param_ticker = selected_ticker
                    st.success(f"✅ {selected_ticker}'s personal params loaded!")

# -----------------------------
# TAB 2: LIVE SCREENER
# -----------------------------
with tab2:
    st.header("💎 Live Diamond Screener")
    
    if not st.session_state.optimization_complete:
        st.warning("⚠️ Run optimization first to load sweet spot parameters!")
        st.stop()
    
    # Param Source Info
    if st.session_state.live_params:
        source = st.session_state.get('param_source', 'unknown')
        if source == "personal":
            st.info(f"🎯 Using **personalized params** for {st.session_state.get('param_ticker', 'ETF')}")
        else:
            st.info("🌐 Using **consensus params** from optimization")
    else:
        st.info("🌐 Using default consensus params (run optimization to customize)")
        st.session_state.live_params = {
            'tsi_thr': 93, 'cci_thr': 125, 'rsi_thr': 71, 'willr_thr': -19,
            'cmf_thr': 0.01, 'use_cci_regress': True, 'cci_regress_days': 2,
            'min_score_trigger': 7, 'rsi_len': 14, 'cci_len': 20, 'willr_len': 14,
            'cmf_len': 20, 'tsi_fast': 6, 'tsi_slow': 3, 'vwap_len': 20,
            'vwap_1': 0.01, 'vwap_2': 0.02, 'wick_thresh': 0.50
        }
    
    # Get tickers from optimization results
    if st.session_state.opt_results:
        screener_tickers = [r['ticker'] for r in st.session_state.opt_results['symbol_results']]
    else:
        screener_tickers = FALLBACK_ETFS
    
    # Scan Button
    if st.button("🔍 Scan Live EOD Data", type="primary"):
        with st.spinner("Fetching live EOD data..."):
            end_dt = date.today()
            start_dt = end_dt - timedelta(days=365)
            
            live_results = []
            progress = st.progress(0)
            
            for i, tkr in enumerate(screener_tickers[:100]):  # Limit to 100 for speed
                df = fetch_daily_ohlcv(tkr, start_dt.isoformat(), (end_dt + timedelta(1)).isoformat())
                if df.empty:
                    continue
                
                feat = apply_thresholds(df, st.session_state.live_params)
                latest = feat.iloc[-1]
                
                if latest['SCORE'] >= st.session_state.live_params['min_score_trigger']:
                    # Get historical stats from optimization
                    opt_result = next((r for r in st.session_state.opt_results['symbol_results'] if r['ticker'] == tkr), None)
                    hist_prob = opt_result['avg_prob'] if opt_result else 0
                    
                    live_results.append({
                        'Ticker': tkr,
                        'Score': int(latest['SCORE']),
                        'Prob% (Historical)': hist_prob,
                        'TSI': round(latest['TSI'], 1),
                        'RSI': round(latest['RSI'], 1),
                        'CCI': round(latest['CCI'], 0),
                        'W%R': round(latest['WILLR'], 1),
                        'CMF': round(latest['CMF'], 3),
                        'Close': round(latest['Close'], 2)
                    })
                
                if (i + 1) % 10 == 0:
                    progress.progress(min((i + 1) / min(100, len(screener_tickers)), 1.0))
            
            progress.empty()
            
            if live_results:
                st.session_state.live_results = pd.DataFrame(live_results).sort_values(
                    ['Prob% (Historical)', 'Score'], ascending=[False, False]
                )
                st.success(f"✅ Found {len(live_results)} Diamond signals!")
            else:
                st.warning("No signals found with current params. Try lowering thresholds.")
    
    # Display Live Results
    if 'live_results' in st.session_state and st.session_state.live_results is not None:
        df_live = st.session_state.live_results
        
        st.subheader("📊 Live Diamond Signals")
        
        # Highlight high-prob diamonds
        def highlight_diamonds(row):
            if row['Prob% (Historical)'] >= 65:
                return ['background-color: #ffd6d6'] * len(row)
            elif row['Score'] >= 7:
                return ['background-color: #e7f7e7'] * len(row)
            return [''] * len(row)
        
        styled = df_live.style.apply(highlight_diamonds, axis=1).format({
            'Prob% (Historical)': '{:.1f}%',
            'Close': '${:.2f}',
            'TSI': '{:.1f}',
            'RSI': '{:.1f}',
            'CCI': '{:.0f}',
            'W%R': '{:.1f}',
            'CMF': '{:.3f}'
        })
        
        st.dataframe(styled, use_container_width=True, height=400)
        
        # Diamond Details
        if len(df_live) > 0:
            st.subheader("💎 Diamond Details")
            selected = st.selectbox("Select ETF for Details", options=df_live['Ticker'].tolist())
            selected_row = df_live[df_live['Ticker'] == selected].iloc[0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Historical Drop Probability (1-2 Days)", f"{selected_row['Prob% (Historical)']}%")
            c2.metric("Current Score", f"{selected_row['Score']}/9")
            c3.metric("Close Price", f"${selected_row['Close']}")
            
            st.write("**Why This ETF Triggered:**")
            reasons = []
            if selected_row['TSI'] > st.session_state.live_params['tsi_thr']:
                reasons.append(f"✓ TSI > {st.session_state.live_params['tsi_thr']} (current: {selected_row['TSI']})")
            if selected_row['RSI'] > st.session_state.live_params['rsi_thr']:
                reasons.append(f"✓ RSI > {st.session_state.live_params['rsi_thr']} (current: {selected_row['RSI']})")
            if selected_row['CCI'] > st.session_state.live_params['cci_thr']:
                reasons.append(f"✓ CCI > {st.session_state.live_params['cci_thr']} (current: {selected_row['CCI']})")
            if selected_row['W%R'] > st.session_state.live_params['willr_thr']:
                reasons.append(f"✓ W%R > {st.session_state.live_params['willr_thr']} (current: {selected_row['W%R']})")
            
            st.markdown("\n".join(reasons) if reasons else "- No specific reasons identified")
            
            # Chart
            if PLOTLY_AVAILABLE:
                st.subheader("📈 Price Chart")
                df_chart = fetch_daily_ohlcv(selected, (date.today() - timedelta(days=180)).isoformat(), 
                                            (date.today() + timedelta(1)).isoformat())
                if not df_chart.empty:
                    fig = go.Figure(data=[go.Candlestick(
                        x=df_chart.index,
                        open=df_chart['Open'],
                        high=df_chart['High'],
                        low=df_chart['Low'],
                        close=df_chart['Close'],
                        name=selected
                    )])
                    fig.update_layout(height=400, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("""
**Strategy:** Bearish mean-reversion. Signal triggers on overheated conditions. 
Historical probability shows % of times ETF dropped ≥ target within 1-2 days after signal.
**Execution:** Buy puts / Sell calls at EOD on signal. Exit Day+1 or Day+2 on target drop.
""")
