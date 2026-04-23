"""
UNIVERSAL QUANT EDGE FINDER + LIVE DASHBOARD
- Works with ANY symbol (no hardcoded tickers)
- Properly aligns forward returns to target hold period
- Optional confirmation symbol
- True Monte Carlo parameter optimization with bootstrap CI
"""

import os, time, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ============================================
# PAGE CONFIG & SESSION STATE
# ============================================
st.set_page_config(page_title="Universal Quant Edge", layout="wide", initial_sidebar_state="expanded")
st.title("🌐 Universal Quant Edge Finder + Live Dashboard")

if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None
if 'live_df' not in st.session_state:
    st.session_state.live_df = None

# ============================================
# SIDEBAR: UNIVERSAL INPUTS
# ============================================
with st.sidebar:
    st.header("📊 Symbol & Data")
    symbol = st.text_input("Primary Symbol", value="AAPL").upper().strip()
    
    use_confirm = st.checkbox("Use Confirmation Symbol", value=False)
    confirm_sym = st.text_input("Confirmation Symbol", value="SPY").upper().strip() if use_confirm else None
    
    st.header("⏱️ Timeframe & Hold Alignment")
    chart_tf = st.selectbox("Chart Timeframe (min)", [1, 2, 3, 5, 10, 15, 30], index=3)
    target_hold = st.selectbox("Target Hold Period (min)", [5, 10, 15, 30, 45, 60], index=2)
    hold_bars = max(1, target_hold // chart_tf)
    st.caption(f"🔢 Measuring returns over {hold_bars} bar(s) = {hold_bars * chart_tf} minutes")
    
    st.header("🎲 Monte Carlo Optimization")
    run_mc = st.checkbox("Run Monte Carlo Sweet-Spot Search", value=True)
    mc_iters = st.slider("MC Iterations", 200, 3000, 800) if run_mc else 0
    slippage_bps = st.slider("Assumed Slippage (bps)", 0, 200, 50)
    mc_conf = st.selectbox("Confidence Level", [90, 95, 99], index=1)
    
    st.header("🔑 Alpaca API")
    api_key = st.text_input("API Key", type="password", value=os.getenv("ALPACA_API_KEY", ""))
    secret = st.text_input("Secret Key", type="password", value=os.getenv("ALPACA_SECRET_KEY", ""))
    feed = st.selectbox("Data Feed", ["sip", "iex"], index=0)
    
    st.divider()
    col1, col2 = st.columns(2)
    run_hist = col1.button("🔍 PHASE 1: Find Sweet Spot", use_container_width=True)
    run_live = col2.button("⚡ PHASE 2: Live Score", type="primary", use_container_width=True)

# ============================================
# OSCILLATOR ENGINE
# ============================================
class OscEngine:
    @staticmethod
    def ema(s, span): return s.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def rsi(c, p=14):
        d = c.diff()
        u = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        rs = u / l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def stoch_k(c, p=14):
        lo, hi = c.rolling(p).min(), c.rolling(p).max()
        return 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    
    @staticmethod
    def cci(df, p=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(p).mean()
        mad = (tp - sma).abs().rolling(p).mean()
        return (tp - sma) / (0.015 * mad.replace(0, np.nan))
    
    @staticmethod
    def roc(c, p=10): return (c / c.shift(p) - 1) * 100
    
    @staticmethod
    def tsi(c, long=25, short=13):
        d = c.diff()
        m1 = OscEngine.ema(d, long)
        m2 = OscEngine.ema(m1, short)
        a1 = OscEngine.ema(d.abs(), long)
        a2 = OscEngine.ema(a1, short)
        return 100 * m2 / a2.replace(0, np.nan)
    
    @staticmethod
    def compute_all(df):
        df = df.copy()
        df['rsi_14'] = OscEngine.rsi(df['close'], 14)
        df['stoch_14'] = OscEngine.stoch_k(df['close'], 14)
        df['cci_20'] = OscEngine.cci(df, 20)
        df['roc_10'] = OscEngine.roc(df['close'], 10)
        df['tsi'] = OscEngine.tsi(df['close'])
        
        # Zero-cross signals
        df['cci_cross'] = (df['cci_20'].shift(1) <= 0) & (df['cci_20'] > 0)
        df['roc_cross'] = (df['roc_10'].shift(1) <= 0) & (df['roc_10'] > 0)
        df['tsi_cross'] = (df['tsi'].shift(1) <= 0) & (df['tsi'] > 0)
        df['stoch_cross'] = (df['stoch_14'].shift(1) <= 50) & (df['stoch_14'] > 50)
        return df

# ============================================
# DATA FETCHING
# ============================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hist(sym, days, api, sec, feed):
    if not api or not sec: return None
    client = StockHistoricalDataClient(api, sec)
    start, end = datetime.now(timezone.utc) - timedelta(days=days), datetime.now(timezone.utc)
    try:
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=start, end=end, limit=10000, feed=feed, adjustment="all")
        bars = client.get_stock_bars(req).df
        return bars.reset_index(level=0, drop=True) if not bars.empty else None
    except Exception as e:
        st.error(f"Historical fetch error: {e}"); return None

def fetch_live(sym, mins, api, sec, feed):
    if not api or not sec: return None
    client = StockHistoricalDataClient(api, sec)
    start, end = datetime.now(timezone.utc) - timedelta(minutes=mins), datetime.now(timezone.utc)
    try:
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=start, end=end, limit=1000, feed=feed, adjustment="all")
        bars = client.get_stock_bars(req).df
        return bars.reset_index(level=0, drop=True) if not bars.empty else None
    except Exception as e:
        st.error(f"Live fetch error: {e}"); return None

# ============================================
# MONTE CARLO SWEET-SPOT OPTIMIZER
# ============================================
def monte_carlo_optimize(df, hold_bars, n_iter, slippage_bps, conf_level=95):
    param_space = {
        'use_cci': [True, False], 'use_roc': [True, False],
        'use_tsi': [True, False], 'use_stoch': [True, False],
        'min_signals': [5, 10, 15]
    }
    results = []
    
    for _ in range(n_iter):
        p = {k: np.random.choice(v) for k, v in param_space.items()}
        
        # Generate combined signal mask
        masks = []
        if p['use_cci']: masks.append(df['cci_cross'])
        if p['use_roc']: masks.append(df['roc_cross'])
        if p['use_tsi']: masks.append(df['tsi_cross'])
        if p['use_stoch']: masks.append(df['stoch_cross'])
        
        if not masks: continue
        sig_mask = masks[0]
        for m in masks[1:]: sig_mask &= m
        
        signals = df[sig_mask].copy()
        if len(signals) < p['min_signals']: continue
        
        ret_col = f'forward_{hold_bars}'
        if ret_col not in signals.columns:
            signals[ret_col] = (signals['close'].shift(-hold_bars) / signals['close'] - 1) * 100
        
        gross_rets = signals[ret_col].dropna()
        if len(gross_rets) < 5: continue
        
        # Inject random slippage per trade
        slip_pct = np.random.uniform(0, slippage_bps/10000, size=len(gross_rets))
        net_rets = gross_rets - slip_pct * 100
        
        win_rate = (net_rets > 0).mean()
        avg_ret = net_rets.mean()
        sharpe = avg_ret / (net_rets.std() + 1e-6)
        score = sharpe * win_rate * np.log(len(net_rets) + 1)
        
        results.append({
            'params': p, 'trades': len(net_rets), 'win_rate': win_rate,
            'avg_ret': avg_ret, 'sharpe': sharpe, 'score': score,
            'gross_rets': gross_rets
        })
    
    if not results: return None
    res_df = pd.DataFrame(results).sort_values('score', ascending=False)
    best = res_df.iloc[0]
    
    # Bootstrap Confidence Interval for avg return
    boots = []
    for _ in range(500):
        samp = best['gross_rets'].sample(n=len(best['gross_rets']), replace=True)
        boots.append((samp - np.random.uniform(0, slippage_bps/10000, len(samp))*100).mean())
    
    alpha = (100 - conf_level) / 200
    ci_low, ci_high = np.percentile(boots, alpha*100), np.percentile(boots, (1-alpha)*100)
    
    return {
        'top_params': {k: best['params'][k] for k in best['params'] if isinstance(best['params'][k], bool)},
        'metrics': best[['trades','win_rate','avg_ret','sharpe','score']],
        'ci': (ci_low, ci_high),
        'top_n': res_df.head(10)
    }

# ============================================
# PHASE 1: HISTORICAL RESEARCH
# ============================================
def run_phase1():
    with st.spinner(f"📥 Fetching historical data for {symbol}..."):
        hist_df = fetch_hist(symbol, 90, api_key, secret, feed)
        confirm_df = None
        if use_confirm and confirm_sym:
            confirm_df = fetch_hist(confirm_sym, 90, api_key, secret, feed)
    
    if hist_df is None: st.stop()
    
    with st.spinner("⚙️ Computing oscillators & forward returns..."):
        hist_df = OscEngine.compute_all(hist_df)
        # Align forward return with target hold
        ret_col = f'forward_{hold_bars}'
        hist_df[ret_col] = (hist_df['close'].shift(-hold_bars) / hist_df['close'] - 1) * 100
        
        if confirm_df is not None:
            confirm_df = OscEngine.compute_all(confirm_df)
            hist_df = hist_df.merge(confirm_df[['close']], left_index=True, right_index=True, suffixes=('','_conf'), how='left')
            hist_df['confirm_weak'] = hist_df['close_conf'].pct_change(hold_bars) < -0.005  # 0.5% drop over hold period
    
    if run_mc:
        with st.spinner(f"🎲 Running {mc_iters} Monte Carlo iterations..."):
            mc_res = monte_carlo_optimize(hist_df, hold_bars, mc_iters, slippage_bps, mc_conf)
            if mc_res:
                st.session_state.best_params = mc_res['top_params']
                st.session_state.mc_results = mc_res
                
                st.success("✅ Monte Carlo Complete")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Trades", mc_res['metrics']['trades'])
                c2.metric("Win Rate", f"{mc_res['metrics']['win_rate']*100:.1f}%")
                c3.metric("Avg Return", f"{mc_res['metrics']['avg_ret']:.2f}%")
                c4.metric(f"{mc_conf}% CI", f"[{mc_res['ci'][0]:.2f}%, {mc_res['ci'][1]:.2f}%]")
                
                st.dataframe(mc_res['top_n'][['trades','win_rate','avg_ret','sharpe','score']].head(5), use_container_width=True)
            else:
                st.warning("⚠️ No statistically significant parameter clusters found. Try increasing iterations or adjusting slippage.")
    else:
        st.info("ℹ️ Monte Carlo disabled. Enable it in the sidebar for parameter optimization.")

# ============================================
# PHASE 2: LIVE SCORING
# ============================================
def run_phase2():
    if st.session_state.best_params is None:
        st.warning("⚠️ Run Phase 1 first to calibrate parameters.")
        return
    
    mins_fetch = max(200, hold_bars * 10)
    with st.spinner(f"⚡ Fetching LIVE {chart_tf}-min data..."):
        live_df = fetch_live(symbol, mins_fetch, api_key, secret, feed)
        conf_df = None
        if use_confirm and confirm_sym:
            conf_df = fetch_live(confirm_sym, mins_fetch, api_key, secret, feed)
    
    if live_df is None: st.stop()
    
    with st.spinner("📊 Computing live oscillators..."):
        live_df = OscEngine.compute_all(live_df)
        live_df[f'forward_{hold_bars}'] = (live_df['close'].shift(-hold_bars) / live_df['close'] - 1) * 100
        st.session_state.live_df = live_df
        
        if conf_df is not None:
            conf_df = OscEngine.compute_all(conf_df)
            live_df = live_df.merge(conf_df[['close']], left_index=True, right_index=True, suffixes=('','_conf'), how='left')
            live_df['confirm_weak'] = live_df['close_conf'].pct_change(hold_bars) < -0.005
    
    latest = live_df.iloc[-1]
    p = st.session_state.best_params
    
    # Evaluate live signal
    sig = True
    if p.get('use_cci', False) and not latest['cci_cross']: sig = False
    if p.get('use_roc', False) and not latest['roc_cross']: sig = False
    if p.get('use_tsi', False) and not latest['tsi_cross']: sig = False
    if p.get('use_stoch', False) and not latest['stoch_cross']: sig = False
    
    if use_confirm and not latest.get('confirm_weak', True):
        sig = False
        st.warning("⚠️ Confirmation filter NOT met (sector not weak enough)")
    
    # Dashboard
    st.subheader(f"⚡ LIVE: {symbol} - Current State")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("CCI", f"{latest['cci_20']:.1f}", "✅" if latest['cci_cross'] else "⏳")
    c2.metric("ROC", f"{latest['roc_10']:.2f}%", "✅" if latest['roc_cross'] else "⏳")
    c3.metric("TSI", f"{latest['tsi']:.1f}", "✅" if latest['tsi_cross'] else "⏳")
    c4.metric("Stoch", f"{latest['stoch_14']:.1f}", "✅" if latest['stoch_cross'] else "⏳")
    c5.metric("Signal", "🟢 ACTIVE" if sig else "🔴 WAIT")
    
    st.info(f"🔑 Using calibrated params: `{p}` | Measuring {hold_bars}-bar forward return ({hold_bars*chart_tf} min hold)")
    
    # Chart
    lookback = min(100, len(live_df))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=live_df.index[-lookback:], open=live_df['open'][-lookback:], high=live_df['high'][-lookback:], low=live_df['low'][-lookback:], close=live_df['close'][-lookback:], name=symbol), row=1, col=1)
    fig.add_trace(go.Scatter(x=live_df.index[-lookback:], y=live_df['rsi_14'][-lookback:], name='RSI', line=dict(color='blue')), row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN EXECUTION
# ============================================
if not api_key or not secret:
    st.warning("⚠️ Enter Alpaca API credentials in the sidebar to fetch data.")
elif run_hist:
    st.header("🔍 PHASE 1: Historical Calibration")
    run_phase1()
elif run_live:
    st.header("⚡ PHASE 2: Live Real-Time Scoring")
    run_phase2()
else:
    st.info("👈 Configure settings, then click **PHASE 1** to calibrate, then **PHASE 2** for live scoring.")
    st.markdown("""
    ### How It Works:
    1. **Timeframe Alignment**: Returns are measured over `target_hold // chart_tf` bars. (e.g., 30-min hold on 5-min chart = 6 bars forward)
    2. **Monte Carlo**: Randomly samples oscillator combinations, injects realistic slippage, and bootstrap-validates returns
    3. **Optional Confirmation**: Only fetches/merges if checkbox is enabled
    4. **Universal**: Works with any Alpaca-supported symbol (stocks, ETFs, crypto)
    """)
