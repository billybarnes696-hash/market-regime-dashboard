"""
QUANT EDGE FINDER + LIVE OSCILLATOR DASHBOARD
Two-Phase: Historical Grid Search → Live Real-Time Scoring
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import product

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Quant Edge System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎯 Quant Edge: Historical Sweet Spot → Live Score")
st.caption("Phase 1: Find best zero-cross combos | Phase 2: Live real-time scoring")


# ============================================
# SESSION STATE INIT
# ============================================
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'historical_results' not in st.session_state:
    st.session_state.historical_results = None
if 'live_df' not in st.session_state:
    st.session_state.live_df = None


# ============================================
# SIDEBAR - INPUTS
# ============================================
with st.sidebar:
    st.header("📊 Symbol & Data")
    symbol = st.text_input("Primary Symbol", value="SOXS").upper().strip()
    confirm_symbol = st.text_input("Confirmation Symbol (e.g., SMH)", value="SMH").upper().strip()
    use_confirmation = st.checkbox("Use confirmation symbol filter", value=True)
    
    st.header("📅 Historical Research")
    hist_days = st.slider("Historical days for grid search", 30, 180, 90)
    hist_tf = st.selectbox("Historical timeframe (min)", [5, 10, 15], index=1)
    
    st.header("⚡ Live Scoring")
    live_tf = st.selectbox("Live timeframe (min)", [1, 2, 5, 10], index=2)
    lookback_bars = st.slider("Live lookback bars", 20, 200, 100)
    
    st.header("🔑 Alpaca API")
    api_key = st.text_input("API Key", type="password", value=os.getenv("ALPACA_API_KEY", ""))
    secret_key = st.text_input("Secret Key", type="password", value=os.getenv("ALPACA_SECRET_KEY", ""))
    feed = st.selectbox("Feed", ["sip", "iex"], index=0)
    
    st.header("⚙️ Strategy Options")
    inverse_etf = st.checkbox("Inverse ETF mode (flip signals)", value=True, help="For SOXS: bullish signal = semiconductor weakness")
    zero_cross_only = st.checkbox("Zero-cross signals only", value=True, help="CCI/ROC/TSI cross 0, Stoch cross 50")
    
    st.divider()
    
    # TWO PHASE BUTTONS
    st.subheader("🚀 Execute")
    col1, col2 = st.columns(2)
    with col1:
        run_hist = st.button("🔍 PHASE 1: Find Sweet Spot", type="secondary", use_container_width=True)
    with col2:
        run_live = st.button("⚡ PHASE 2: Live Score", type="primary", use_container_width=True)
    
    st.caption("Run Phase 1 first, then Phase 2 for live scoring")


# ============================================
# OSCILLATOR ENGINE (Zero-Cross Focused)
# ============================================
class OscillatorEngine:
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = OscillatorEngine.ema(up, period) / OscillatorEngine.ema(down, period).replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def stoch_k(close: pd.Series, period: int = 14) -> pd.Series:
        low = close.rolling(period, min_periods=2).min()
        high = close.rolling(period, min_periods=2).max()
        return 100 * (close - low) / (high - low).replace(0, np.nan)
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period, min_periods=2).mean()
        mad = (tp - sma).abs().rolling(period, min_periods=2).mean()
        return (tp - sma) / (0.015 * mad.replace(0, np.nan))
    
    @staticmethod
    def roc(close: pd.Series, period: int = 10) -> pd.Series:
        return (close / close.shift(period) - 1) * 100
    
    @staticmethod
    def tsi(close: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
        delta = close.diff()
        m1 = OscillatorEngine.ema(delta, long_period)
        m2 = OscillatorEngine.ema(m1, short_period)
        a1 = OscillatorEngine.ema(delta.abs(), long_period)
        a2 = OscillatorEngine.ema(a1, short_period)
        return 100 * m2 / a2.replace(0, np.nan)
    
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Compute zero-cross focused oscillators."""
        result = df.copy()
        
        # Core oscillators with user's preferred params
        result['rsi_14'] = OscillatorEngine.rsi(result['close'], 14)
        result['stoch_14'] = OscillatorEngine.stoch_k(result['close'], 14)
        result['cci_14'] = OscillatorEngine.cci(result, 14)  # User's preferred period
        result['roc_10'] = OscillatorEngine.roc(result['close'], 10)  # User's preferred period
        result['tsi'] = OscillatorEngine.tsi(result['close'], 15, 7)  # User's preferred params
        
        # Zero-cross signals (bullish = crossing above threshold)
        result['cci_cross_0'] = (result['cci_14'].shift(1) <= 0) & (result['cci_14'] > 0)
        result['roc_cross_0'] = (result['roc_10'].shift(1) <= 0) & (result['roc_10'] > 0)
        result['tsi_cross_0'] = (result['tsi'].shift(1) <= 0) & (result['tsi'] > 0)
        result['stoch_cross_50'] = (result['stoch_14'].shift(1) <= 50) & (result['stoch_14'] > 50)  # Midline cross
        
        # Composite (equal weight)
        cci_norm = np.clip(result['cci_14'] / 150, -1, 1)
        tsi_norm = np.clip(result['tsi'] / 20, -1, 1)
        rsi_norm = (result['rsi_14'] - 50) / 50
        roc_norm = np.clip(result['roc_10'] / 10, -1, 1)
        stoch_norm = (result['stoch_14'] - 50) / 50
        
        result['composite'] = (cci_norm + tsi_norm + rsi_norm + roc_norm + stoch_norm) / 5
        result['composite_signal'] = OscillatorEngine.ema(result['composite'], 5)
        
        # Forward returns for backtesting
        for i in [1, 2, 3, 5]:
            result[f'forward_{i}'] = (result['close'].shift(-i) / result['close'] - 1) * 100
        
        return result
    
    @staticmethod
    def generate_signal(row, params, inverse_etf=False):
        """Generate signal using best historical params."""
        signals = []
        
        # Zero-cross signals
        if params.get('cci_cross'):
            signals.append(row['cci_cross_0'])
        if params.get('roc_cross'):
            signals.append(row['roc_cross_0'])
        if params.get('tsi_cross'):
            signals.append(row['tsi_cross_0'])
        if params.get('stoch_cross'):
            signals.append(row['stoch_cross_50'])
        
        if not signals:
            return None
        
        # All selected signals must align
        signal = all(signals)
        
        # Inverse ETF logic: flip for SOXS
        if inverse_etf:
            signal = not signal  # Bullish on semis = bearish on SOXS
        
        return signal


# ============================================
# DATA FETCHING: CACHED vs LIVE
# ============================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(symbol, days_back, api_key, secret_key, feed):
    """Fetch historical data (CACHED for grid search)."""
    if not api_key or not secret_key:
        return None
    
    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=10000,
            feed=feed,
            adjustment="all",
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            return None
        return bars.reset_index(level=0, drop=True)
    except Exception as e:
        st.error(f"Historical fetch error: {e}")
        return None


def fetch_live_data(symbol, minutes_back, api_key, secret_key, feed):
    """Fetch LIVE data (NO CACHE - always fresh)."""
    if not api_key or not secret_key:
        return None
    
    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes_back)
    
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=1000,
            feed=feed,
            adjustment="all",
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            return None
        return bars.reset_index(level=0, drop=True)
    except Exception as e:
        st.error(f"Live fetch error: {e}")
        return None


# ============================================
# PHASE 1: HISTORICAL GRID SEARCH
# ============================================
def run_historical_research(symbol, confirm_symbol, days_back, tf_minutes, api_key, secret_key, feed, 
                           inverse_etf, use_confirmation, zero_cross_only):
    """Phase 1: Find best zero-cross combinations on historical data."""
    
    with st.spinner(f"🔍 Fetching {days_back} days of {symbol} history..."):
        df = fetch_historical_data(symbol, days_back, api_key, secret_key, feed)
        confirm_df = None
        if use_confirmation and confirm_symbol:
            confirm_df = fetch_historical_data(confirm_symbol, days_back, api_key, secret_key, feed)
    
    if df is None or df.empty:
        st.error("Failed to fetch historical data")
        return None
    
    with st.spinner("⚙️ Computing oscillators..."):
        df = OscillatorEngine.compute_all(df)
        if confirm_df is not None:
            confirm_df = OscillatorEngine.compute_all(confirm_df)
            # Merge for confirmation filter
            df = pd.merge(df, confirm_df[['close']], left_index=True, right_index=True, 
                         suffixes=('', '_confirm'), how='left')
            df['confirm_weak'] = df['close_confirm'].pct_change(2) < -0.005  # 2-bar drop
    
    st.success(f"✅ Loaded {len(df)} {tf_minutes}-min bars for research")
    
    # Grid search parameters (zero-cross focused)
    st.subheader("🔬 Testing Zero-Cross Combinations")
    
    # Only test zero-cross combos if requested
    if zero_cross_only:
        combos = list(product([True, False], repeat=4))  # cci, roc, tsi, stoch
        combo_names = ['CCI×0', 'ROC×0', 'TSI×0', 'Stoch×50']
    else:
        # Include level-based signals too
        combos = list(product([-200,-150,-100], [-20,-15,-10], [1,2,3]))
        combo_names = ['CCI_level', 'TSI_level', 'Hold_bars']
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for i, combo in enumerate(combos):
        if zero_cross_only:
            params = dict(zip(['cci_cross','roc_cross','tsi_cross','stoch_cross'], combo))
            # Skip if no signals selected
            if not any(combo):
                continue
            # Generate signals
            df['signal'] = df.apply(lambda row: OscillatorEngine.generate_signal(row, params, inverse_etf), axis=1)
            # Apply confirmation filter
            if use_confirmation and 'confirm_weak' in df.columns:
                df['signal'] = df['signal'] & df['confirm_weak'].fillna(True)
            signals = df[df['signal'] & df['signal'].notna()]
        else:
            # Level-based search (simplified)
            cci_th, tsi_th, hold = combo
            mask = (df['cci_14'] < cci_th) & (df['tsi'] < tsi_th)
            if use_confirmation and 'confirm_weak' in df.columns:
                mask = mask & df['confirm_weak'].fillna(True)
            signals = df[mask]
            params = {'cci_th': cci_th, 'tsi_th': tsi_th, 'hold': hold}
        
        if len(signals) >= 10:
            returns = signals['forward_1'].dropna()
            if len(returns) >= 10:
                results.append({
                    'params': str(params),
                    'trades': len(returns),
                    'win_rate': (returns > 0).mean() * 100,
                    'avg_return': returns.mean(),
                    'total_return': returns.sum(),
                    'sharpe': returns.mean() / (returns.std() + 0.01),
                })
        
        progress.progress((i+1)/len(combos))
        status.text(f"Testing {i+1}/{len(combos)} combos...")
    
    progress.empty()
    status.empty()
    
    if results:
        results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
        st.dataframe(results_df.head(10), use_container_width=True)
        
        # Store best params
        best = results_df.iloc[0]
        st.session_state.best_params = eval(best['params']) if zero_cross_only else best['params']
        st.session_state.historical_results = results_df
        st.success(f"🏆 Best combo: {best['params']} | Win Rate: {best['win_rate']:.1f}% | Sharpe: {best['sharpe']:.2f}")
        return results_df
    else:
        st.warning("No statistically significant strategies found")
        return None


# ============================================
# PHASE 2: LIVE SCORING
# ============================================
def run_live_scoring(symbol, confirm_symbol, live_tf, lookback_bars, api_key, secret_key, feed,
                    inverse_etf, use_confirmation):
    """Phase 2: Live real-time scoring using best historical params."""
    
    if st.session_state.best_params is None:
        st.warning("⚠️ Run Phase 1 first to find best parameters")
        return
    
    minutes_fetch = lookback_bars * live_tf + 50  # Buffer
    
    with st.spinner(f"⚡ Fetching LIVE {live_tf}-min data for {symbol}..."):
        # FORCE FRESH FETCH (no cache)
        df = fetch_live_data(symbol, minutes_fetch, api_key, secret_key, feed)
        confirm_df = None
        if use_confirmation and confirm_symbol:
            confirm_df = fetch_live_data(confirm_symbol, minutes_fetch, api_key, secret_key, feed)
    
    if df is None or df.empty:
        st.error("Failed to fetch live data")
        return
    
    with st.spinner("📊 Computing live oscillators..."):
        df = OscillatorEngine.compute_all(df)
        if confirm_df is not None:
            confirm_df = OscillatorEngine.compute_all(confirm_df)
            df = pd.merge(df, confirm_df[['close']], left_index=True, right_index=True, 
                         suffixes=('', '_confirm'), how='left')
            df['confirm_weak'] = df['close_confirm'].pct_change(2) < -0.005
    
    st.session_state.live_df = df  # Store for dashboard
    latest = df.iloc[-1]
    
    # Current signal using best params
    signal = OscillatorEngine.generate_signal(latest, st.session_state.best_params, inverse_etf)
    
    # Apply confirmation filter live
    if use_confirmation and 'confirm_weak' in df.columns and latest['confirm_weak'] == False:
        signal = False
        st.warning("⚠️ Confirmation filter not met (semis not weak enough)")
    
    # Display live dashboard
    st.subheader(f"⚡ LIVE: {symbol} - Current Oscillator State")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Composite", f"{latest['composite']:.3f}")
    with col2:
        st.metric("CCI(14)", f"{latest['cci_14']:.1f}")
    with col3:
        st.metric("TSI", f"{latest['tsi']:.1f}")
    with col4:
        st.metric("RSI(14)", f"{latest['rsi_14']:.1f}")
    with col5:
        st.metric("Stoch(14)", f"{latest['stoch_14']:.1f}")
    
    # Signal display
    if signal is True:
        st.success("🟢 LIVE SIGNAL: ENTER LONG" if not inverse_etf else "🟢 LIVE SIGNAL: ENTER SHORT (Inverse)")
    elif signal is False:
        st.error("🔴 NO SIGNAL: Wait for alignment")
    else:
        st.warning("🟡 INSUFFICIENT DATA")
    
    # Zero-cross status
    st.subheader("🎯 Zero-Cross Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "✅" if latest['cci_cross_0'] else "⏳"
        st.metric(f"{status} CCI×0", f"{latest['cci_14']:.1f}")
    with col2:
        status = "✅" if latest['roc_cross_0'] else "⏳"
        st.metric(f"{status} ROC×0", f"{latest['roc_10']:.2f}%")
    with col3:
        status = "✅" if latest['tsi_cross_0'] else "⏳"
        st.metric(f"{status} TSI×0", f"{latest['tsi']:.1f}")
    with col4:
        status = "✅" if latest['stoch_cross_50'] else "⏳"
        st.metric(f"{status} Stoch×50", f"{latest['stoch_14']:.1f}")
    
    # Price + Composite chart
    st.subheader("📈 Live Price & Composite")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Price
    fig.add_trace(go.Candlestick(
        x=df.index[-lookback_bars:],
        open=df['open'][-lookback_bars:],
        high=df['high'][-lookback_bars:],
        low=df['low'][-lookback_bars:],
        close=df['close'][-lookback_bars:],
        name=symbol
    ), row=1, col=1)
    
    # Composite
    fig.add_trace(go.Scatter(
        x=df.index[-lookback_bars:],
        y=df['composite'][-lookback_bars:],
        name='Composite',
        line=dict(color='blue', width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index[-lookback_bars:],
        y=df['composite_signal'][-lookback_bars:],
        name='Signal',
        line=dict(color='orange', width=1, dash='dot')
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Composite", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best params reminder
    st.info(f"🔑 Using best historical params: {st.session_state.best_params}")


# ============================================
# MAIN
# ============================================
def main():
    if not api_key or not secret_key:
        st.warning("⚠️ Enter Alpaca API keys to fetch data")
        st.info("Get free keys: app.alpaca.markets")
        return
    
    # Phase 1: Historical Research
    if run_hist:
        st.header("🔍 PHASE 1: Historical Grid Search")
        results = run_historical_research(
            symbol, confirm_symbol, hist_days, hist_tf, 
            api_key, secret_key, feed, inverse_etf, use_confirmation, zero_cross_only
        )
        if results is not None:
            st.success("✅ Phase 1 complete! Now run Phase 2 for live scoring.")
    
    # Phase 2: Live Scoring
    if run_live:
        st.header("⚡ PHASE 2: Live Real-Time Scoring")
        run_live_scoring(
            symbol, confirm_symbol, live_tf, lookback_bars,
            api_key, secret_key, feed, inverse_etf, use_confirmation
        )
    
    # Initial state
    if not run_hist and not run_live:
        st.info("👈 Configure settings above, then click Phase 1 → Phase 2")
        st.markdown("""
        ### How to Use:
        1. **Phase 1**: Click "Find Sweet Spot" to grid-search historical zero-cross combinations
        2. **Review**: See which oscillator combo had best Sharpe/win rate
        3. **Phase 2**: Click "Live Score" to apply best params to fresh real-time data
        4. **Trade**: Use live signal + zero-cross status for entry timing
        """)


if __name__ == "__main__":
    main()
