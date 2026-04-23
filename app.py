"""
QUANT EDGE FINDER + LIVE OSCILLATOR DASHBOARD
With RUN button - You control when analysis executes
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import product

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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

st.title("📊 Quant Edge Finder + Live Oscillator Dashboard")
st.caption("Enter symbol → Click RUN → Get analysis")


# ============================================
# SIDEBAR - INPUTS (No auto-run)
# ============================================
with st.sidebar:
    st.header("📊 Input")
    symbol = st.text_input("Symbol", value="SOXS").upper().strip()
    
    st.header("📅 Data Settings")
    days_back = st.slider("Days of history", 30, 180, 90)
    timeframe_minutes = st.selectbox("Timeframe (minutes)", [5, 10, 15, 30], index=0)
    
    st.header("🔑 API (Optional)")
    api_key = st.text_input("API Key", type="password", value=os.getenv("ALPACA_API_KEY", ""))
    secret_key = st.text_input("Secret Key", type="password", value=os.getenv("ALPACA_SECRET_KEY", ""))
    feed = st.selectbox("Feed", ["sip", "iex"], index=0)
    
    st.header("⚙️ Analysis Options")
    run_research = st.checkbox("Run full grid search", value=False)
    
    # RUN BUTTON - This is what you asked for
    st.divider()
    run_button = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)
    
    st.caption("Click RUN to fetch data and analyze")


# ============================================
# OSCILLATOR ENGINE
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
    def bbp(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
        ma = close.rolling(period, min_periods=2).mean()
        sd = close.rolling(period, min_periods=2).std()
        upper = ma + std * sd
        lower = ma - std * sd
        return (close - lower) / (upper - lower).replace(0, np.nan)
    
    @staticmethod
    def compute_all(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """Compute all oscillators."""
        result = df.copy()
        
        # Resample if needed (using 'min' instead of 'T' for pandas 3.0)
        if timeframe_minutes > 1:
            freq = f'{timeframe_minutes}min'
            try:
                result = result.resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            except:
                st.warning(f"Using minute data (resample to {freq} failed)")
        
        # Calculate oscillators
        result['rsi_14'] = OscillatorEngine.rsi(result['close'], 14)
        result['stoch_14'] = OscillatorEngine.stoch_k(result['close'], 14)
        result['cci_20'] = OscillatorEngine.cci(result, 20)
        result['roc_10'] = OscillatorEngine.roc(result['close'], 10)
        result['tsi'] = OscillatorEngine.tsi(result['close'], 25, 13)
        result['bbp_20'] = OscillatorEngine.bbp(result['close'], 20, 2.0)
        
        # Composite oscillator
        cci_norm = np.clip(result['cci_20'] / 150, -1, 1)
        tsi_norm = np.clip(result['tsi'] / 20, -1, 1)
        rsi_norm = (result['rsi_14'] - 50) / 50
        roc_norm = np.clip(result['roc_10'] / 10, -1, 1)
        stoch_norm = (result['stoch_14'] - 50) / 50
        
        result['composite'] = (cci_norm + tsi_norm + rsi_norm + roc_norm + stoch_norm) / 5
        result['composite_signal'] = OscillatorEngine.ema(result['composite'], 5)
        result['composite_hist'] = result['composite'] - result['composite_signal']
        
        # Forward returns
        for i in [1, 2, 3, 5]:
            result[f'forward_{i}'] = (result['close'].shift(-i) / result['close'] - 1) * 100
        
        return result


# ============================================
# DATA FETCHING
# ============================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol, days_back, api_key, secret_key, feed):
    """Fetch data from Alpaca."""
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
        
        bars = bars.reset_index(level=0, drop=True)
        return bars
        
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ============================================
# VISUALIZATION
# ============================================
def create_dashboard(df, symbol, timeframe_minutes):
    """Create the main dashboard."""
    
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    latest = df.iloc[-1]
    
    # Current metrics
    st.subheader(f"🎯 {symbol} - Current Oscillator State")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Composite", f"{latest['composite']:.3f}")
    with col2:
        st.metric("CCI(20)", f"{latest['cci_20']:.1f}")
    with col3:
        st.metric("TSI", f"{latest['tsi']:.1f}")
    with col4:
        st.metric("RSI(14)", f"{latest['rsi_14']:.1f}")
    with col5:
        st.metric("Stoch(14)", f"{latest['stoch_14']:.1f}")
    
    # Signal
    if latest['composite'] > 0 and latest['composite_signal'] > 0:
        st.success("🟢 BULLISH - Composite above signal line")
    elif latest['composite'] < 0 and latest['composite_signal'] < 0:
        st.error("🔴 BEARISH - Composite below signal line")
    else:
        st.warning("🟡 NEUTRAL - No clear signal")
    
    # Price chart
    st.subheader("📈 Price & Composite Oscillator")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25])
    
    # Price
    fig.add_trace(go.Candlestick(
        x=df.index[-100:],
        open=df['open'][-100:],
        high=df['high'][-100:],
        low=df['low'][-100:],
        close=df['close'][-100:],
        name=symbol
    ), row=1, col=1)
    
    # Composite
    fig.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['composite'][-100:],
        name='Composite',
        line=dict(color='blue', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['composite_signal'][-100:],
        name='Signal',
        line=dict(color='orange', width=1, dash='dot')
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Histogram
    colors = ['red' if h < 0 else 'green' for h in df['composite_hist'][-100:]]
    fig.add_trace(go.Bar(
        x=df.index[-100:],
        y=df['composite_hist'][-100:],
        name='Histogram',
        marker_color=colors,
        opacity=0.6
    ), row=3, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Composite", row=2, col=1)
    fig.update_yaxes(title_text="Histogram", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual oscillators
    st.subheader("📊 Individual Oscillators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index[-100:], y=df['rsi_14'][-100:], name='RSI(14)'))
        fig2.add_hline(y=30, line_dash="dash", line_color="green")
        fig2.add_hline(y=70, line_dash="dash", line_color="red")
        fig2.update_layout(height=300, title="RSI (14)")
        st.plotly_chart(fig2, use_container_width=True)
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index[-100:], y=df['stoch_14'][-100:], name='Stochastic'))
        fig4.add_hline(y=20, line_dash="dash", line_color="green")
        fig4.add_hline(y=80, line_dash="dash", line_color="red")
        fig4.update_layout(height=300, title="Stochastic (14)")
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index[-100:], y=df['cci_20'][-100:], name='CCI(20)'))
        fig3.add_hline(y=-100, line_dash="dash", line_color="green")
        fig3.add_hline(y=100, line_dash="dash", line_color="red")
        fig3.update_layout(height=300, title="CCI (20)")
        st.plotly_chart(fig3, use_container_width=True)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df.index[-100:], y=df['tsi'][-100:], name='TSI'))
        fig5.add_hline(y=0, line_dash="dash", line_color="gray")
        fig5.update_layout(height=300, title="TSI (25,13)")
        st.plotly_chart(fig5, use_container_width=True)


# ============================================
# GRID SEARCH
# ============================================
def run_grid_search(df):
    """Simple grid search for optimal parameters."""
    st.subheader("🔬 Parameter Grid Search")
    
    results = []
    total = 0
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Parameter ranges
    cci_levels = [-200, -150, -100, -50]
    tsi_levels = [-20, -15, -10, -5]
    hold_bars = [1, 2, 3, 5]
    
    total_combos = len(cci_levels) * len(tsi_levels) * len(hold_bars)
    
    for i, (cci_th, tsi_th, hold) in enumerate(product(cci_levels, tsi_levels, hold_bars)):
        status.text(f"Testing combo {i+1}/{total_combos}")
        progress_bar.progress((i+1)/total_combos)
        
        # Simple signal: CCI < threshold AND TSI < threshold
        mask = (df['cci_20'] < cci_th) & (df['tsi'] < tsi_th)
        signals = df[mask]
        
        if len(signals) > 20:
            returns = signals[f'forward_{hold}'].dropna()
            if len(returns) > 10:
                results.append({
                    'cci_th': cci_th,
                    'tsi_th': tsi_th,
                    'hold_bars': hold,
                    'trades': len(returns),
                    'win_rate': (returns > 0).mean() * 100,
                    'avg_return': returns.mean(),
                    'sharpe': returns.mean() / (returns.std() + 0.01),
                })
    
    progress_bar.empty()
    status.empty()
    
    if results:
        results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
        st.dataframe(results_df.head(10), use_container_width=True)
        return results_df
    else:
        st.warning("No statistically significant strategies found")
        return None


# ============================================
# MAIN
# ============================================
def main():
    # Only run when button is clicked
    if not run_button:
        st.info("👈 Enter symbol and settings, then click RUN ANALYSIS")
        return
    
    # Validate inputs
    if not symbol:
        st.error("Please enter a symbol")
        return
    
    # Check API keys
    if not api_key or not secret_key:
        st.warning("⚠️ Enter Alpaca API keys to fetch real data")
        st.info("Get free API keys from app.alpaca.markets")
        return
    
    # Fetch data
    with st.spinner(f"Fetching {days_back} days of {symbol} data..."):
        minute_df = fetch_data(symbol, days_back, api_key, secret_key, feed)
    
    if minute_df is None or minute_df.empty:
        st.error(f"No data returned for {symbol}. Check symbol and API keys.")
        return
    
    # Compute oscillators
    with st.spinner("Computing oscillators..."):
        df = OscillatorEngine.compute_all(minute_df, timeframe_minutes)
    
    st.success(f"✅ Loaded {len(df)} {timeframe_minutes}-minute bars from {df.index[0]} to {df.index[-1]}")
    
    # Create dashboard
    create_dashboard(df, symbol, timeframe_minutes)
    
    # Optional grid search
    if run_research:
        run_grid_search(df)


if __name__ == "__main__":
    main()
