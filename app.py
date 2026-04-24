#!/usr/bin/env python3
"""
Stable Market Engine v13.3 - Simplified Working Version
✅ Direct 1-hour bars from Alpaca
✅ Consistent oscillator calculations
✅ Clean signal generation
✅ Multi-timeframe (1H, Daily, Weekly)
"""

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# -----------------------------
# CONFIG & SETUP
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Market Engine v13.3", layout="wide")

# -----------------------------
# SIMPLE OSCILLATOR
# -----------------------------
def calculate_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    """Simple and reliable oscillator based on RSI, MACD, and Price Action"""
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Price vs Moving Average
    df['sma20'] = df['Close'].rolling(20).mean()
    df['price_vs_sma'] = (df['Close'] / df['sma20'] - 1) * 100
    
    # Rate of Change
    df['roc'] = df['Close'].pct_change(5) * 100
    
    # Volume trend
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Combined oscillator (normalized to -1 to 1 range)
    rsi_norm = (df['rsi'] - 50) / 50
    roc_norm = np.clip(df['roc'] / 10, -1, 1)
    price_norm = np.clip(df['price_vs_sma'] / 10, -1, 1)
    
    df['oscillator'] = (rsi_norm * 0.5 + roc_norm * 0.3 + price_norm * 0.2)
    df['signal'] = df['oscillator'].ewm(span=5).mean()
    df['gap'] = df['oscillator'] - df['signal']
    
    # Forward returns for Monte Carlo
    for n in [1, 2, 5]:
        df[f'fwd_ret_{n}d'] = df['Close'].shift(-n) / df['Close'] - 1
    
    return df

# -----------------------------
# DATA FETCHING
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hourly_bars(symbol: str, months: int, key: str, secret: str) -> pd.DataFrame:
    """Fetch 1-hour bars from Alpaca"""
    cache_file = CACHE_DIR / f"{symbol}_hourly.parquet"
    
    # Check cache
    if cache_file.exists():
        age = pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit='s')
        if age < pd.Timedelta(hours=4):
            try:
                return pd.read_parquet(cache_file)
            except:
                pass
    
    try:
        client = StockHistoricalDataClient(key, secret)
        end = pd.Timestamp.now()
        start = end - pd.DateOffset(months=months)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=start,
            end=end,
            adjustment='raw'
        )
        
        bars = client.get_stock_bars(request).df
        
        if symbol in bars.index.get_level_values(0):
            df = bars.xs(symbol, level=0).copy()
        else:
            df = bars.copy()
        
        # Clean up
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Keep only necessary columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Filter market hours
        df = df.between_time('09:30', '16:00')
        
        # Handle any NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Cache
        df.to_parquet(cache_file)
        return df
        
    except Exception as e:
        st.warning(f"Error fetching hourly {symbol}: {str(e)[:100]}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_bars(symbols: List[str], years: int, key: str, secret: str) -> Dict[str, pd.DataFrame]:
    """Fetch daily bars for multiple symbols"""
    result = {}
    
    for symbol in symbols:
        cache_file = CACHE_DIR / f"{symbol}_daily.parquet"
        
        if cache_file.exists():
            age = pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit='s')
            if age < pd.Timedelta(hours=12):
                try:
                    result[symbol] = pd.read_parquet(cache_file)
                    continue
                except:
                    pass
        
        try:
            client = StockHistoricalDataClient(key, secret)
            end = pd.Timestamp.now()
            start = end - pd.DateOffset(years=years)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment='raw'
            )
            
            bars = client.get_stock_bars(request).df
            
            if symbol in bars.index.get_level_values(0):
                df = bars.xs(symbol, level=0).copy()
            else:
                df = bars.copy()
            
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Handle NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            df.to_parquet(cache_file)
            result[symbol] = df
            
        except Exception as e:
            st.warning(f"Error fetching daily {symbol}: {str(e)[:100]}")
            result[symbol] = pd.DataFrame()
    
    return result

# -----------------------------
# SIGNAL GENERATION
# -----------------------------
def get_signal(row) -> Tuple[str, float]:
    """Generate signal based on oscillator"""
    if row is None or pd.isna(row.get('oscillator', 0)):
        return "NEUTRAL", 0
    
    osc = row.get('oscillator', 0)
    gap = row.get('gap', 0)
    
    if osc > 0.3 and gap > 0:
        return "CALL", osc
    elif osc < -0.3 and gap < 0:
        return "PUT", osc
    elif osc > 0.1:
        return "WEAK CALL", osc
    elif osc < -0.1:
        return "WEAK PUT", osc
    else:
        return "NEUTRAL", osc

def compute_grade(daily_strength: float, rsi: float) -> Tuple[str, str]:
    """Compute grade based on oscillator strength and RSI"""
    if daily_strength > 0.4:
        if rsi < 70:
            return "A+", "Strong Bullish / Healthy"
        else:
            return "B+", "Bullish / Extended"
    elif daily_strength > 0.2:
        return "A", "Bullish"
    elif daily_strength > 0:
        return "B", "Moderate Bullish"
    elif daily_strength > -0.2:
        return "C", "Neutral"
    elif daily_strength > -0.4:
        return "D", "Bearish"
    else:
        return "F", "Strong Bearish"

# -----------------------------
# MONTE CARLO SIMULATION
# -----------------------------
def monte_carlo_from_history(df: pd.DataFrame, horizon_days: int = 2, n_sims: int = 1000) -> Dict:
    """Simple Monte Carlo simulation based on historical returns"""
    if df.empty or len(df) < 50:
        return {'p_up': 50, 'p_down': 50, 'median': 0, 'mean': 0}
    
    returns = df['Close'].pct_change(horizon_days).dropna()
    
    if len(returns) < 10:
        return {'p_up': 50, 'p_down': 50, 'median': 0, 'mean': 0}
    
    # Bootstrap sampling
    np.random.seed(42)
    samples = np.random.choice(returns.values, size=(n_sims, len(returns)), replace=True)
    sim_returns = samples.mean(axis=1)
    
    return {
        'p_up': float((sim_returns > 0).mean() * 100),
        'p_down': float((sim_returns < 0).mean() * 100),
        'median': float(np.median(sim_returns)),
        'mean': float(np.mean(sim_returns))
    }

# -----------------------------
# MAIN APP
# -----------------------------
st.title("📈 Market Engine v13.3 - Multi-Timeframe Oscillator")
st.caption("1-Hour, Daily, and Weekly signals from consistent oscillator | Monte Carlo predictions")

with st.sidebar:
    st.header("🔑 Alpaca Credentials")
    api_key = st.text_input("API Key", type="password")
    secret_key = st.text_input("Secret Key", type="password")
    feed = st.selectbox("Data Feed", ["iex", "sip"], index=0, help="IEX is free, SIP requires premium")
    
    st.header("📊 Symbol Input")
    symbols_text = st.text_area("Enter symbols (comma separated)", "QQQ, SMH, NVDA, XLF", height=100)
    csv_file = st.file_uploader("Or upload CSV with symbols", type=['csv'])
    
    st.header("⚙️ Settings")
    lookback_months = st.slider("Hourly data lookback (months)", 1, 6, 3)
    lookback_years = st.slider("Daily data lookback (years)", 2, 5, 3)
    
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Parse symbols
symbols = []
if symbols_text.strip():
    symbols = [s.strip().upper() for s in symbols_text.replace('\n', ',').split(',') if s.strip()]
if csv_file is not None:
    try:
        df_csv = pd.read_csv(io.BytesIO(csv_file.getvalue()))
        if 'symbol' in df_csv.columns:
            symbols.extend([s.strip().upper() for s in df_csv['symbol'].dropna().tolist()])
        elif 'ticker' in df_csv.columns:
            symbols.extend([s.strip().upper() for s in df_csv['ticker'].dropna().tolist()])
        else:
            symbols.extend([str(s).strip().upper() for s in df_csv.iloc[:, 0].dropna().tolist()])
    except:
        st.error("Error parsing CSV file")

symbols = list(dict.fromkeys(symbols))  # Remove duplicates

if not run_button:
    st.info("👈 Enter your Alpaca API credentials and click 'Run Analysis'")
    st.stop()

if not api_key or not secret_key:
    st.error("❌ Please enter your Alpaca API credentials")
    st.stop()

if not symbols:
    st.error("❌ Please enter at least one symbol")
    st.stop()

# Fetch data
with st.spinner(f"📡 Fetching market data for {len(symbols)} symbols..."):
    # Add benchmark
    all_symbols = list(set(symbols + ['SPY']))
    daily_data = fetch_daily_bars(all_symbols, lookback_years, api_key, secret_key)
    
    hourly_data = {}
    progress_bar = st.progress(0)
    for i, sym in enumerate(symbols):
        hourly_data[sym] = fetch_hourly_bars(sym, lookback_months, api_key, secret_key)
        progress_bar.progress((i + 1) / len(symbols))
    progress_bar.empty()

# Calculate indicators and generate results
results = []
detail_data = {}

for sym in symbols:
    hourly = hourly_data.get(sym, pd.DataFrame())
    daily = daily_data.get(sym, pd.DataFrame())
    
    if hourly.empty or daily.empty:
        st.warning(f"⚠️ Insufficient data for {sym}")
        continue
    
    # Calculate oscillators
    hourly_osc = calculate_oscillator(hourly)
    daily_osc = calculate_oscillator(daily)
    
    # Weekly from daily
    if not daily.empty:
        weekly = daily.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        weekly_osc = calculate_oscillator(weekly)
    else:
        weekly_osc = pd.DataFrame()
    
    # Get latest signals
    if len(hourly_osc) > 0:
        last_hourly = hourly_osc.iloc[-1]
        hourly_signal, hourly_strength = get_signal(last_hourly)
    else:
        hourly_signal, hourly_strength = "NO DATA", 0
    
    if len(daily_osc) > 0:
        last_daily = daily_osc.iloc[-1]
        daily_signal, daily_strength = get_signal(last_daily)
        daily_rsi = last_daily.get('rsi', 50)
        
        # Monte Carlo prediction
        mc_2d = monte_carlo_from_history(daily_osc, 2)
        mc_5d = monte_carlo_from_history(daily_osc, 5)
    else:
        daily_signal, daily_strength = "NO DATA", 0
        daily_rsi = 50
        mc_2d = {'p_up': 50, 'p_down': 50, 'median': 0}
        mc_5d = {'p_up': 50, 'p_down': 50, 'median': 0}
    
    if len(weekly_osc) > 0:
        last_weekly = weekly_osc.iloc[-1]
        weekly_signal, weekly_strength = get_signal(last_weekly)
    else:
        weekly_signal, weekly_strength = "NO DATA", 0
    
    # Compute grade
    grade, grade_reason = compute_grade(daily_strength, daily_rsi)
    
    # Net bias (probability advantage)
    net_bias = mc_2d.get('p_up', 50) - 50
    
    results.append({
        'Symbol': sym,
        'Grade': grade,
        'Grade Reason': grade_reason,
        '1H Signal': hourly_signal,
        'Daily Signal': daily_signal,
        'Weekly Signal': weekly_signal,
        'RSI (Daily)': round(daily_rsi, 1),
        'Oscillator': round(daily_strength, 2),
        'Prob Up 2D %': round(mc_2d.get('p_up', 50), 1),
        'Prob Down 2D %': round(mc_2d.get('p_down', 50), 1),
        'Net Bias %': round(net_bias, 1),
        'MC 2D Med %': round(mc_2d.get('median', 0) * 100, 2),
        'MC 5D Med %': round(mc_5d.get('median', 0) * 100, 2),
        'Status': 'OK'
    })
    
    # Store for detail view
    detail_data[sym] = {
        'hourly': hourly_osc,
        'daily': daily_osc,
        'weekly': weekly_osc,
        'mc_2d': mc_2d,
        'mc_5d': mc_5d
    }

# Display results
if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Oscillator', ascending=False)
    
    st.subheader("📊 Ranked Results")
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="market_analysis.csv",
        mime="text/csv"
    )
    
    # Detail view
    st.subheader("🔍 Detailed Chart View")
    selected = st.selectbox("Select symbol for detailed charts", df_results['Symbol'].tolist())
    
    if selected and selected in detail_data:
        data = detail_data[selected]
        hourly = data['hourly']
        daily = data['daily']
        weekly = data['weekly']
        
        if not hourly.empty and not daily.empty:
            # Create subplot
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    f"{selected} - Daily Price",
                    "1-Hour Oscillator",
                    "Daily Oscillator",
                    "Weekly Oscillator"
                ),
                vertical_spacing=0.08,
                row_heights=[0.35, 0.22, 0.22, 0.21]
            )
            
            # Price chart (daily)
            daily_last_90 = daily.tail(90)
            fig.add_trace(
                go.Candlestick(
                    x=daily_last_90.index,
                    open=daily_last_90['Open'],
                    high=daily_last_90['High'],
                    low=daily_last_90['Low'],
                    close=daily_last_90['Close'],
                    name="Price",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 1-Hour oscillator
            if not hourly.empty:
                hourly_last_200 = hourly.tail(200)
                fig.add_trace(
                    go.Scatter(
                        x=hourly_last_200.index,
                        y=hourly_last_200['oscillator'],
                        name="Oscillator",
                        line=dict(color='#2196F3', width=2)
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=hourly_last_200.index,
                        y=hourly_last_200['signal'],
                        name="Signal",
                        line=dict(color='#FF5722', width=1.5, dash='dash')
                    ),
                    row=2, col=1
                )
            
            # Daily oscillator
            if not daily.empty:
                daily_last_200 = daily.tail(200)
                fig.add_trace(
                    go.Scatter(
                        x=daily_last_200.index,
                        y=daily_last_200['oscillator'],
                        name="Oscillator",
                        line=dict(color='#2196F3', width=2)
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=daily_last_200.index,
                        y=daily_last_200['signal'],
                        name="Signal",
                        line=dict(color='#FF5722', width=1.5, dash='dash')
                    ),
                    row=3, col=1
                )
            
            # Weekly oscillator
            if not weekly.empty:
                weekly_last_100 = weekly.tail(100)
                fig.add_trace(
                    go.Scatter(
                        x=weekly_last_100.index,
                        y=weekly_last_100['oscillator'],
                        name="Oscillator",
                        line=dict(color='#2196F3', width=2)
                    ),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=weekly_last_100.index,
                        y=weekly_last_100['signal'],
                        name="Signal",
                        line=dict(color='#FF5722', width=1.5, dash='dash')
                    ),
                    row=4, col=1
                )
            
            # Add zero lines
            for row in [2, 3, 4]:
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=1)
            
            # Layout updates
            fig.update_layout(
                height=1100,
                showlegend=False,
                xaxis_rangeslider_visible=False,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            fig.update_xaxes(title_text="Date", row=4, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Oscillator", row=2, col=1)
            fig.update_yaxes(title_text="Oscillator", row=3, col=1)
            fig.update_yaxes(title_text="Oscillator", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monte Carlo metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("2D Prob Up", f"{data['mc_2d'].get('p_up', 50):.1f}%")
            with col2:
                st.metric("2D Median Return", f"{data['mc_2d'].get('median', 0)*100:.2f}%")
            with col3:
                st.metric("5D Prob Up", f"{data['mc_5d'].get('p_up', 50):.1f}%")
            with col4:
                st.metric("5D Median Return", f"{data['mc_5d'].get('median', 0)*100:.2f}%")
            
            # Recent signals table
            st.markdown("#### Recent Daily Signals")
            if not daily.empty:
                recent = daily.tail(10)[['oscillator', 'signal', 'rsi']].round(2)
                recent.columns = ['Oscillator', 'Signal', 'RSI']
                st.dataframe(recent, use_container_width=True)

else:
    st.error("❌ No data available for the selected symbols. Please check your API credentials and symbol list.")
