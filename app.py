#!/usr/bin/env python3
"""
Stable Market Engine v13.3 - Simplified Working Version
✅ Direct 1-hour bars from Alpaca
✅ Consistent oscillator calculations
✅ Clean signal generation
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
    
    return df

# -----------------------------
# DATA FETCHING
# -----------------------------
@st.cache_data(ttl=1800)
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
        
        # Cache
        df.to_parquet(cache_file)
        return df
        
    except Exception as e:
        st.warning(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
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
            
            df.to_parquet(cache_file)
            result[symbol] = df
            
        except Exception as e:
            st.warning(f"Error fetching daily {symbol}: {e}")
            result[symbol] = pd.DataFrame()
    
    return result

# -----------------------------
# SIGNAL GENERATION
# -----------------------------
def get_signal(row) -> Tuple[str, float]:
    """Generate signal based on oscillator"""
    if pd.isna(row.get('oscillator', 0)):
        return "NEUTRAL", 0
    
    osc = row['oscillator']
    gap = row.get('gap', 0)
    
    if osc > 0.3 and gap > 0:
        return "CALL", osc
    elif osc < -0.3 and gap < 0:
        return "PUT", osc
    elif osc > 0:
        return "WEAK CALL", osc
    elif osc < 0:
        return "WEAK PUT", osc
    else:
        return "NEUTRAL", osc

# -----------------------------
# MAIN APP
# -----------------------------
st.title("📈 Market Engine v13.3 - Multi-Timeframe Oscillator")
st.caption("1-Hour, Daily, and Weekly signals from consistent oscillator")

with st.sidebar:
    st.header("Alpaca Credentials")
    api_key = st.text_input("API Key", type="password")
    secret_key = st.text_input("Secret Key", type="password")
    
    st.header("Symbols")
    symbols_input = st.text_area("Enter symbols (comma separated)", "QQQ,SMH,NVDA,XLF")
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    st.header("Settings")
    lookback_months = st.slider("Hourly data lookback (months)", 1, 6, 3)
    lookback_years = st.slider("Daily data lookback (years)", 2, 5, 3)
    
    run_button = st.button("Run Analysis", type="primary")

if not run_button:
    st.info("Enter your Alpaca API credentials and click 'Run Analysis'")
    st.stop()

if not api_key or not secret_key:
    st.error("Please enter your Alpaca API credentials")
    st.stop()

# Fetch data
with st.spinner("Fetching market data..."):
    # Add benchmark
    all_symbols = list(set(symbols + ['SPY']))
    daily_data = fetch_daily_bars(all_symbols, lookback_years, api_key, secret_key)
    
    hourly_data = {}
    for sym in symbols:
        hourly_data[sym] = fetch_hourly_bars(sym, lookback_months, api_key, secret_key)

# Calculate indicators
results = []
for sym in symbols:
    hourly = hourly_data.get(sym, pd.DataFrame())
    daily = daily_data.get(sym, pd.DataFrame())
    benchmark = daily_data.get('SPY', pd.DataFrame())
    
    if hourly.empty or daily.empty:
        st.warning(f"Insufficient data for {sym}")
        continue
    
    # Calculate oscillators
    hourly_osc = calculate_oscillator(hourly)
    daily_osc = calculate_oscillator(daily)
    
    # Weekly from daily
    weekly = daily.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max', 
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    weekly_osc = calculate_oscillator(weekly)
    
    # Get latest signals
    if len(hourly_osc) > 0:
        hourly_signal, hourly_strength = get_signal(hourly_osc.iloc[-1])
    else:
        hourly_signal, hourly_strength = "NO DATA", 0
    
    if len(daily_osc) > 0:
        daily_signal, daily_strength = get_signal(daily_osc.iloc[-1])
        daily_rsi = daily_osc.iloc[-1].get('rsi', 50)
    else:
        daily_signal, daily_strength = "NO DATA", 0
        daily_rsi = 50
    
    if len(weekly_osc) > 0:
        weekly_signal, weekly_strength = get_signal(weekly_osc.iloc[-1])
    else:
        weekly_signal, weekly_strength = "NO DATA", 0
    
    # Determine grade
    if daily_strength > 0.3 and hourly_strength > 0:
        grade = "A+"
        grade_reason = "Strong Bullish"
    elif daily_strength > 0.2:
        grade = "A"
        grade_reason = "Bullish"
    elif daily_strength > 0:
        grade = "B"
        grade_reason = "Moderate Bullish"
    elif daily_strength > -0.2:
        grade = "C"
        grade_reason = "Neutral"
    elif daily_strength > -0.3:
        grade = "D"
        grade_reason = "Moderate Bearish"
    else:
        grade = "F"
        grade_reason = "Bearish"
    
    results.append({
        'Symbol': sym,
        'Grade': grade,
        'Grade Reason': grade_reason,
        '1H Signal': hourly_signal,
        'Daily Signal': daily_signal,
        'Weekly Signal': weekly_signal,
        'RSI (Daily)': round(daily_rsi, 1),
        'Daily Oscillator': round(daily_strength, 2),
        'Status': 'OK'
    })

# Display results
if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Daily Oscillator', ascending=False)
    
    st.subheader("📊 Ranked Results")
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df_results.to_csv(index=False)
    st.download_button("Download CSV", csv, "market_analysis.csv", "text/csv")
    
    # Detail view for selected symbol
    st.subheader("🔍 Detail View")
    selected = st.selectbox("Select symbol for charts", df_results['Symbol'].tolist())
    
    if selected:
        # Get data for selected symbol
        hourly = hourly_data.get(selected, pd.DataFrame())
        daily = daily_data.get(selected, pd.DataFrame())
        
        if not hourly.empty and not daily.empty:
            hourly_osc = calculate_oscillator(hourly)
            daily_osc = calculate_oscillator(daily)
            
            # Create charts
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(f"{selected} - Price", "1-Hour Oscillator", "Daily Oscillator"),
                vertical_spacing=0.1,
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Price chart
            fig.add_trace(
                go.Candlestick(
                    x=daily_osc.index[-60:],
                    open=daily_osc['Open'][-60:],
                    high=daily_osc['High'][-60:],
                    low=daily_osc['Low'][-60:],
                    close=daily_osc['Close'][-60:],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # 1-Hour oscillator
            fig.add_trace(
                go.Scatter(x=hourly_osc.index[-120:], y=hourly_osc['oscillator'][-120:], 
                          name="Oscillator", line=dict(color='blue', width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=hourly_osc.index[-120:], y=hourly_osc['signal'][-120:], 
                          name="Signal", line=dict(color='red', width=1, dash='dash')),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Daily oscillator
            fig.add_trace(
                go.Scatter(x=daily_osc.index[-120:], y=daily_osc['oscillator'][-120:], 
                          name="Oscillator", line=dict(color='blue', width=2)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=daily_osc.index[-120:], y=daily_osc['signal'][-120:], 
                          name="Signal", line=dict(color='red', width=1, dash='dash')),
                row=3, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
            
            fig.update_layout(height=900, xaxis_rangeslider_visible=False)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Oscillator", row=2, col=1)
            fig.update_yaxes(title_text="Oscillator", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent signals
            st.markdown("### Recent Signals (Last 10 periods)")
            recent = pd.DataFrame({
                'Date': daily_osc.index[-10:],
                'Daily Oscillator': daily_osc['oscillator'][-10:].round(3),
                'Daily Signal': daily_osc['signal'][-10:].round(3),
                'RSI': daily_osc['rsi'][-10:].round(1)
            })
            st.dataframe(recent, use_container_width=True, hide_index=True)

else:
    st.error("No data available for the selected symbols")
