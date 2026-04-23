"""
SWEET SPOT RESEARCH ENGINE
Finds best oscillator combos and thresholds from historical data
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import product

import streamlit as st
import plotly.graph_objects as go

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pytz

NY_TZ = "America/New_York"

st.set_page_config(page_title="Sweet Spot Research Engine", layout="wide")
st.title("🔬 Sweet Spot Research Engine")
st.caption("Discovers best oscillator combos, thresholds, and hold times")

# ============================================
# DATA FETCHING
# ============================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol: str, days: int, key: str, secret: str, feed: str) -> pd.DataFrame:
    """Fetch minute data from Alpaca."""
    client = StockHistoricalDataClient(key, secret)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=50000,
        feed=feed,
        adjustment="all",
    )
    
    try:
        bars = client.get_stock_bars(req).df
        if bars.empty:
            return pd.DataFrame()
        bars = bars.reset_index(level=0, drop=True)
        bars.index = pd.to_datetime(bars.index).tz_convert(NY_TZ).tz_localize(None)
        return bars
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()


# ============================================
# OSCILLATOR CALCULATIONS
# ============================================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def tsi(series, long_period=25, short_period=13, signal_period=7):
    delta = series.diff()
    abs_delta = delta.abs()
    double_smoothed = ema(ema(delta, long_period), short_period)
    double_abs = ema(ema(abs_delta, long_period), short_period)
    tsi_line = 100 * double_smoothed / double_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal_period)
    return tsi_line, signal_line

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period, min_periods=2).mean()
    md = (tp - ma).abs().rolling(period, min_periods=2).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def roc(series, period=10):
    return (series / series.shift(period) - 1) * 100

def stoch(series, period=14):
    low = series.rolling(period, min_periods=2).min()
    high = series.rolling(period, min_periods=2).max()
    return 100 * (series - low) / (high - low).replace(0, np.nan)


# ============================================
# RESAMPLE TO TIMEFRAME
# ============================================
def resample_to_timeframe(df, minutes):
    """Resample minute data to desired timeframe."""
    freq = f'{minutes}min'
    return df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


# ============================================
# SWEET SPOT SEARCH
# ============================================
def find_sweet_spots(df, timeframes, hold_bars_list, threshold_ranges):
    """Grid search for best oscillator combos."""
    results = []
    total_combos = len(timeframes) * len(hold_bars_list) * len(threshold_ranges) * 4
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    combo_count = 0
    
    for tf_minutes in timeframes:
        # Resample data
        tf_df = resample_to_timeframe(df, tf_minutes)
        if len(tf_df) < 100:
            continue
        
        # Calculate oscillators
        tf_df['tsi'], _ = tsi(tf_df['close'], 25, 13, 7)
        tf_df['cci'] = cci(tf_df, 20)
        tf_df['roc'] = roc(tf_df['close'], 10)
        tf_df['rsi'] = rsi(tf_df['close'], 14)
        tf_df['stoch'] = stoch(tf_df['close'], 14)
        
        # Forward returns
        for hold in hold_bars_list:
            tf_df[f'fwd_{hold}'] = (tf_df['close'].shift(-hold) / tf_df['close'] - 1) * 100
        
        for thresholds in threshold_ranges:
            tsi_th, cci_th, roc_th, rsi_th = thresholds
            
            # Generate signal: ALL oscillators must be above threshold
            # (for cross above zero, threshold = 0)
            signal = (
                (tf_df['tsi'] > tsi_th) & 
                (tf_df['cci'] > cci_th) & 
                (tf_df['roc'] > roc_th) & 
                (tf_df['rsi'] > rsi_th)
            )
            
            # Also capture the CROSS event (not just level)
            cross_signal = (
                (tf_df['tsi'].shift(1) <= tsi_th) & (tf_df['tsi'] > tsi_th) &
                (tf_df['cci'].shift(1) <= cci_th) & (tf_df['cci'] > cci_th) &
                (tf_df['roc'].shift(1) <= roc_th) & (tf_df['roc'] > roc_th) &
                (tf_df['rsi'].shift(1) <= rsi_th) & (tf_df['rsi'] > rsi_th)
            )
            
            for hold in hold_bars_list:
                for sig_name, sig_mask in [("Level", signal), ("Cross", cross_signal)]:
                    signals = tf_df[sig_mask].copy()
                    returns = signals[f'fwd_{hold}'].dropna()
                    
                    if len(returns) < 20:
                        continue
                    
                    win_rate = (returns > 0).mean() * 100
                    avg_return = returns.mean()
                    sharpe = avg_return / returns.std() if returns.std() > 0 else 0
                    t_stat = avg_return / (returns.std() / np.sqrt(len(returns))) if returns.std() > 0 else 0
                    
                    combo_count += 1
                    status.text(f"Testing combo {combo_count}...")
                    progress_bar.progress(min(combo_count / total_combos, 1.0))
                    
                    results.append({
                        'timeframe_min': tf_minutes,
                        'hold_bars': hold,
                        'hold_minutes': tf_minutes * hold,
                        'tsi_th': tsi_th,
                        'cci_th': cci_th,
                        'roc_th': roc_th,
                        'rsi_th': rsi_th,
                        'signal_type': sig_name,
                        'trades': len(returns),
                        'win_rate': round(win_rate, 1),
                        'avg_return': round(avg_return, 2),
                        'sharpe': round(sharpe, 2),
                        't_stat': round(t_stat, 2),
                        'p5': round(np.percentile(returns, 5), 2),
                        'p95': round(np.percentile(returns, 95), 2),
                    })
    
    progress_bar.empty()
    status.empty()
    
    return pd.DataFrame(results)


# ============================================
# MAIN APP
# ============================================
with st.sidebar:
    st.header("🔑 Alpaca API")
    api_key = st.text_input("API Key", type="password", value=os.getenv("ALPACA_API_KEY", ""))
    secret_key = st.text_input("Secret Key", type="password", value=os.getenv("ALPACA_SECRET_KEY", ""))
    feed = st.selectbox("Feed", ["sip", "iex"], index=0)
    
    st.header("📊 Research Settings")
    symbol = st.text_input("Symbol", value="SOXS").upper().strip()
    days_back = st.slider("Days of historical data", 60, 180, 90)
    
    st.header("🔬 Search Parameters")
    timeframes = st.multiselect("Timeframes (minutes)", [5, 10, 15, 30], default=[10])
    hold_bars = st.multiselect("Hold bars", [1, 2, 3, 5], default=[2, 3])
    
    st.header("🎯 Thresholds to Test")
    tsi_thresholds = st.multiselect("TSI thresholds", [-10, -5, 0, 5, 10], default=[0])
    cci_thresholds = st.multiselect("CCI thresholds", [-100, -50, 0, 50, 100], default=[0])
    roc_thresholds = st.multiselect("ROC thresholds", [-5, -2, 0, 2, 5], default=[0])
    rsi_thresholds = st.multiselect("RSI thresholds", [30, 40, 50, 60, 70], default=[50])
    
    st.divider()
    run_button = st.button("🚀 FIND SWEET SPOTS", type="primary", use_container_width=True)


if not run_button:
    st.info("👈 Set parameters and click FIND SWEET SPOTS")
    st.stop()

if not api_key or not secret_key:
    st.error("Enter Alpaca API credentials")
    st.stop()

# Fetch data
with st.spinner(f"Fetching {days_back} days of {symbol} data..."):
    minute_df = fetch_data(symbol, days_back, api_key, secret_key, feed)

if minute_df.empty:
    st.error(f"No data for {symbol}")
    st.stop()

st.success(f"✅ Loaded {len(minute_df)} minute bars from {minute_df.index[0]} to {minute_df.index[-1]}")

# Generate threshold combinations
threshold_combos = list(product(tsi_thresholds, cci_thresholds, roc_thresholds, rsi_thresholds))

# Find sweet spots
st.subheader("🔬 Searching for Sweet Spots...")
results_df = find_sweet_spots(minute_df, timeframes, hold_bars, threshold_combos)

if results_df.empty:
    st.warning("No statistically significant combinations found. Try wider thresholds or more data.")
    st.stop()

# Display results
st.subheader("🏆 Top 20 Sweet Spots (Sorted by Win Rate)")

display_cols = ['timeframe_min', 'hold_minutes', 'signal_type', 'tsi_th', 'cci_th', 'roc_th', 'rsi_th', 
                'trades', 'win_rate', 'avg_return', 'sharpe', 't_stat', 'p5', 'p95']

results_df = results_df.sort_values('win_rate', ascending=False)

st.dataframe(
    results_df[display_cols].head(20).style.format({
        'win_rate': '{:.1f}%',
        'avg_return': '{:.2f}%',
        'sharpe': '{:.2f}',
        't_stat': '{:.2f}',
        'p5': '{:.2f}%',
        'p95': '{:.2f}%',
    }),
    use_container_width=True,
    hide_index=True,
)

# Best overall
best = results_df.iloc[0]
st.subheader("🎯 OPTIMAL STRATEGY")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Timeframe", f"{best['timeframe_min']} min")
col2.metric("Hold Time", f"{best['hold_minutes']} min ({best['hold_bars']} bars)")
col3.metric("Win Rate", f"{best['win_rate']:.1f}%")
col4.metric("Avg Return", f"{best['avg_return']:.2f}%")

st.markdown(f"""
### Entry Rules (All must be true):

| Oscillator | Threshold | Current Value (if known) |
|------------|-----------|--------------------------|
| **TSI({best['tsi_th']})** | > {best['tsi_th']} | — |
| **CCI({best['cci_th']})** | > {best['cci_th']} | — |
| **ROC({best['roc_th']})** | > {best['roc_th']} | — |
| **RSI({best['rsi_th']})** | > {best['rsi_th']} | — |

### Exit Rules:
- Exit after **{best['hold_minutes']} minutes**
- Or exit earlier if Stoch(14) crosses below 80

### Expected Performance:
- Win Rate: **{best['win_rate']}%**
- Average Return: **{best['avg_return']}%**
- Sharpe Ratio: **{best['sharpe']}**
- 5th-95th Percentile Range: [{best['p5']}%, {best['p95']}%]
- Based on **{best['trades']}** historical trades
""")

# Chart of top strategies
st.subheader("📊 Win Rate vs Sharpe Ratio")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results_df.head(50)['win_rate'],
    y=results_df.head(50)['sharpe'],
    mode='markers',
    marker=dict(
        size=results_df.head(50)['trades'] / 10,
        color=results_df.head(50)['timeframe_min'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Timeframe (min)")
    ),
    text=[f"{r['timeframe_min']}min | WR: {r['win_rate']}% | Sharpe: {r['sharpe']}" for _, r in results_df.head(50).iterrows()],
    hoverinfo='text'
))
fig.update_layout(
    height=500,
    xaxis_title="Win Rate (%)",
    yaxis_title="Sharpe Ratio",
    title="Top 50 Strategies: Win Rate vs Sharpe"
)
st.plotly_chart(fig, use_container_width=True)

# Download
csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Full Results CSV", csv, f"{symbol}_sweet_spots.csv", "text/csv")
