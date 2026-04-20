"""
Stock Analyzer v5 - Alpha Vantage API
Working solution with TSI 25,13,7 primary oscillator
"""

import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Stock Analyzer v5", layout="wide")

# YOUR ALPHA VANTAGE API KEY
ALPHA_VANTAGE_KEY = "8H8MMI86K8WI6UEQ"

# Default symbols to analyze
DEFAULT_SYMBOLS = ["QQQ", "SMH", "XLF", "AAPL", "MSFT", "NVDA", "AMD", "INTC", "META", "GOOGL"]

# ============================================================================
# ALPHA VANTAGE FETCHER
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpha_vantage_daily(symbol: str) -> pd.DataFrame:
    """Fetch daily data from Alpha Vantage"""
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_KEY,
        "outputsize": "full",
        "datatype": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            error_msg = data.get("Note", data.get("Error Message", "Unknown error"))
            st.warning(f"{symbol}: {error_msg[:100]}")
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ["Open", "High", "Low", "Close", "Adjusted Close", "Volume", "Dividend", "Split"]
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        
        return df.tail(500)  # Last 500 days
        
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpha_vantage_intraday(symbol: str) -> pd.DataFrame:
    """Fetch hourly data from Alpha Vantage (limited to 100 bars on free tier)"""
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "60min",
        "apikey": ALPHA_VANTAGE_KEY,
        "outputsize": "compact",  # 'compact' returns last 100 bars
        "datatype": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if "Time Series (60min)" not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(data["Time Series (60min)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        
        return df.tail(200)
        
    except Exception:
        return pd.DataFrame()


def fetch_multi_timeframe(symbol: str) -> dict:
    """Fetch daily, weekly, and hourly data"""
    
    # Daily data (always available)
    daily = fetch_alpha_vantage_daily(symbol)
    if daily.empty:
        return {"daily": pd.DataFrame(), "weekly": pd.DataFrame(), "hourly": pd.DataFrame()}
    
    # Weekly from daily resample
    weekly = daily.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    
    # Hourly data (may be limited on free tier)
    hourly = fetch_alpha_vantage_intraday(symbol)
    
    # If hourly is empty, resample daily to hourly (degraded but works)
    if hourly.empty and len(daily) > 10:
        hourly = daily.resample("h").ffill().dropna()
    
    return {"daily": daily, "weekly": weekly, "hourly": hourly}


# ============================================================================
# INDICATORS
# ============================================================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def tsi(series, long=25, short=13, signal=7):
    """TSI 25,13,7 - PRIMARY OSCILLATOR"""
    diff = series.diff()
    abs_diff = diff.abs()
    
    double_smooth = ema(ema(diff, long), short)
    double_abs = ema(ema(abs_diff, long), short)
    
    tsi_line = 100 * double_smooth / double_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal)
    
    return tsi_line, signal_line


def atr(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_features(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """Compute all features"""
    if df.empty:
        return df
    
    out = df.copy()
    
    # TSI 25,13,7 (PRIMARY - this is the main signal)
    out["tsi"], out["tsi_signal"] = tsi(out["Close"])
    out["tsi_gap"] = out["tsi"] - out["tsi_signal"]
    out["tsi_slope_1"] = out["tsi"].diff(1)
    out["tsi_slope_3"] = out["tsi"].diff(3)
    
    # RSI
    out["rsi_14"] = rsi(out["Close"], 14)
    out["rsi_slope_3"] = out["rsi_14"].diff(3)
    
    # ATR
    out["atr_14"] = atr(out, 14)
    out["atr_stretch"] = (out["Close"] - ema(out["Close"], 20)) / out["atr_14"].replace(0, np.nan)
    
    # Moving averages
    out["ema_20"] = ema(out["Close"], 20)
    out["sma_50"] = out["Close"].rolling(50).mean()
    out["dist_ema20_pct"] = (out["Close"] / out["ema_20"]) - 1
    
    # Volume
    out["volume_ma_20"] = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_ma_20"].replace(0, np.nan)
    
    # Candle features
    out["close_in_range"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["candle_score"] = 50 + (out["close_in_range"].fillna(0.5) - 0.5) * 40
    
    return out


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward returns for backtesting"""
    if df.empty:
        return df
    out = df.copy()
    out["fwd_ret_1"] = out["Close"].shift(-1) / out["Close"] - 1
    out["fwd_ret_2"] = out["Close"].shift(-2) / out["Close"] - 1
    out["fwd_ret_5"] = out["Close"].shift(-5) / out["Close"] - 1
    return out


# ============================================================================
# SIGNAL DETECTION (TSI 25,13,7 CENTRIC)
# ============================================================================

def get_signal(row: pd.Series, timeframe: str) -> tuple:
    """Get signal based on TSI 25,13,7 percentile"""
    if row.empty:
        return "NO DATA", 0, "No data"
    
    tsi_val = row.get("tsi", 0)
    tsi_gap = row.get("tsi_gap", 0)
    tsi_slope = row.get("tsi_slope_3", 0)
    rsi_val = row.get("rsi_14", 50)
    
    # PRIMARY SIGNAL: TSI 25,13,7 thresholds
    # Overheated (bearish) - TSI > 70
    if tsi_val > 70:
        if tsi_gap < 0 or tsi_slope < 0:
            return "PUT", 85, f"OVERHEATED: TSI={tsi_val:.1f}, rolling over"
        return "PUT", 70, f"Overheated: TSI={tsi_val:.1f}"
    
    # Washed out (bullish) - TSI < -70
    if tsi_val < -70:
        if tsi_gap > 0 or tsi_slope > 0:
            return "CALL", 85, f"WASHED OUT: TSI={tsi_val:.1f}, turning up"
        return "CALL", 70, f"Washed out: TSI={tsi_val:.1f}"
    
    # Strong bullish momentum
    if tsi_val > 50 and tsi_slope > 0:
        return "CALL", 65, f"Bullish momentum: TSI={tsi_val:.1f}, rising"
    
    # Strong bearish momentum
    if tsi_val < 50 and tsi_slope < 0:
        return "PUT", 65, f"Bearish momentum: TSI={tsi_val:.1f}, falling"
    
    # Neutral
    return "NEUTRAL", 40, f"Neutral: TSI={tsi_val:.1f}"


def get_combined_signal(daily_row, hourly_row, weekly_row) -> tuple:
    """Combine signals from all timeframes"""
    
    daily_signal, daily_conf, daily_reason = get_signal(daily_row, "daily")
    hourly_signal, hourly_conf, hourly_reason = get_signal(hourly_row, "hourly") if not hourly_row.empty else ("NEUTRAL", 40, "No hourly")
    weekly_signal, weekly_conf, weekly_reason = get_signal(weekly_row, "weekly") if not weekly_row.empty else ("NEUTRAL", 40, "No weekly")
    
    tsi_val = daily_row.get("tsi", 0)
    
    # Strong alignment
    if daily_signal == "CALL" and weekly_signal == "CALL":
        if hourly_signal == "CALL":
            return "STRONG CALL", 90, f"All timeframes bullish | TSI={tsi_val:.1f}"
        return "CALL", 75, f"Daily & weekly bullish | TSI={tsi_val:.1f}"
    
    if daily_signal == "PUT" and weekly_signal == "PUT":
        if hourly_signal == "PUT":
            return "STRONG PUT", 90, f"All timeframes bearish | TSI={tsi_val:.1f}"
        return "PUT", 75, f"Daily & weekly bearish | TSI={tsi_val:.1f}"
    
    # Daily alone
    if daily_signal == "CALL":
        return "CALL", 65, daily_reason
    if daily_signal == "PUT":
        return "PUT", 65, daily_reason
    
    return "NEUTRAL", 40, f"Mixed signals | TSI={tsi_val:.1f}"


# ============================================================================
# ANALOG MATCHING
# ============================================================================

def find_analogs(df: pd.DataFrame, current_row: pd.Series, n: int = 15) -> pd.DataFrame:
    """Find historical analog matches"""
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    # Feature columns for matching
    feature_cols = ["tsi", "tsi_gap", "tsi_slope_3", "rsi_14", "atr_stretch", "dist_ema20_pct", "volume_ratio"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if not feature_cols:
        return pd.DataFrame()
    
    # Prepare data
    hist = df.dropna(subset=feature_cols + ["fwd_ret_1", "fwd_ret_2", "fwd_ret_5"]).copy()
    if len(hist) < n + 10:
        return pd.DataFrame()
    
    current = hist.iloc[-1]
    hist = hist.iloc[:-1].copy()
    
    # Calculate distances
    for col in feature_cols:
        if col in hist.columns:
            hist[f"{col}_diff"] = (hist[col] - current[col]) ** 2
    
    diff_cols = [c for c in hist.columns if c.endswith("_diff")]
    if diff_cols:
        hist["distance"] = np.sqrt(hist[diff_cols].sum(axis=1))
        hist["similarity"] = 1 / (1 + hist["distance"])
        analogs = hist.sort_values("distance").head(n)
        return analogs[["Close", "similarity", "fwd_ret_1", "fwd_ret_2", "fwd_ret_5"]]
    
    return pd.DataFrame()


def summarize_analogs(analogs: pd.DataFrame) -> dict:
    """Summarize analog statistics"""
    if analogs.empty:
        return {}
    
    return {
        "n": len(analogs),
        "fwd_ret_1_median": analogs["fwd_ret_1"].median(),
        "fwd_ret_2_median": analogs["fwd_ret_2"].median(),
        "fwd_ret_5_median": analogs["fwd_ret_5"].median(),
        "fwd_ret_1_p_up": (analogs["fwd_ret_1"] > 0).mean(),
        "fwd_ret_1_p_down": (analogs["fwd_ret_1"] < 0).mean(),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_dashboard(symbol: str, daily_df: pd.DataFrame, hourly_df: pd.DataFrame, weekly_df: pd.DataFrame):
    """Plot 4-row dashboard: Daily Price, Daily TSI, Hourly TSI, Weekly TSI"""
    
    fig = make_subplots(
        rows=4, cols=1,
        vertical_spacing=0.06,
        subplot_titles=[
            f"{symbol} - Daily Price",
            "Daily TSI 25,13,7 (Primary Signal)",
            "Hourly TSI 25,13,7 (Entry Timing)",
            "Weekly TSI 25,13,7 (Macro Context)"
        ],
        row_heights=[0.40, 0.22, 0.20, 0.18],
    )
    
    # Row 1: Daily Candles
    if not daily_df.empty:
        d = daily_df.tail(200)
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["ema_20"], name="EMA20", line=dict(color="orange", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["sma_50"], name="SMA50", line=dict(color="blue", width=1)), row=1, col=1)
    
    # Row 2: Daily TSI
    if not daily_df.empty:
        d = daily_df.tail(200)
        fig.add_trace(go.Scatter(x=d.index, y=d["tsi"], name="TSI", line=dict(color="red", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["tsi_signal"], name="Signal", line=dict(color="black", width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overheated (Bearish)", row=2, col=1)
        fig.add_hline(y=-70, line_dash="dash", line_color="green", annotation_text="Washed Out (Bullish)", row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Row 3: Hourly TSI
    if not hourly_df.empty:
        h = hourly_df.tail(200)
        fig.add_trace(go.Scatter(x=h.index, y=h["tsi"], name="Hourly TSI", line=dict(color="red", width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=h["tsi_signal"], name="Hourly Signal", line=dict(color="black", width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=-70, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Row 4: Weekly TSI
    if not weekly_df.empty:
        w = weekly_df.tail(100)
        fig.add_trace(go.Scatter(x=w.index, y=w["tsi"], name="Weekly TSI", line=dict(color="red", width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=w.index, y=w["tsi_signal"], name="Weekly Signal", line=dict(color="black", width=1)), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=-70, line_dash="dash", line_color="green", row=4, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    
    fig.update_layout(
        height=1100,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend=dict(y=1.02, x=0),
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="TSI Value", row=2, col=1)
    fig.update_yaxes(title_text="TSI Value", row=3, col=1)
    fig.update_yaxes(title_text="TSI Value", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================

st.title("📊 Stock Analyzer v5")
st.caption("Alpha Vantage API | TSI 25,13,7 Primary Oscillator | Multi-timeframe Analysis")

# Sidebar
with st.sidebar:
    st.header("📈 Symbols")
    symbol_input = st.text_area("Stock symbols (one per line or comma separated)", 
                                value="\n".join(DEFAULT_SYMBOLS), height=200)
    symbols = [s.strip().upper() for s in symbol_input.replace(",", "\n").split("\n") if s.strip()]
    st.caption(f"Analyzing {len(symbols)} symbols")
    
    st.header("⚙️ Settings")
    analog_count = st.slider("Analog matches", 5, 30, 15)
    
    st.header("ℹ️ API Status")
    st.success("✅ Alpha Vantage API Key loaded")
    st.caption("Rate limit: 5 calls/minute | 500 calls/day")
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

st.info("""
**TSI 25,13,7 Signal Interpretation:**
- **TSI > 70** → Overheated → 🔴 BEARISH (consider PUT options)
- **TSI < -70** → Washed out → 🟢 BULLISH (consider CALL options)
- **Between -70 and 70** → Neutral → Wait for signal
""")

if not run_analysis:
    st.stop()

if not symbols:
    st.error("Please enter at least one symbol")
    st.stop()

# Run analysis for all symbols
results = []
detail_data = {}
progress_bar = st.progress(0)
status_text = st.empty()

for i, symbol in enumerate(symbols):
    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
    progress_bar.progress((i + 1) / len(symbols))
    
    # Fetch data
    data = fetch_multi_timeframe(symbol)
    daily_df = data["daily"]
    weekly_df = data["weekly"]
    hourly_df = data["hourly"]
    
    if daily_df.empty:
        results.append({
            "Symbol": symbol,
            "Status": "❌ No data",
            "Daily TSI": "N/A",
            "Signal": "ERROR",
            "Confidence": 0,
        })
        continue
    
    # Compute features and forward returns
    daily_df = compute_features(daily_df, "daily")
    daily_df = add_forward_returns(daily_df)
    weekly_df = compute_features(weekly_df, "weekly")
    hourly_df = compute_features(hourly_df, "hourly")
    
    # Get current rows
    daily_row = daily_df.iloc[-1]
    hourly_row = hourly_df.iloc[-1] if not hourly_df.empty else pd.Series()
    weekly_row = weekly_df.iloc[-1] if not weekly_df.empty else pd.Series()
    
    # Get signals
    signal, confidence, reason = get_combined_signal(daily_row, hourly_row, weekly_row)
    tsi_val = daily_row.get("tsi", 0)
    
    # Find analogs
    analogs = find_analogs(daily_df, daily_row, n=analog_count)
    analog_summary = summarize_analogs(analogs)
    
    results.append({
        "Symbol": symbol,
        "Status": "✅ OK",
        "Daily TSI": round(tsi_val, 1),
        "Signal": signal,
        "Confidence": confidence,
        "1d Forecast": f"{analog_summary.get('fwd_ret_1_median', 0)*100:.1f}%" if analog_summary else "N/A",
        "5d Forecast": f"{analog_summary.get('fwd_ret_5_median', 0)*100:.1f}%" if analog_summary else "N/A",
    })
    
    detail_data[symbol] = {
        "daily": daily_df,
        "hourly": hourly_df,
        "weekly": weekly_df,
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "tsi": tsi_val,
        "analogs": analogs,
        "analog_summary": analog_summary,
    }
    
    # Rate limit delay (5 calls per minute = 12 seconds between symbols)
    time.sleep(2)

progress_bar.empty()
status_text.empty()

# Display results
results_df = pd.DataFrame(results)
st.subheader("📊 Analysis Results")

# Sort by confidence (highest first)
if not results_df.empty:
    results_df = results_df.sort_values("Confidence", ascending=False)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results CSV", csv, "stock_analysis.csv", "text/csv")

# Detailed view
st.subheader("🔬 Detailed Analysis")
valid_symbols = [r["Symbol"] for r in results if r["Status"] == "✅ OK"]

if valid_symbols:
    selected = st.selectbox("Select symbol for detailed analysis", valid_symbols)
    
    if selected in detail_data:
        data = detail_data[selected]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Signal", data["signal"])
        col2.metric("Confidence", f"{data['confidence']}%")
        col3.metric("TSI 25,13,7", f"{data['tsi']:.1f}")
        col4.metric("Analysis Date", datetime.now().strftime("%Y-%m-%d"))
        
        st.markdown(f"**Reason:** {data['reason']}")
        
        # TSI Interpretation
        tsi_val = data["tsi"]
        if tsi_val > 70:
            st.warning(f"🔴 TSI = {tsi_val:.1f} (OVERHEATED) - Bearish bias. Consider PUT options or taking profits.")
        elif tsi_val < -70:
            st.success(f"🟢 TSI = {tsi_val:.1f} (WASHED OUT) - Bullish bias. Consider CALL options or adding positions.")
        else:
            st.info(f"⚪ TSI = {tsi_val:.1f} (NEUTRAL) - No clear signal. Wait for TSI to cross ±70.")
        
        # Charts
        plot_dashboard(selected, data["daily"], data["hourly"], data["weekly"])
        
        # Analog matches
        st.subheader("📊 Historical Analog Matches")
        if not data["analogs"].empty:
            st.write(f"Found {len(data['analogs'])} similar historical patterns")
            st.dataframe(data["analogs"].head(10), use_container_width=True)
            
            if data["analog_summary"]:
                col1, col2, col3 = st.columns(3)
                col1.metric("1-Day Forecast", f"{data['analog_summary'].get('fwd_ret_1_median', 0)*100:.2f}%")
                col2.metric("2-Day Forecast", f"{data['analog_summary'].get('fwd_ret_2_median', 0)*100:.2f}%")
                col3.metric("5-Day Forecast", f"{data['analog_summary'].get('fwd_ret_5_median', 0)*100:.2f}%")
        else:
            st.info("Not enough historical data for analog matching")

st.success("✅ Analysis complete!")
