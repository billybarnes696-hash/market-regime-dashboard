import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
TRADING_DAYS = 252
MIN_HISTORY_DAYS = 252  # Minimum 1 year of data required

# ==========
# ERROR HANDLING UTILS
# ==========
class StrategyError(Exception):
    """Custom error for strategy validation failures"""
    pass

def validate_data(close, required_tickers):
    """Validate downloaded data meets requirements"""
    errors = []
    warnings_list = []
    
    if close.empty:
        errors.append("❌ No data downloaded. Check internet connection.")
        return False, errors, warnings_list
    
    missing = [t for t in required_tickers if t not in close.columns]
    if missing:
        errors.append(f"❌ Missing required tickers: {', '.join(missing)}")
    
    if len(close) < MIN_HISTORY_DAYS:
        errors.append(f"⚠️ Insufficient history: {len(close)} days (need {MIN_HISTORY_DAYS}+)")
    
    nan_counts = close.isna().sum()
    high_nan = nan_counts[nan_counts > len(close) * 0.3]
    if not high_nan.empty:
        warnings_list.append(f"⚠️ High NaN count in: {', '.join(high_nan.index)}")
    
    return len(errors) == 0, errors, warnings_list

def validate_signals(composite, confidence, pos_df):
    """Validate signals are actually generating trades"""
    errors = []
    warnings_list = []
    
    if composite.empty:
        errors.append("❌ Composite signal is empty")
        return False, errors, warnings_list
    
    if confidence.empty:
        errors.append("❌ Confidence signal is empty")
        return False, errors, warnings_list
    
    # Check for stuck positions
    if pos_df is not None and not pos_df.empty:
        unique_positions = pos_df['position'].nunique()
        if unique_positions == 1:
            warnings_list.append(f"⚠️ Position stuck at {pos_df['position'].iloc[0]} - no rotation occurring")
        
        # Check if we ever go defensive
        defensive_days = (pos_df['position'] <= 0).sum()
        total_days = len(pos_df)
        if defensive_days == 0 and total_days > 100:
            warnings_list.append("⚠️ Never went defensive (SHY/SH) - may miss drawdown protection")
        
        # Check trade frequency
        trades = (pos_df['position'].diff().abs() > 0).sum()
        if trades == 0:
            errors.append("❌ No trades executed - check thresholds")
        elif trades > len(pos_df) * 0.1:
            warnings_list.append(f"⚠️ High turnover: {trades} trades ({trades/len(pos_df)*100:.1f}% of days)")
    
    # Check confidence range
    conf_min, conf_max = confidence.min(), confidence.max()
    if conf_max < 30:
        errors.append(f"❌ Confidence too low (max: {conf_max:.1f}%) - signals won't trigger")
    elif conf_min == conf_max:
        errors.append(f"❌ Confidence stuck at {conf_min:.1f}% - calculation issue")
    
    return len(errors) == 0, errors, warnings_list

def validate_backtest(equity, bh_equity):
    """Validate backtest results are reasonable"""
    errors = []
    warnings_list = []
    
    if equity.empty:
        errors.append("❌ Strategy equity curve is empty")
        return False, errors, warnings_list
    
    # Check for NaN/Inf in equity
    if equity.isna().any() or np.isinf(equity).any():
        errors.append("❌ Equity curve contains NaN or Inf values")
    
    # Check for extreme drawdowns
    dd = (equity / equity.cummax() - 1).min()
    if dd < -0.8:
        warnings_list.append(f"⚠️ Extreme drawdown: {dd*100:.1f}%")
    
    # Check if strategy significantly underperforms
    if len(equity) > 100 and len(bh_equity) > 100:
        strat_ret = equity.iloc[-1] / equity.iloc[0] - 1
        bh_ret = bh_equity.iloc[-1] / bh_equity[0] - 1
        if strat_ret < bh_ret * 0.5:
            warnings_list.append(f"⚠️ Strategy return ({strat_ret*100:.1f}%) is less than half of Buy&Hold ({bh_ret*100:.1f}%)")
    
    return len(errors) == 0, errors, warnings_list

# ==========
# DATA FETCHING
# ==========
@st.cache_data(ttl=3600)
def fetch_adjclose(tickers, years=5):
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(years * 365.25) + 30)
    
    try:
        data = yf.download(
            tickers=tickers, 
            start=start, 
            end=end + pd.Timedelta(days=1),
            auto_adjust=True, 
            progress=False, 
            group_by="ticker", 
            threads=True
        )
    except Exception as e:
        st.error(f"🚨 Data download failed: {str(e)}")
        return pd.DataFrame()
    
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        st.error("🚨 No data returned from Yahoo Finance")
        return pd.DataFrame()
    
    try:
        if isinstance(tickers, str) or len(tickers) == 1:
            adj = data["Close"].to_frame(tickers if isinstance(tickers, str) else tickers[0])
        else:
            close_cols = {}
            for t in tickers:
                try:
                    if t in data.columns.get_level_values(0) and "Close" in data[t].columns:
                        close_cols[t] = data[t]["Close"]
                except: 
                    continue
            adj = pd.DataFrame(close_cols)
        
        if adj.empty:
            st.error("🚨 No valid price columns found")
            return pd.DataFrame()
        
        return adj.dropna(how="all").ffill().dropna()
    except Exception as e:
        st.error(f"🚨 Data processing failed: {str(e)}")
        return pd.DataFrame()

# ==========
# SIGNAL GENERATION
# ==========
def build_simple_composite(close):
    """Simplified composite based on key ratios"""
    try:
        ratios = [
            ("HYG", "SHY"), ("SMH", "SPY"), ("SPY", "VXX"), 
            ("XLF", "SPY"), ("IWM", "SPY")
        ]
        
        signals = []
        for num, den in ratios:
            if num not in close.columns or den not in close.columns: 
                continue
            ratio = (close[num] / close[den]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) < 50: 
                continue
            
            # Simple momentum: price above 50-day MA = bullish
            ma50 = ratio.rolling(50).mean()
            sig = pd.Series(0, index=ratio.index)
            sig[ratio > ma50] = 1
            sig[ratio < ma50] = -1
            signals.append(sig.ffill())
        
        if not signals: 
            return pd.Series(0, index=close.index)
        
        composite = pd.concat(signals, axis=1).mean(axis=1).ffill()
        
        # Validate composite
        if composite.isna().all():
            st.warning("⚠️ Composite signal is all NaN")
            return pd.Series(0, index=close.index)
        
        return composite
    except Exception as e:
        st.error(f"🚨 Composite calculation failed: {str(e)}")
        return pd.Series(0, index=close.index if 'close' in locals() else pd.DatetimeIndex([]))

def build_positions_3state(composite, spy_close, conf_thr=30, use_sma=True):
    """3-State: -1=SH, 0=SHY, 1=SPY"""
    try:
        df = pd.DataFrame({
            "comp": composite, 
            "spy": spy_close.reindex(composite.index).ffill()
        }).dropna()
        
        if df.empty: 
            st.error("🚨 No valid data for position building")
            return pd.DataFrame()
        
        if use_sma:
            df["sma_200"] = df["spy"].rolling(200).mean()
            df["bull"] = (df["spy"] > df["sma_200"]).astype(int)
        else:
            df["bull"] = 1
        
        # Clear 3-state logic
        def get_position(row):
            if row["bull"] == 1 and row["comp"] > 0.1:
                return 1  # SPY
            elif row["bull"] == 0 and row["comp"] < -0.1:
                return -1  # SH (Short)
            else:
                return 0  # SHY (Neutral)
        
        df["position"] = df.apply(get_position, axis=1)
        
        # Add stickiness to reduce whipsaw
        for i in range(1, len(df)):
            if df["position"].iloc[i] != df["position"].iloc[i-1]:
                if abs(df["comp"].iloc[i]) < 0.15:
                    df["position"].iloc[i] = df["position"].iloc[i-1]
        
        df["asset"] = df["position"].map({1: "SPY", 0: "SHY", -1: "SH"})
        return df
    except Exception as e:
        st.error(f"🚨 Position building failed: {str(e)}")
        return pd.DataFrame()

def backtest_3state(spy, shy, sh, pos_df, cost_bps=5.0):
    """Proper backtest with lagged positions"""
    try:
        idx = spy.index.intersection(pos_df.index)
        df = pd.DataFrame({
            "SPY": spy.loc[idx],
            "SHY": shy.loc[idx],
            "SH": sh.loc[idx],
            "pos": pos_df["position"].loc[idx]
        }).dropna()
        
        if df.empty: 
            return pd.Series(), 0
        
        rets = pd.DataFrame({
            "SPY": df["SPY"].pct_change().fillna(0),
            "SHY": df["SHY"].pct_change().fillna(0),
            "SH": df["SH"].pct_change().fillna(0)
        })
        
        strat_ret = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            prev_pos = df["pos"].iloc[i-1] if i > 0 else 1
            asset = {1: "SPY", 0: "SHY", -1: "SH"}.get(prev_pos, "SPY")
            strat_ret.iloc[i] = rets[asset].iloc[i]
        
        turnover = (df["pos"].diff().abs().fillna(0) > 0).astype(int)
        strat_ret -= turnover * (cost_bps / 10000.0)
        
        equity = (1.0 + strat_ret).cumprod()
        
        # Validate equity
        if equity.isna().any() or np.isinf(equity).any():
            st.warning("⚠️ Equity curve contains invalid values")
            equity = equity.replace([np.inf, -np.inf], np.nan).ffill()
        
        return equity, turnover.sum()
    except Exception as e:
        st.error(f"🚨 Backtest failed: {str(e)}")
        return pd.Series(), 0

def perf_stats(equity):
    try:
        eq = equity.dropna()
        if len(eq) < 10: 
            return {"Return": np.nan, "CAGR": np.nan, "DD": np.nan, "Sharpe": np.nan}
        rets = eq.pct_change().dropna()
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1 if years > 0 else np.nan
        dd = (eq / eq.cummax() - 1).min()
        sharpe = (rets.mean()/rets.std())*np.sqrt(252) if rets.std() > 0 else np.nan
        return {
            "Return": (eq.iloc[-1]/eq.iloc[0])-1,
            "CAGR": cagr, 
            "DD": float(dd),
            "Sharpe": float(sharpe)
        }
    except Exception as e:
        st.error(f"🚨 Performance stats calculation failed: {str(e)}")
        return {"Return": np.nan, "CAGR": np.nan, "DD": np.nan, "Sharpe": np.nan}

# ==========
# UI
# ==========
st.set_page_config(page_title="3-State Rotation: SPY/SHY/SH", layout="wide")
st.title("🎯 3-State Rotation: SPY → SHY → SH")
st.caption("Long SPY (Bull) | Cash/SHY (Neutral) | Short SH (Bear) | With Error Checking")

st.sidebar.header("Controls")
years = st.sidebar.slider("History (years)", 3, 10, 5)
use_sma = st.sidebar.checkbox("Use 200-SMA Filter (Required for Shorting)", value=True)
cost_bps = st.sidebar.slider("Trading Cost (bps)", 0, 50, 5)
show_debug = st.sidebar.checkbox("Show Debug Info", value=True)

TICKERS = ["SPY", "SHY", "SH", "HYG", "VXX", "SMH", "XLF", "IWM"]
REQUIRED = ["SPY", "SHY", "SH"]

# Data Fetching
with st.spinner("📥 Fetching data..."):
    close = fetch_adjclose(TICKERS, years=years)

# Data Validation
data_ok, data_errors, data_warnings = validate_data(close, REQUIRED)

if data_warnings:
    for w in data_warnings:
        st.warning(w)

if not data_ok:
    for e in data_errors:
        st.error(e)
    st.stop()

# Signal Generation
with st.spinner("🔧 Building signals..."):
    composite = build_simple_composite(close)
    pos_df = build_positions_3state(composite, close["SPY"], use_sma=use_sma)

if pos_df.empty:
    st.error("🚨 No positions generated. Check signal parameters.")
    st.stop()

# Signal Validation
signal_ok, signal_errors, signal_warnings = validate_signals(composite, pd.Series(), pos_df)

if signal_warnings:
    for w in signal_warnings:
        st.warning(w)

# Backtest
with st.spinner("📊 Running backtest..."):
    eq_strat, trades = backtest_3state(close["SPY"], close["SHY"], close["SH"], pos_df, cost_bps)
    bh = (1 + close["SPY"].pct_change().fillna(0)).cumprod()

# Backtest Validation
backtest_ok, backtest_errors, backtest_warnings = validate_backtest(eq_strat, bh)

if backtest_warnings:
    for w in backtest_warnings:
        st.warning(w)

if not backtest_ok:
    for e in backtest_errors:
        st.error(e)
    st.stop()

# Stats
stats = {
    "Strategy: 3-State": perf_stats(eq_strat),
    "Buy&Hold: SPY": perf_stats(bh)
}

st.subheader("📊 Performance Comparison")
st.dataframe(pd.DataFrame([
    {"Strategy": k, "Return %": f"{v['Return']*100:.1f}" if not np.isnan(v['Return']) else "N/A", 
     "CAGR %": f"{v['CAGR']*100:.2f}" if not np.isnan(v['CAGR']) else "N/A", 
     "Max DD %": f"{v['DD']*100:.1f}" if not np.isnan(v['DD']) else "N/A", 
     "Sharpe": f"{v['Sharpe']:.2f}" if not np.isnan(v['Sharpe']) else "N/A"}
    for k, v in stats.items()
]), use_container_width=True)

# Chart
st.subheader("📈 Equity Curves ($10k Start)")
try:
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(eq_strat.index, eq_strat*10000, label="3-State Strategy", linewidth=2)
    ax.plot(bh.index, bh*10000, label="Buy&Hold SPY", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
except Exception as e:
    st.error(f"🚨 Chart rendering failed: {str(e)}")

# Diagnostic: Did it actually trade?
st.subheader("🔍 Did It Actually Rotate? (Last 50 Days)")
try:
    diag = pos_df.tail(50)[["comp", "bull", "position", "asset"]].copy()
    st.dataframe(diag, use_container_width=True)
except Exception as e:
    st.error(f"🚨 Diagnostic table failed: {str(e)}")

# Asset allocation
if len(pos_df) > 0:
    st.subheader("📊 Time in Each Asset")
    try:
        alloc = pos_df["asset"].value_counts()
        st.bar_chart(alloc)
    except Exception as e:
        st.error(f"🚨 Asset allocation chart failed: {str(e)}")

# Debug info
if show_debug:
    with st.expander("🔍 Debug & Error Checking Info"):
        st.write(f"**Data Points:** {len(close)}")
        st.write(f"**Position Changes:** {trades}")
        st.write(f"**Unique Assets Used:** {pos_df['asset'].unique().tolist()}")
        st.write(f"**Bull Market %:** {(pos_df['bull']==1).mean()*100:.1f}%")
        st.write(f"**Short Signals:** {(pos_df['position']==-1).sum()} days")
        st.write(f"**Neutral Days:** {(pos_df['position']==0).sum()} days")
        st.write(f"**Long Days:** {(pos_df['position']==1).sum()} days")
        
        # Confidence check
        st.write(f"**Composite Range:** {pos_df['comp'].min():.2f} to {pos_df['comp'].max():.2f}")
        
        # Validation summary
        st.write("---")
        st.write("**Validation Summary:**")
        st.write(f"✅ Data Valid: {data_ok}")
        st.write(f"✅ Signals Valid: {signal_ok}")
        st.write(f"✅ Backtest Valid: {backtest_ok}")
        
        # Recommendations
        st.write("---")
        st.write("**Recommendations:**")
        if (pos_df['position']==1).mean() > 0.95:
            st.warning("⚠️ Strategy is long >95% of time - similar to Buy&Hold")
        if (pos_df['position']==-1).sum() == 0:
            st.warning("⚠️ Never shorted - SPY may have stayed above 200-SMA entire period")
        if trades < 5:
            st.warning("⚠️ Very few trades - may be too conservative")
        if trades > len(pos_df) * 0.1:
            st.warning("⚠️ High turnover - consider increasing thresholds")

# Final reality check
st.subheader("⚠️ Reality Check")
strat_ret = stats["Strategy: 3-State"]['Return']
bh_ret = stats["Buy&Hold: SPY"]['Return']

if not np.isnan(strat_ret) and not np.isnan(bh_ret):
    if strat_ret < bh_ret * 0.7:
        st.error(f"🚨 Strategy significantly underperforming ({strat_ret*100:.1f}% vs {bh_ret*100:.1f}%)")
        st.info("💡 **Try:** Lower confidence threshold, disable 200-SMA filter, or accept that Buy&Hold wins in strong bull markets")
    elif strat_ret > bh_ret:
        st.success(f"✅ Strategy outperforming Buy&Hold! ({strat_ret*100:.1f}% vs {bh_ret*100:.1f}%)")
    else:
        st.info(f"⚖️ Strategy near Buy&Hold ({strat_ret*100:.1f}% vs {bh_ret*100:.1f}%) - check if drawdown is lower")
