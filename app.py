import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
TRADING_DAYS = 252

@st.cache_data(ttl=3600)
def fetch_adjclose(tickers, years=5):
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(years * 365.25) + 30)
    try:
        data = yf.download(tickers=tickers, start=start, end=end + pd.Timedelta(days=1),
                           auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception as e:
        st.error(f"Data download failed: {e}")
        return pd.DataFrame()

    if isinstance(tickers, str) or len(tickers) == 1:
        adj = data["Close"].to_frame(tickers if isinstance(tickers, str) else tickers[0])
    else:
        close_cols = {}
        for t in tickers:
            try:
                if t in data.columns.get_level_values(0) and "Close" in data[t].columns:
                    close_cols[t] = data[t]["Close"]
            except: continue
        adj = pd.DataFrame(close_cols)
    return adj.dropna(how="all").ffill().dropna()

def build_simple_composite(close):
    """Simplified composite based on key ratios"""
    ratios = [
        ("HYG", "SHY"), ("SMH", "SPY"), ("SPY", "VXX"), 
        ("XLF", "SPY"), ("IWM", "SPY")
    ]
    
    signals = []
    for num, den in ratios:
        if num not in close.columns or den not in close.columns: continue
        ratio = (close[num] / close[den]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) < 50: continue
        
        # Simple momentum: price above 50-day MA = bullish
        ma50 = ratio.rolling(50).mean()
        sig = pd.Series(0, index=ratio.index)
        sig[ratio > ma50] = 1
        sig[ratio < ma50] = -1
        signals.append(sig.ffill())
    
    if not signals: return pd.Series(0, index=close.index)
    return pd.concat(signals, axis=1).mean(axis=1).ffill()

def build_positions_3state(composite, spy_close, conf_thr=30, use_sma=True):
    """3-State: -1=SH, 0=SHY, 1=SPY"""
    df = pd.DataFrame({
        "comp": composite, 
        "spy": spy_close.reindex(composite.index).ffill()
    }).dropna()
    
    if df.empty: return pd.DataFrame()
    
    if use_sma:
        df["sma_200"] = df["spy"].rolling(200).mean()
        df["bull"] = (df["spy"] > df["sma_200"]).astype(int)
    else:
        df["bull"] = 1
    
    # Clear 3-state logic (NO confidence bottleneck)
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
            # Only allow change if signal is strong
            if abs(df["comp"].iloc[i]) < 0.15:
                df["position"].iloc[i] = df["position"].iloc[i-1]
    
    df["asset"] = df["position"].map({1: "SPY", 0: "SHY", -1: "SH"})
    return df

def backtest_3state(spy, shy, sh, pos_df, cost_bps=5.0):
    """Proper backtest with lagged positions"""
    idx = spy.index.intersection(pos_df.index)
    df = pd.DataFrame({
        "SPY": spy.loc[idx],
        "SHY": shy.loc[idx],
        "SH": sh.loc[idx],
        "pos": pos_df["position"].loc[idx]
    }).dropna()
    
    if df.empty: return pd.Series(), 0
    
    # Calculate returns
    rets = pd.DataFrame({
        "SPY": df["SPY"].pct_change().fillna(0),
        "SHY": df["SHY"].pct_change().fillna(0),
        "SH": df["SH"].pct_change().fillna(0)
    })
    
    # Strategy returns (use PREVIOUS day's position)
    strat_ret = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        prev_pos = df["pos"].iloc[i-1] if i > 0 else 1
        asset = {1: "SPY", 0: "SHY", -1: "SH"}.get(prev_pos, "SPY")
        strat_ret.iloc[i] = rets[asset].iloc[i]
    
    # Transaction costs
    turnover = (df["pos"].diff().abs().fillna(0) > 0).astype(int)
    strat_ret -= turnover * (cost_bps / 10000.0)
    
    equity = (1.0 + strat_ret).cumprod()
    return equity, turnover.sum()

def perf_stats(equity):
    eq = equity.dropna()
    if len(eq) < 10: return {"Return": np.nan, "CAGR": np.nan, "DD": np.nan, "Sharpe": np.nan}
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

# ========== UI ==========
st.set_page_config(page_title="3-State Rotation: SPY/SHY/SH", layout="wide")
st.title("🎯 3-State Rotation: SPY → SHY → SH")
st.caption("Long SPY (Bull) | Cash/SHY (Neutral) | Short SH (Bear)")

st.sidebar.header("Controls")
years = st.sidebar.slider("History (years)", 3, 10, 5)
use_sma = st.sidebar.checkbox("Use 200-SMA Filter (Required for Shorting)", value=True)
cost_bps = st.sidebar.slider("Trading Cost (bps)", 0, 50, 5)

TICKERS = ["SPY", "SHY", "SH", "HYG", "VXX", "SMH", "XLF", "IWM"]

with st.spinner("📥 Fetching..."):
    close = fetch_adjclose(TICKERS, years=years)

if not all(t in close.columns for t in ["SPY", "SHY", "SH"]):
    st.error("❌ Missing SPY, SHY, or SH")
    st.stop()

with st.spinner("🔧 Building signals..."):
    composite = build_simple_composite(close)
    pos_df = build_positions_3state(composite, close["SPY"], use_sma=use_sma)

if pos_df.empty:
    st.error("❌ No positions generated")
    st.stop()

# Backtest
eq_strat, trades = backtest_3state(close["SPY"], close["SHY"], close["SH"], pos_df, cost_bps)
bh = (1 + close["SPY"].pct_change().fillna(0)).cumprod()

# Stats
stats = {
    "Strategy: 3-State": perf_stats(eq_strat),
    "Buy&Hold: SPY": perf_stats(bh)
}
st.subheader("Performance")
st.dataframe(pd.DataFrame([
    {"Strategy": k, "Return %": f"{v['Return']*100:.1f}", "CAGR %": f"{v['CAGR']*100:.2f}", 
     "Max DD %": f"{v['DD']*100:.1f}", "Sharpe": f"{v['Sharpe']:.2f}"}
    for k, v in stats.items()
]), use_container_width=True)

# Chart
st.subheader("Equity ($10k Start)")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(eq_strat.index, eq_strat*10000, label="3-State Strategy", linewidth=2)
ax.plot(bh.index, bh*10000, label="Buy&Hold SPY", linestyle="--")
ax.set_xlabel("Date"); ax.set_ylabel("Equity ($)"); ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)

# Diagnostic: Did it actually trade?
st.subheader("🔍 Did It Actually Rotate? (Last 50 Days)")
diag = pos_df.tail(50)[["comp", "bull", "position", "asset"]].copy()
st.dataframe(diag, use_container_width=True)

# Asset allocation
if len(pos_df) > 0:
    alloc = pos_df["asset"].value_counts()
    st.subheader("📊 Time in Each Asset")
    st.bar_chart(alloc)

# Debug info
with st.expander("🔍 Debug Info"):
    st.write(f"**Position Changes:** {trades}")
    st.write(f"**Unique Assets Used:** {pos_df['asset'].unique().tolist()}")
    st.write(f"**Bull Market %:** {(pos_df['bull']==1).mean()*100:.1f}%")
    st.write(f"**Short Signals:** {(pos_df['position']==-1).sum()} days")
