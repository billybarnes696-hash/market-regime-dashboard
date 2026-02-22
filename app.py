import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 24, 52, 18
TSI_R, TSI_S, TSI_SIGNAL = 40, 20, 10
STOCH_LEN, STOCH_SMOOTHK, STOCH_SMOOTHD = 14, 3, 3
CCI_LEN = 100
TRADING_DAYS = 252

# ==========
# DATA & UTILS
# ==========
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
            except:
                continue
        adj = pd.DataFrame(close_cols)
    
    return adj.dropna(how="all").ffill().dropna()

def normalize_100(s):
    s = s.dropna()
    return 100.0 * s / float(s.iloc[0]) if not s.empty else s

def close_only_stoch(close, length=14, smoothk=3, smoothd=3):
    lo, hi = close.rolling(length).min(), close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = (100.0 * (close - lo) / denom).rolling(smoothk).mean()
    return k, k.rolling(smoothd).mean()

def close_only_cci(close, length=100):
    sma = close.rolling(length).mean()
    mad = (close - sma).abs().rolling(length).mean()
    return (close - sma) / (0.015 * mad.replace(0, np.nan))

def indicator_pack(close):
    close = close.dropna()
    min_req = max(MACD_SLOW, TSI_R, CCI_LEN) + 20
    if len(close) < min_req:
        return pd.DataFrame(columns=["macdh", "tsi", "stochk", "cci"])
    
    try:
        macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd is None or macd.empty or len(macd.columns) < 2:
            return pd.DataFrame(columns=["macdh", "tsi", "stochk", "cci"])
        macdh_col = [c for c in macd.columns if "MACDh" in c or "macdh" in c.lower()]
        macdh = macd[macdh_col[0]].rename("macdh") if macdh_col else macd.iloc[:, 1].rename("macdh")
        
        tsi = ta.tsi(close, fast=TSI_S, slow=TSI_R, signal=TSI_SIGNAL)
        if tsi is None or tsi.empty:
            return pd.DataFrame(columns=["macdh", "tsi", "stochk", "cci"])
        tsi = tsi.iloc[:, 0].rename("tsi")
        
        stoch_k, _ = close_only_stoch(close, length=STOCH_LEN, smoothk=STOCH_SMOOTHK, smoothd=STOCH_SMOOTHD)
        cci = close_only_cci(close, length=CCI_LEN)
        
        return pd.concat([macdh, tsi, stoch_k.rename("stochk"), cci.rename("cci")], axis=1).dropna()
    except Exception:
        return pd.DataFrame(columns=["macdh", "tsi", "stochk", "cci"])

def score_from_indicators(ind):
    if ind.empty: return pd.Series(dtype=float)
    bull = (ind["macdh"] > 0) & (ind["tsi"] > 0) & (ind["stochk"] > 50) & (ind["cci"] > 0)
    bear = (ind["macdh"] < 0) & (ind["tsi"] < 0) & (ind["stochk"] < 50) & (ind["cci"] < 0)
    sc = pd.Series(0.0, index=ind.index)
    sc[bull], sc[bear] = 1.0, -1.0
    return sc

def make_ratio(close_df, num, den):
    if num not in close_df.columns or den not in close_df.columns:
        return pd.Series(dtype=float)
    return (close_df[num] / close_df[den]).replace([np.inf, -np.inf], np.nan).rename(f"{num}:{den}")

def perf_stats(equity):
    eq = equity.dropna()
    if len(eq) < 5: return {"Total Return": np.nan, "CAGR": np.nan, "Max Drawdown": np.nan, "Sharpe": np.nan}
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    sharpe = (rets.mean() / rets.std()) * np.sqrt(TRADING_DAYS) if rets.std() != 0 else np.nan
    return {
        "Total Return": (eq.iloc[-1] / eq.iloc[0]) - 1.0,
        "CAGR": cagr,
        "Max Drawdown": float(dd.min()),
        "Sharpe": float(sharpe)
    }

# ==========
# SIGNAL ENGINE
# ==========
def build_composite_scores(close):
    ratios = {
        "SPXS:SVOL": {"series": make_ratio(close, "SPXS", "SVOL"), "invert": True,  "weight": 0.22},
        "HYG:SHY":   {"series": make_ratio(close, "HYG",  "SHY"),  "invert": False, "weight": 0.18},
        "SMH:SPY":   {"series": make_ratio(close, "SMH",  "SPY"),  "invert": False, "weight": 0.14},
        "SPY:VXX":   {"series": make_ratio(close, "SPY",  "VXX"),  "invert": False, "weight": 0.08},
        "XLF:SPY":   {"series": make_ratio(close, "XLF",  "SPY"),  "invert": False, "weight": 0.10},
        "RSP:SPY":   {"series": make_ratio(close, "RSP",  "SPY"),  "invert": False, "weight": 0.08},
        "IWM:SPY":   {"series": make_ratio(close, "IWM",  "SPY"),  "invert": False, "weight": 0.08},
        "XLY:SPY":   {"series": make_ratio(close, "XLY",  "SPY"),  "invert": False, "weight": 0.06},
        "SOXX:SPY":  {"series": make_ratio(close, "SOXX", "SPY"),  "invert": False, "weight": 0.06},
    }
    scores = {}
    for name, cfg in ratios.items():
        s = cfg["series"].dropna()
        if s.empty: continue
        ind = indicator_pack(s)
        if ind.empty: continue
        sc = score_from_indicators(ind)
        if sc.empty: continue
        scores[name] = -sc if cfg["invert"] else sc
    
    if not scores: return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), pd.Series()
    
    all_idx = scores[list(scores.keys())[0]].index
    for sc in scores.values(): all_idx = all_idx.intersection(sc.index)
    scores_df = pd.DataFrame({k: v.loc[all_idx] for k, v in scores.items()}).dropna()
    if scores_df.empty: return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), pd.Series()
    
    w = pd.Series({k: ratios[k]["weight"] for k in scores_df.columns})
    w = w / w.sum()
    composite = (scores_df * w).sum(axis=1)
    
    # FIXED CONFIDENCE CALCULATION
    comp_sign = np.sign(composite.replace(0, np.nan))
    align = (np.sign(scores_df).replace(0, np.nan).eq(comp_sign, axis=0)).mean(axis=1).fillna(0)
    strength = scores_df.abs().mean(axis=1)
    confidence = (0.6 * align + 0.4 * strength) * 100.0
    
    credit_gate = (scores_df["HYG:SHY"] > 0) if "HYG:SHY" in scores_df.columns else pd.Series(False, index=composite.index)
    stress_gate = (scores_df["SPXS:SVOL"] > 0) if "SPXS:SVOL" in scores_df.columns else pd.Series(False, index=composite.index)
    return scores_df, composite, confidence, credit_gate, stress_gate

def build_positions_3state(composite, confidence, spy_close, buy_thr, sell_thr, conf_thr, use_sma=True, use_gates=False, credit_gate=None, stress_gate=None):
    df = pd.DataFrame({"comp": composite, "conf": confidence, "spy": spy_close.reindex(composite.index).ffill()}).dropna()
    if df.empty: return pd.DataFrame()
    
    if use_sma:
        df["sma_200"] = df["spy"].rolling(200).mean()
        df["bull_market"] = (df["spy"] > df["sma_200"]).astype(int)
    else:
        df["bull_market"] = 1
        
    if use_gates and credit_gate is not None and stress_gate is not None:
        df["credit"] = credit_gate.reindex(df.index).fillna(False).astype(int)
        df["stress"] = stress_gate.reindex(df.index).fillna(False).astype(int)
    else:
        df["credit"] = 1
        df["stress"] = 1

    df["trend"] = df["comp"] - df["comp"].shift(10)
    df = df.dropna()
    
    # 3-State Logic: 1=SPY, 0=SHY, -1=SH
    position = pd.Series(0, index=df.index) 
    
    # LONG SPY: Composite > Buy + Confidence + Bull Market + Gates
    long_cond = (df["comp"] > buy_thr) & (df["conf"] >= conf_thr) & (df["bull_market"] == 1)
    if use_gates:
        long_cond &= (df["credit"] == 1) & (df["stress"] == 1)
    position[long_cond] = 1
    
    # SHORT SH: Composite < Sell + Confidence + Bear Market (Below 200 SMA) + Gates
    short_cond = (df["comp"] < sell_thr) & (df["conf"] >= conf_thr) & (df["bull_market"] == 0)
    if use_gates:
        short_cond &= (df["credit"] == 0) & (df["stress"] == 0)
    position[short_cond] = -1
    
    # Hold previous position if no signal
    for i in range(1, len(position)):
        if position.iloc[i] == 0:
            position.iloc[i] = position.iloc[i-1]
            
    df["position"] = position
    df["asset"] = position.map({1: "SPY", 0: "SHY", -1: "SH"})
    return df

def backtest_3state(spy, shy, sh, pos_df, cost_bps=5.0):
    idx = spy.index.intersection(pos_df.index)
    df = pd.DataFrame({
        "SPY": spy.loc[idx],
        "SHY": shy.loc[idx],
        "SH": sh.loc[idx],
        "pos": pos_df["position"].loc[idx]
    }).dropna()
    
    if df.empty: return pd.Series(), 0
    
    rets = pd.DataFrame({
        "SPY": df["SPY"].pct_change().fillna(0),
        "SHY": df["SHY"].pct_change().fillna(0),
        "SH": df["SH"].pct_change().fillna(0)
    })
    
    strat_ret = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        prev_pos = df["pos"].iloc[i-1] if i > 0 else df["pos"].iloc[i]
        prev_asset = "SPY" if prev_pos == 1 else ("SH" if prev_pos == -1 else "SHY")
        strat_ret.iloc[i] = rets[prev_asset].iloc[i]
        
    turnover = (df["pos"].diff().abs().fillna(0) > 0).astype(int)
    strat_ret -= turnover * (cost_bps / 10000.0)
    
    equity = (1.0 + strat_ret).cumprod()
    return equity, turnover.sum()

# ==========
# UI
# ==========
st.set_page_config(page_title="Regime Turbo: SPY/SHY/SH", layout="wide")
st.title("🚀 Regime Turbo: SPY/SHY/SH Rotation")
st.caption("Fixes: Confidence Calculation, 200-SMA Filter, Shorting Logic")

st.sidebar.header("Strategy Controls")
years = st.sidebar.slider("History (years)", 3, 10, 5)
buy_thr = st.sidebar.slider("BUY Threshold (Long SPY)", 0.05, 0.60, 0.10, 0.05)
sell_thr = st.sidebar.slider("SELL Threshold (Short SH)", -0.60, -0.05, -0.10, 0.05)
conf_thr = st.sidebar.slider("Min Confidence", 30, 95, 35, 5) # Lowered default
use_sma = st.sidebar.checkbox("Use 200-SMA Safety Filter", value=True)
use_gates = st.sidebar.checkbox("Use Credit/Stress Gates", value=False) # Default OFF to allow signals
cost_bps = st.sidebar.slider("Trading Cost (bps)", 0.0, 50.0, 5.0, 1.0)

TICKERS = ["SPY", "SHY", "SH", "SPXS", "SVOL", "VXX", "HYG", "SMH", "SOXX", "XLF", "RSP", "IWM", "XLY"]

with st.spinner("📥 Fetching data..."):
    close = fetch_adjclose(TICKERS, years=years)

if not all(t in close.columns for t in ["SPY", "SHY", "SH"]):
    st.error("❌ Missing SPY, SHY, or SH data.")
    st.stop()

with st.spinner("🔧 Building signals..."):
    scores_df, composite, confidence, credit_gate, stress_gate = build_composite_scores(close)

if composite.empty:
    st.error("❌ Failed to build signals. Try increasing history.")
    st.stop()

pos_df = build_positions_3state(composite, confidence, close["SPY"], buy_thr, sell_thr, conf_thr, use_sma, use_gates, credit_gate, stress_gate)

if pos_df.empty:
    st.error("❌ No positions generated. Lower confidence threshold or check data.")
    st.stop()

# Backtest
eq_strat, trades = backtest_3state(close["SPY"], close["SHY"], close["SH"], pos_df, cost_bps)
idx = close["SPY"].index.intersection(pos_df.index)
bh = (1.0 + close["SPY"].loc[idx].pct_change().fillna(0)).cumprod()

# Stats
stats = {
    "Strategy: SPY/SHY/SH": perf_stats(eq_strat),
    "Buy&Hold: SPY": perf_stats(bh)
}
rows = []
for name, s in stats.items():
    rows.append({
        "Portfolio": name,
        "Total Return %": f"{s['Total Return']*100:.1f}" if not np.isnan(s['Total Return']) else "N/A",
        "CAGR %": f"{s['CAGR']*100:.2f}" if not np.isnan(s['CAGR']) else "N/A",
        "Max DD %": f"{s['Max Drawdown']*100:.1f}" if not np.isnan(s['Max Drawdown']) else "N/A",
        "Sharpe": f"{s['Sharpe']:.2f}" if not np.isnan(s['Sharpe']) else "N/A",
        "Trades": int(trades) if "Strategy" in name else 0
    })

st.subheader("Performance Comparison")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# Chart
st.subheader("Equity Curves ($10k Start)")
fig = plt.figure(figsize=(12, 6))
plt.plot(eq_strat.index, eq_strat * 10000, label="SPY/SHY/SH Rotation", linewidth=2)
plt.plot(bh.index, bh * 10000, label="Buy&Hold SPY", linestyle='--', linewidth=2)
plt.xlabel("Date"); plt.ylabel("Equity ($)"); plt.legend(); plt.grid(True, alpha=0.3)
st.pyplot(fig)

# Diagnostics
st.subheader("🔍 Signal Diagnostics (Last 50 Bars)")
diag = pos_df.tail(50)[["comp", "conf", "bull_market", "position", "asset"]].copy()
st.dataframe(diag, use_container_width=True)
st.caption("Position: 1=SPY, 0=SHY, -1=SH. Shorting only allowed if bull_market=0 (SPY < 200 SMA).")

# Debug Info
with st.expander("🔍 Debug Info"):
    st.write(f"Confidence Range: {confidence.min():.1f} - {confidence.max():.1f}")
    st.write(f"Signals > Threshold: {(confidence >= conf_thr).sum()}")
    st.write(f"Current Asset: {pos_df['asset'].iloc[-1]}")
