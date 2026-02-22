import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

# =========================
# LOCKED INDICATOR SETTINGS
# =========================
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 24, 52, 18
TSI_R, TSI_S, TSI_SIGNAL = 40, 20, 10
STOCH_LEN, STOCH_SMOOTHK, STOCH_SMOOTHD = 14, 3, 3
CCI_LEN = 100  # close-only CCI-like

TRADING_DAYS = 252

# ==========
# UTILITIES
# ==========
@st.cache_data(ttl=3600)
def fetch_adjclose(tickers, years=5):
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(years * 365.25) + 30)
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance shape depends on # tickers
    if isinstance(tickers, str) or len(tickers) == 1:
        adj = data["Close"].to_frame(tickers if isinstance(tickers, str) else tickers[0])
    else:
        close_cols = {}
        for t in tickers:
            if (t in data.columns.get_level_values(0)) and ("Close" in data[t].columns):
                close_cols[t] = data[t]["Close"]
        adj = pd.DataFrame(close_cols)

    adj = adj.dropna(how="all").ffill().dropna()
    return adj

def normalize_100(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    return 100.0 * s / float(s.iloc[0])

def close_only_stoch(close: pd.Series, length=14, smoothk=3, smoothd=3):
    lo = close.rolling(length).min()
    hi = close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = 100.0 * (close - lo) / denom
    k = k.rolling(smoothk).mean()
    d = k.rolling(smoothd).mean()
    return k, d

def close_only_cci(close: pd.Series, length=100):
    sma = close.rolling(length).mean()
    mad = (close - sma).abs().rolling(length).mean()
    return (close - sma) / (0.015 * mad.replace(0, np.nan))

def indicator_pack(close: pd.Series) -> pd.DataFrame:
    close = close.dropna()
    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    macdh = macd.iloc[:, 1]  # histogram
    tsi = ta.tsi(close, fast=TSI_S, slow=TSI_R, signal=TSI_SIGNAL).iloc[:, 0]
    stoch_k, _ = close_only_stoch(close, length=STOCH_LEN, smoothk=STOCH_SMOOTHK, smoothd=STOCH_SMOOTHD)
    cci = close_only_cci(close, length=CCI_LEN)

    out = pd.concat(
        [macdh.rename("macdh"), tsi.rename("tsi"), stoch_k.rename("stochk"), cci.rename("cci")],
        axis=1
    ).dropna()
    return out

def score_from_indicators(ind: pd.DataFrame) -> pd.Series:
    # +1 bull, -1 bear, else 0
    bull = (ind["macdh"] > 0) & (ind["tsi"] > 0) & (ind["stochk"] > 50) & (ind["cci"] > 0)
    bear = (ind["macdh"] < 0) & (ind["tsi"] < 0) & (ind["stochk"] < 50) & (ind["cci"] < 0)
    sc = pd.Series(0.0, index=ind.index)
    sc[bull] = 1.0
    sc[bear] = -1.0
    return sc

def make_ratio(close_df: pd.DataFrame, num: str, den: str) -> pd.Series:
    return (close_df[num] / close_df[den]).rename(f"{num}:{den}")

def perf_stats(equity: pd.Series) -> dict:
    eq = equity.dropna()
    if len(eq) < 5:
        return {}
    rets = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(TRADING_DAYS)) if rets.std() != 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Ann. Vol": vol,
        "Sharpe (rf=0)": sharpe
    }

def build_composite_scores(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    # Components + weights
    ratios = {
        "SPXS:SVOL (Stress/Carry)": {"series": make_ratio(close, "SPXS", "SVOL"), "invert": True,  "weight": 0.22},
        "HYG:SHY (Credit)":         {"series": make_ratio(close, "HYG",  "SHY"),  "invert": False, "weight": 0.18},
        "SMH:SPY (Semis Lead)":     {"series": make_ratio(close, "SMH",  "SPY"),  "invert": False, "weight": 0.14},
        "SOXX:SPY (Semis Alt)":     {"series": make_ratio(close, "SOXX", "SPY"),  "invert": False, "weight": 0.06},
        "SPY:VXX (Vol Confirm)":    {"series": make_ratio(close, "SPY",  "VXX"),  "invert": False, "weight": 0.08},
        "XLF:SPY (Financials)":     {"series": make_ratio(close, "XLF",  "SPY"),  "invert": False, "weight": 0.10},
        "RSP:SPY (Equal-weight)":   {"series": make_ratio(close, "RSP",  "SPY"),  "invert": False, "weight": 0.08},
        "IWM:SPY (Small caps)":     {"series": make_ratio(close, "IWM",  "SPY"),  "invert": False, "weight": 0.08},
        "XLY:SPY (Discretionary)":  {"series": make_ratio(close, "XLY",  "SPY"),  "invert": False, "weight": 0.06},
    }

    # Score each component
    scores = {}
    for name, cfg in ratios.items():
        s = cfg["series"].dropna()
        ind = indicator_pack(s)
        sc = score_from_indicators(ind)
        if cfg["invert"]:
            sc = -sc
        scores[name] = sc

    # Align to common index
    all_idx = None
    for sc in scores.values():
        all_idx = sc.index if all_idx is None else all_idx.intersection(sc.index)

    scores_df = pd.DataFrame({k: v.loc[all_idx] for k, v in scores.items()}).dropna()

    w = pd.Series({k: ratios[k]["weight"] for k in scores_df.columns})
    w = w / w.sum()

    composite = (scores_df * w).sum(axis=1)

    # Confidence
    comp_sign = np.sign(composite.replace(0, np.nan))
    align = (np.sign(scores_df).replace(0, np.nan).eq(comp_sign, axis=0)).mean(axis=1).fillna(0)
    strength = scores_df.abs().mean(axis=1)
    confidence = (0.6 * align + 0.4 * strength) * 100.0

    # Gates
    credit_gate = (scores_df["HYG:SHY (Credit)"] > 0)
    stress_gate = (scores_df["SPXS:SVOL (Stress/Carry)"] > 0)

    return scores_df, composite, confidence, credit_gate, stress_gate

def signal_labels(comp_level, comp_trend):
    if comp_level >= 0:
        return "Healing + improving ✅" if comp_trend >= 0 else "Healing but deteriorating ⚠️"
    else:
        return "Risk-off but improving (bottoming) ✅" if comp_trend >= 0 else "Risk-off and deteriorating 🚨"

def build_positions(composite, confidence, credit_gate, stress_gate,
                    buy_thr, sell_thr, conf_thr, trend_window):
    df = pd.DataFrame({
        "comp": composite,
        "conf": confidence,
        "credit": credit_gate.astype(int),
        "stress": stress_gate.astype(int),
    }).dropna()
    df["trend"] = df["comp"] - df["comp"].shift(trend_window)
    df = df.dropna()

    buy = (df["comp"] > buy_thr) & (df["trend"] > 0) & (df["conf"] >= conf_thr) & (df["credit"] == 1) & (df["stress"] == 1)
    sell = (df["comp"] < sell_thr) & (df["trend"] < 0) & (df["conf"] >= conf_thr) & (df["credit"] == 0) & (df["stress"] == 0)

    pos_spy = pd.Series(index=df.index, dtype=float)
    pos_spy.iloc[0] = 1.0
    for i in range(1, len(df)):
        if buy.iloc[i]:
            pos_spy.iloc[i] = 1.0
        elif sell.iloc[i]:
            pos_spy.iloc[i] = 0.0
        else:
            pos_spy.iloc[i] = pos_spy.iloc[i-1]

    df["buy"] = buy.astype(int)
    df["sell"] = sell.astype(int)
    df["pos_spy"] = pos_spy
    return df

def backtest_two_assets(price_spy, price_def, pos_spy, cost_bps=0.0, cash_mode=False):
    """
    cash_mode=True: defensive asset earns 0% daily return (cash)
    else: defensive asset uses price_def returns.
    """
    idx = price_spy.index.intersection(pos_spy.index)
    df = pd.DataFrame({
        "SPY": price_spy.loc[idx],
        "DEF": price_def.loc[idx] if (price_def is not None) else np.nan,
        "pos_spy": pos_spy.loc[idx],
    }).dropna(subset=["SPY", "pos_spy"]).copy()

    spy_ret = df["SPY"].pct_change().fillna(0)
    if cash_mode:
        def_ret = pd.Series(0.0, index=df.index)
    else:
        def_ret = df["DEF"].pct_change().fillna(0)

    pos_def = 1.0 - df["pos_spy"]

    strat_ret = df["pos_spy"].shift(1).fillna(df["pos_spy"].iloc[0]) * spy_ret + \
                pos_def.shift(1).fillna(pos_def.iloc[0]) * def_ret

    turnover = (df["pos_spy"].diff().abs().fillna(0) > 0).astype(int)
    cost = turnover * (cost_bps / 10000.0)
    strat_ret = strat_ret - cost

    equity = (1.0 + strat_ret).cumprod()
    return equity, turnover.sum()

# =========================
# UI
# =========================
st.set_page_config(page_title="Market Regime Dashboard (3 Allocations)", layout="wide")
st.title("📊 Market Regime Dashboard — Compare 3 Allocations")
st.caption("Same regime signal, evaluated as: (1) SPY/Cash, (2) SPY/SHY, (3) SPY/AGG. "
           "Locked indicators: MACD(24,52,18), TSI(40,20,10), Stoch(14,3,3), CCI(100-close-only).")

st.sidebar.header("Controls")
years = st.sidebar.slider("History (years)", 3, 10, 5)
trend_window = st.sidebar.selectbox("Trend window (days)", [5, 10, 15, 20], index=1)
buy_thr = st.sidebar.slider("BUY threshold (Composite)", 0.05, 0.60, 0.20, 0.05)
sell_thr = st.sidebar.slider("SELL threshold (Composite)", -0.60, -0.05, -0.20, 0.05)
conf_thr = st.sidebar.slider("Min Confidence", 40, 95, 70, 5)
cost_bps = st.sidebar.slider("Trading cost (bps per switch)", 0.0, 25.0, 0.0, 0.5)

# Data
TICKERS = [
    "SPY", "SHY", "AGG",
    "SPXS", "SVOL", "VXX",
    "HYG",
    "SMH", "SOXX", "XLF", "RSP", "IWM", "XLY",
]
close = fetch_adjclose(TICKERS, years=years)

# Composite
scores_df, composite, confidence, credit_gate, stress_gate = build_composite_scores(close)

# Positions
pos_df = build_positions(composite, confidence, credit_gate, stress_gate,
                         buy_thr=buy_thr, sell_thr=sell_thr, conf_thr=conf_thr, trend_window=trend_window)

# Latest metrics
latest = pos_df.iloc[-1]
comp_level = float(latest["comp"])
comp_trend = float(latest["trend"])
conf_val = float(latest["conf"])
label = signal_labels(comp_level, comp_trend)

action = "HOLD / NEUTRAL"
if latest["buy"] == 1:
    action = "BUY (Long SPY)"
elif latest["sell"] == 1:
    action = "SELL (Risk-off)"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Composite Level (latest)", f"{comp_level:.2f}")
c2.metric(f"Composite Trend ({trend_window}d Δ)", f"{comp_trend:.2f}")
c3.metric("Confidence Score", f"{conf_val:.0f}")
c4.metric("Regime Label", label)
st.info(f"**{action}**")

# Backtests (3 allocations)
spy = close["SPY"].dropna()
shy = close["SHY"].dropna()
agg = close["AGG"].dropna()

eq_cash, trades_cash = backtest_two_assets(spy, None, pos_df["pos_spy"], cost_bps=cost_bps, cash_mode=True)
eq_shy, trades_shy   = backtest_two_assets(spy, shy,  pos_df["pos_spy"], cost_bps=cost_bps, cash_mode=False)
eq_agg, trades_agg   = backtest_two_assets(spy, agg,  pos_df["pos_spy"], cost_bps=cost_bps, cash_mode=False)

# Buy & hold SPY baseline
idx = spy.index.intersection(pos_df.index)
bh = (1.0 + spy.loc[idx].pct_change().fillna(0)).cumprod()

# Stats table
stats_rows = []
for name, eq, trades in [
    ("Strategy: SPY/CASH", eq_cash, trades_cash),
    ("Strategy: SPY/SHY",  eq_shy,  trades_shy),
    ("Strategy: SPY/AGG",  eq_agg,  trades_agg),
    ("Buy&Hold: SPY",      bh,      0),
]:
    s = perf_stats(eq)
    stats_rows.append({
        "Portfolio": name,
        "Total Return %": round(s["Total Return"] * 100, 1),
        "CAGR %": round(s["CAGR"] * 100, 2),
        "Max DD %": round(s["Max Drawdown"] * 100, 1),
        "Ann Vol %": round(s["Ann. Vol"] * 100, 1),
        "Sharpe (rf=0)": round(s["Sharpe (rf=0)"], 2),
        "Trades": int(trades),
    })

st.subheader("Performance Comparison (same signal, 3 allocations)")
st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

# Equity curves
st.subheader("Equity Curves (Normalized, $10k start)")
fig = plt.figure()
base = 10000
plt.plot(eq_cash.index, eq_cash * base, label="SPY/CASH")
plt.plot(eq_shy.index,  eq_shy  * base, label="SPY/SHY")
plt.plot(eq_agg.index,  eq_agg  * base, label="SPY/AGG")
plt.plot(bh.index,      bh      * base, label="Buy&Hold SPY")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
st.pyplot(fig)

# Overlay: Composite vs SPY
st.subheader("Overlay: Composite vs SPY (Normalized)")
overlay = pd.DataFrame({
    "Composite": normalize_100((pos_df["comp"] + 2.0).loc[idx]),  # shift for stability
    "SPY": normalize_100(spy.loc[idx]),
}).dropna()

fig2 = plt.figure()
plt.plot(overlay.index, overlay["Composite"], label="Composite (shifted+normalized)")
plt.plot(overlay.index, overlay["SPY"], label="SPY (normalized)")
plt.xlabel("Date")
plt.ylabel("Normalized (base=100)")
plt.legend()
st.pyplot(fig2)

# Components table
st.subheader("Components (Latest)")
w = pd.Series({
    "SPXS:SVOL (Stress/Carry)": 0.22,
    "HYG:SHY (Credit)": 0.18,
    "SMH:SPY (Semis Lead)": 0.14,
    "SOXX:SPY (Semis Alt)": 0.06,
    "SPY:VXX (Vol Confirm)": 0.08,
    "XLF:SPY (Financials)": 0.10,
    "RSP:SPY (Equal-weight)": 0.08,
    "IWM:SPY (Small caps)": 0.08,
    "XLY:SPY (Discretionary)": 0.06,
})
w = w / w.sum()
rows = []
for col in scores_df.columns:
    rows.append({
        "Component": col,
        "Weight": float(w[col]),
        "Latest Score (-1/0/+1)": float(scores_df[col].iloc[-1]),
        "Weighted (latest)": float(scores_df[col].iloc[-1] * w[col]),
    })
st.dataframe(pd.DataFrame(rows).sort_values("Weight", ascending=False), use_container_width=True)

# Recent signals
st.subheader("Recent Signals (last 120 bars)")
tail = pos_df.tail(120).copy()
tail_out = tail[["comp", "trend", "conf", "credit", "stress", "buy", "sell", "pos_spy"]].rename(columns={
    "comp":"Composite",
    "trend":"Composite Trend",
    "conf":"Confidence",
    "credit":"Credit Gate",
    "stress":"Stress Gate",
    "buy":"BUY Signal",
    "sell":"SELL Signal",
    "pos_spy":"Position (1=SPY,0=DEF)"
})
st.dataframe(tail_out, use_container_width=True)
