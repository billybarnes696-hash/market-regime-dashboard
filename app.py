import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import altair as alt
from datetime import datetime

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(page_title="Robust Market Regime Dashboard", layout="wide")
st.title("🧭 Robust Market Regime Dashboard (5y + SPY overlay)")
st.caption("Ratios + MACD/TSI/Stoch canaries + optional breadth → composite regime score with SPY overlay")

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
with st.sidebar:
    st.header("Settings")

    period = st.selectbox("History window", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Interval", ["1d"], index=0)  # keep daily for stability

    st.subheader("Indicators")
    stoch_len = st.slider("Canary Stoch length", 10, 60, 14)
    tsi_long = st.slider("TSI long", 10, 80, 40)
    tsi_short = st.slider("TSI short", 5, 40, 20)
    tsi_signal = st.slider("TSI signal", 3, 30, 10)

    st.subheader("Confirmation")
    confirm_days = st.slider("Confirm crosses (days)", 1, 5, 2)

    st.subheader("Breadth (optional)")
    use_breadth = st.toggle("Compute breadth (slower)", value=True)
    breadth_max_tickers = st.slider("Breadth basket size", 20, 140, 60, step=10)

    st.subheader("Scoring weights")
    w_stress = st.slider("Stress weight (SPXS:SVOL)", 0.05, 0.50, 0.25, 0.05)
    w_credit = st.slider("Credit weight (HYG:SHY)", 0.05, 0.50, 0.25, 0.05)
    w_lead = st.slider("Leadership weight (SOXX:SPY)", 0.05, 0.50, 0.20, 0.05)
    w_fin = st.slider("Financials weight (XLF:SPY)", 0.05, 0.50, 0.20, 0.05)
    w_house = st.slider("Housing weight (ITB:SPY)", 0.00, 0.30, 0.10, 0.05)
    w_breadth = st.slider("Breadth weight", 0.00, 0.40, 0.20, 0.05)

    st.subheader("Alert thresholds")
    risk_off_th = st.slider("Risk-Off if Composite ≤", -1.0, 0.0, -0.35, 0.05)
    risk_on_th = st.slider("Risk-On if Composite ≥", 0.0, 1.0, 0.35, 0.05)

# Normalize weights so composite stays comparable
weights = np.array([w_stress, w_credit, w_lead, w_fin, w_house, w_breadth], dtype=float)
wsum = weights.sum()
if wsum == 0:
    st.error("All weights are zero. Increase at least one weight.")
    st.stop()
weights = weights / wsum
w_stress, w_credit, w_lead, w_fin, w_house, w_breadth = weights.tolist()

# ---------------------------
# COMPONENT DEFINITIONS
# invert=True means rising ratio = more stress (bad), so score is negated
# ---------------------------
COMPONENTS = [
    {"name": "Stress vs Carry", "a": "SPXS", "b": "SVOL", "invert": True,  "weight": w_stress},
    {"name": "Credit Gate",     "a": "HYG",  "b": "SHY",  "invert": False, "weight": w_credit},
    {"name": "Semis Lead",      "a": "SOXX", "b": "SPY",  "invert": False, "weight": w_lead},
    {"name": "Financials Lead", "a": "XLF",  "b": "SPY",  "invert": False, "weight": w_fin},
    {"name": "Housing/Cyclic",  "a": "ITB",  "b": "SPY",  "invert": False, "weight": w_house},
]

ratio_tickers = sorted({c["a"] for c in COMPONENTS} | {c["b"] for c in COMPONENTS} | {"SPY"})

# Breadth universe (practical, liquid mix)
BREADTH_TICKERS_BASE = [
    "SPY","QQQ","IWM","MDY","IJR","RSP",
    "XLF","XLK","XLI","XLY","XLP","XLV","XLE","XLB",
    "SOXX","SMH","ITB","XHB","XRT","XME",
    "HYG","SHY",
    "EWY","EWG","EEM","FXI","EWZ",
    "NVDA","AMD","AMAT","MSFT","AMZN","NFLX",
    "JPM","BAC","WFC","GS","MS","C",
]
BREADTH_TICKERS_PAD = [
    "AAPL","GOOGL","META","TSLA","INTC","MU","DIS","MA","HD","NKE","FDX","CAT","DE",
    "XBI","IBB","VNQ","IYR","GLD","SLV","USO","XOP","OIH","KRE","KBE","VUG","IWF","IWD","VTV"
]
breadth_tickers = (BREADTH_TICKERS_BASE + BREADTH_TICKERS_PAD)[:breadth_max_tickers]

# ---------------------------
# HELPERS
# ---------------------------
@st.cache_data(ttl=3600)
def download_closes(tickers, period="5y", interval="1d") -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        return pd.DataFrame(closes).dropna(how="all")
    else:
        if "Close" in df.columns and len(tickers) == 1:
            return df[["Close"]].rename(columns={"Close": tickers[0]}).dropna(how="all")
        return pd.DataFrame()

def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([a, b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return (df.iloc[:, 0] / df.iloc[:, 1]).dropna()

def canary_stoch(x: pd.Series, length: int = 14) -> pd.Series:
    x = x.dropna()
    low = x.rolling(length).min()
    high = x.rolling(length).max()
    denom = (high - low).replace(0, np.nan)
    return 100 * (x - low) / denom

def proxy_cci(x: pd.Series, length: int = 100) -> pd.Series:
    x = x.dropna()
    sma = x.rolling(length).mean()
    mad = (x - sma).abs().rolling(length).mean()
    return (x - sma) / (0.015 * mad)

def rolling_confirm_sign(x: pd.Series, days: int = 2) -> pd.Series:
    # +1 if last N values > 0, -1 if last N values < 0, else 0
    def f(arr):
        if np.all(arr > 0):
            return 1.0
        if np.all(arr < 0):
            return -1.0
        return 0.0
    return x.rolling(days).apply(f, raw=True)

def label_regime(score: float) -> str:
    if score <= -0.50:
        return "🟥 Risk-Off"
    if score <= -0.15:
        return "🟧 Deteriorating"
    if score < 0.15:
        return "🟨 Transition"
    if score < 0.50:
        return "🟩 Healing"
    return "🟦 Risk-On"

def z_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty or s.std() == 0:
        return pd.Series(index=series.index, dtype=float)
    z = (s - s.mean()) / s.std()
    # map to a “price-like” band around 100
    return (100 + 10 * z).reindex(series.index)

def rebase_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)
    return (100 * (s / s.iloc[0])).reindex(series.index)

# ---------------------------
# DOWNLOAD RATIO DATA
# ---------------------------
closes_ratio = download_closes(ratio_tickers, period=period, interval=interval)
if closes_ratio.empty:
    st.error("Could not download price data from Yahoo. Try rebooting or changing window.")
    st.stop()

# ---------------------------
# BUILD DAILY SCORE SERIES PER COMPONENT
# ---------------------------
component_rows = []
hist = pd.DataFrame(index=closes_ratio.index)

for c in COMPONENTS:
    a, b = c["a"], c["b"]

    if a not in closes_ratio.columns or b not in closes_ratio.columns:
        component_rows.append([c["name"], f"{a}:{b}", "missing"])
        continue

    ratio = safe_ratio(closes_ratio[a], closes_ratio[b])
    if ratio.empty or len(ratio) < 250:
        component_rows.append([c["name"], f"{a}:{b}", "too little data"])
        continue

    # Indicators on ratio (ratio-safe)
    macd_line = ta.macd(ratio, fast=24, slow=52, signal=18).iloc[:, 0]
    tsi_df = ta.tsi(ratio, long=tsi_long, short=tsi_short, signal=tsi_signal)
    tsi_line = tsi_df.iloc[:, 0] if isinstance(tsi_df, pd.DataFrame) else tsi_df

    stoch_k = canary_stoch(ratio, length=stoch_len)
    cci = proxy_cci(ratio, length=100)

    macd_sign = rolling_confirm_sign(macd_line, days=confirm_days)
    tsi_sign = rolling_confirm_sign(tsi_line, days=confirm_days)

    stoch_zone = pd.Series(0.0, index=ratio.index)
    stoch_zone[stoch_k <= 20] = +0.5
    stoch_zone[stoch_k >= 80] = -0.5

    cci_zone = pd.Series(0.0, index=ratio.index)
    cci_zone[cci >= 100] = +0.5
    cci_zone[cci <= -100] = -0.5

    score = 0.4 * macd_sign + 0.4 * tsi_sign + 0.1 * stoch_zone + 0.1 * cci_zone
    score = score.reindex(hist.index)

    if c["invert"]:
        score = -score

    hist[c["name"]] = score
    component_rows.append([c["name"], f"{a}:{b}", "ok"])

# ---------------------------
# COMPOSITE HISTORY (ratios-only)
# ---------------------------
hist["Composite (ratios)"] = 0.0
for c in COMPONENTS:
    if c["name"] in hist.columns:
        hist["Composite (ratios)"] += hist[c["name"]] * c["weight"]

composite_series = hist["Composite (ratios)"].dropna()

# ---------------------------
# OPTIONAL: LATEST BREADTH (not full history)
# ---------------------------
breadth_score = np.nan
breadth_metrics = {}

if use_breadth:
    closes_b = download_closes(breadth_tickers, period=period, interval=interval)
    if not closes_b.empty and len(closes_b) >= 220:
        sma50 = closes_b.rolling(50).mean()
        sma200 = closes_b.rolling(200).mean()

        latest = closes_b.iloc[-1]
        above50 = (latest > sma50.iloc[-1]).mean() * 100
        above200 = (latest > sma200.iloc[-1]).mean() * 100
        above_both = ((latest > sma50.iloc[-1]) & (latest > sma200.iloc[-1])).mean() * 100

        def pct_to_score(p):
            return float(np.clip((p - 50) / 10, -1, 1))  # 40=-1, 50=0, 60=+1

        breadth_score = 0.4 * pct_to_score(above200) + 0.4 * pct_to_score(above50) + 0.2 * pct_to_score(above_both)

        breadth_metrics = {
            "%>50DMA": above50,
            "%>200DMA": above200,
            "%>50&200": above_both,
            "n": int(closes_b.shape[1])
        }

# Latest composite including breadth (latest point only)
latest_composite = float(composite_series.iloc[-1]) if not composite_series.empty else np.nan
if pd.notna(breadth_score):
    latest_composite = float(latest_composite + breadth_score * w_breadth)

regime = label_regime(latest_composite) if pd.notna(latest_composite) else "—"

alert = "—"
if pd.notna(latest_composite):
    if latest_composite <= risk_off_th:
        alert = "🚨 RISK-OFF ALERT: defensive posture favored"
    elif latest_composite >= risk_on_th:
        alert = "✅ RISK-ON / HEALING: risk appetite improving"
    else:
        alert = "⚠️ TRANSITION: mixed signals (watch confirmations)"

# ---------------------------
# TOP SUMMARY
# ---------------------------
c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    st.subheader("🧠 Composite Regime (latest)")
    st.metric("Regime", regime, f"{latest_composite:.2f}" if pd.notna(latest_composite) else "—")
    st.write(f"**Alert:** {alert}")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)")

with c2:
    st.subheader("🧪 Breadth (latest)")
    if use_breadth and breadth_metrics:
        st.metric("% above 50DMA", f"{breadth_metrics['%>50DMA']:.1f}%")
        st.metric("% above 200DMA", f"{breadth_metrics['%>200DMA']:.1f}%")
        st.metric("% above 50&200", f"{breadth_metrics['%>50&200']:.1f}%")
        st.caption(f"Basket size: {breadth_metrics['n']} tickers")
        st.write(f"**Breadth score:** {breadth_score:.2f}")
    else:
        st.info("Breadth is off or not available. Toggle it on in the sidebar.")

with c3:
    st.subheader("🧩 What to watch")
    st.write(
        """
- **SPXS:SVOL** = stress canary (inverted in the score)
- **HYG:SHY** = credit gate (healing vs deterioration)
- **SOXX:SPY** = leadership health
- **XLF:SPY** = recovery/credit-cycle tell
- **ITB:SPY** = cyclical/housing confirmation
- **Canary Stoch** highlights turns; MACD/TSI confirm regime
        """.strip()
    )

st.divider()

# ---------------------------
# COMPONENT TABLE (latest values)
# ---------------------------
st.subheader("📦 Component Health (latest)")
rows = []
for c in COMPONENTS:
    name = c["name"]
    if name not in hist.columns:
        continue
    s = hist[name].dropna()
    if s.empty:
        continue
    latest = float(s.iloc[-1])
    rows.append({
        "Component": name,
        "Weight": round(c["weight"], 3),
        "Latest Score": round(latest, 3),
        "Weighted": round(latest * c["weight"], 3)
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------------------------
# CHART 1: RAW COMPOSITE HISTORY (5y)
# ---------------------------
st.subheader("📈 Composite History (raw, ratios-only)")
st.caption("This is your daily composite regime score built from ratio canaries (full history, not capped).")
raw_df = composite_series.to_frame("Composite (ratios)").dropna()
st.line_chart(raw_df, use_container_width=True)

# ---------------------------
# CHART 2: OVERLAY COMPOSITE vs SPY (both rebased around ~100)
# ---------------------------
st.subheader("🧪 Predictiveness check: Composite overlay vs SPY (5y)")
st.caption("SPY is rebased to 100. Composite is z-scored and mapped around 100 so you can compare turns visually.")

spy_close = download_closes(["SPY"], period=period, interval=interval)["SPY"].dropna()

df_overlay = pd.DataFrame({
    "SPY_100": rebase_100(spy_close),
    "Composite_100": z_to_100(composite_series),
}).dropna()

if df_overlay.empty:
    st.warning("Not enough data to build the overlay chart.")
else:
    plot_df = df_overlay.reset_index().rename(columns={"index": "Date"})
    chart = (
        alt.Chart(plot_df)
        .transform_fold(
            ["SPY_100", "Composite_100"],
            as_=["Series", "Value"]
        )
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Rebased / Comparable scale (around 100)"),
            color=alt.Color("Series:N"),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=".2f")]
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

with st.expander("Show overlay data (last 60 rows)"):
    st.dataframe(df_overlay.tail(60), use_container_width=True)
