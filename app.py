import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(page_title="Robust Market Regime Dashboard", layout="wide")
st.title("🧭 Robust Market Regime Dashboard")
st.caption("Ratios + MACD/TSI/Stoch canaries + live breadth (% above MAs) → single composite regime score")

# ---------------------------
# USER CONTROLS
# ---------------------------
with st.sidebar:
    st.header("Settings")

    period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)
    stoch_len = st.slider("Canary Stoch length", 10, 60, 14)
    tsi_long = st.slider("TSI long", 10, 80, 40)
    tsi_short = st.slider("TSI short", 5, 40, 20)
    tsi_signal = st.slider("TSI signal", 3, 30, 10)

    st.subheader("Signal confirmation")
    confirm_days = st.slider("Confirm crosses (days)", 1, 5, 2)

    st.subheader("Breadth basket")
    use_breadth = st.toggle("Compute breadth (slower)", value=True)
    breadth_max_tickers = st.slider("Breadth basket size", 20, 120, 60, step=10)

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
# DATA HELPERS
# ---------------------------
@st.cache_data(ttl=3600)
def download_closes(tickers, period="2y") -> pd.DataFrame:
    """
    Downloads adjusted close for tickers. Returns DataFrame with columns=tickers.
    """
    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yf returns either a single-index or multi-index depending on tickers count
    if isinstance(df.columns, pd.MultiIndex):
        # pull Close for each ticker
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        out = pd.DataFrame(closes).dropna(how="all")
        return out
    else:
        # single ticker
        if "Close" in df.columns:
            out = df[["Close"]].rename(columns={"Close": tickers[0]})
            return out.dropna(how="all")
        return pd.DataFrame()

def safe_ratio(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return (df.iloc[:, 0] / df.iloc[:, 1]).dropna()

def canary_stoch(x: pd.Series, length: int = 14) -> pd.Series:
    """
    Stochastic %K for a single series:
    %K = 100 * (x - rolling_low) / (rolling_high - rolling_low)
    Works on ratios and any close series.
    """
    x = x.dropna()
    low = x.rolling(length).min()
    high = x.rolling(length).max()
    denom = (high - low).replace(0, np.nan)
    k = 100 * (x - low) / denom
    return k

def proxy_cci(x: pd.Series, length: int = 100) -> pd.Series:
    """
    CCI-like oscillator for single series.
    """
    x = x.dropna()
    sma = x.rolling(length).mean()
    mad = (x - sma).abs().rolling(length).mean()
    cci = (x - sma) / (0.015 * mad)
    return cci

def confirmed_sign(series: pd.Series, days: int = 2) -> float:
    """
    Returns +1 if last 'days' values are >0, -1 if last 'days' values are <0, else 0.
    """
    if series is None or series.dropna().empty:
        return 0.0
    tail = series.dropna().iloc[-days:]
    if len(tail) < days:
        return 0.0
    if (tail > 0).all():
        return 1.0
    if (tail < 0).all():
        return -1.0
    return 0.0

def label_regime(x: float) -> str:
    if x <= -0.50:
        return "🟥 Risk-Off"
    if x <= -0.15:
        return "🟧 Deteriorating"
    if x < 0.15:
        return "🟨 Transition"
    if x < 0.50:
        return "🟩 Healing"
    return "🟦 Risk-On"

def strength_label(z: float) -> str:
    if z >= 0.7:
        return "Strong +"
    if z >= 0.3:
        return "Moderate +"
    if z > -0.3:
        return "Neutral"
    if z > -0.7:
        return "Moderate -"
    return "Strong -"

# ---------------------------
# DEFINE COMPONENTS
# ---------------------------
# invert=True means "rising ratio = more stress" (bad), so we flip the sign.
COMPONENTS = [
    {"name": "Stress vs Carry", "ticker_a": "SPXS", "ticker_b": "SVOL", "invert": True,  "weight": w_stress},
    {"name": "Credit Gate",     "ticker_a": "HYG",  "ticker_b": "SHY",  "invert": False, "weight": w_credit},
    {"name": "Semis Lead",      "ticker_a": "SOXX", "ticker_b": "SPY",  "invert": False, "weight": w_lead},
    {"name": "Financials Lead", "ticker_a": "XLF",  "ticker_b": "SPY",  "invert": False, "weight": w_fin},
    {"name": "Housing/Cyclic",  "ticker_a": "ITB",  "ticker_b": "SPY",  "invert": False, "weight": w_house},
]

ratio_tickers = sorted({c["ticker_a"] for c in COMPONENTS} | {c["ticker_b"] for c in COMPONENTS} | {"SPY"})

# Breadth universe (practical, not huge)
BREADTH_TICKERS_BASE = [
    # Your high-signal leaders / sectors / globals (mix = more robust)
    "SPY","QQQ","IWM","MDY","IJR","RSP",
    "XLF","XLK","XLI","XLY","XLP","XLV","XLE","XLB",
    "SOXX","SMH","ITB","XHB","XRT","XME",
    "HYG","SHY",
    "EWY","EWG","EEM","FXI","EWZ",
    # Stocks from your leaders (small set, avoids rate limit)
    "NVDA","AMD","AMAT","MSFT","AMZN","NFLX",
    "JPM","BAC","WFC","GS","MS","C",
]

# If user wants larger breadth, we pad with liquid ETFs / megacaps
BREADTH_TICKERS_PAD = [
    "AAPL","GOOGL","META","TSLA","INTC","MU","DIS","MA","HD","NKE","FDX","CAT","DE",
    "XBI","IBB","VNQ","IYR","GLD","SLV","USO","XOP","OIH","KRE","KBE","VUG","IWF","IWD","VTV"
]

breadth_tickers = (BREADTH_TICKERS_BASE + BREADTH_TICKERS_PAD)[:breadth_max_tickers]

# ---------------------------
# DOWNLOAD DATA (cached)
# ---------------------------
closes_ratio = download_closes(ratio_tickers, period=period)

if closes_ratio.empty:
    st.error("Could not download price data (Yahoo). Try rebooting or changing time window.")
    st.stop()

# ---------------------------
# BUILD COMPONENT SIGNALS
# ---------------------------
def component_signals(series: pd.Series) -> dict:
    """
    Compute MACD, TSI, StochK, Proxy CCI from a single series and return a robust score.
    """
    s = series.dropna()
    if len(s) < 200:
        return {"score": np.nan}

    # MACD(24,52,18)
    macd = ta.macd(s, fast=24, slow=52, signal=18)
    macd_line = macd.iloc[:, 0]

    # TSI (ratio-safe)
    tsi = ta.tsi(s, long=tsi_long, short=tsi_short, signal=tsi_signal)
    tsi_line = tsi.iloc[:, 0] if isinstance(tsi, pd.DataFrame) else tsi

    # Canary Stoch
    stoch_k = canary_stoch(s, length=stoch_len)

    # Proxy CCI for additional “CCI-like” momentum
    cci = proxy_cci(s, length=100)

    # Confirmed signs
    macd_sign = confirmed_sign(macd_line, days=confirm_days)  # +1 / 0 / -1
    tsi_sign = confirmed_sign(tsi_line, days=confirm_days)

    # Stoch zones (oversold/overbought)
    stoch_last = stoch_k.dropna().iloc[-1] if not stoch_k.dropna().empty else np.nan
    stoch_zone = 0.0
    if pd.notna(stoch_last):
        if stoch_last <= 20:
            stoch_zone = +0.5  # oversold = potential healing
        elif stoch_last >= 80:
            stoch_zone = -0.5  # overbought in ratio can be “exhaustion”; treat as caution

    # CCI zones
    cci_last = cci.dropna().iloc[-1] if not cci.dropna().empty else np.nan
    cci_zone = 0.0
    if pd.notna(cci_last):
        if cci_last >= 100:
            cci_zone = +0.5
        elif cci_last <= -100:
            cci_zone = -0.5

    # Robust component score combines:
    # MACD sign (0.4), TSI sign (0.4), Stoch zone (0.1), CCI zone (0.1)
    score = 0.4 * macd_sign + 0.4 * tsi_sign + 0.1 * stoch_zone + 0.1 * cci_zone

    return {
        "score": float(score),
        "macd_sign": float(macd_sign),
        "tsi_sign": float(tsi_sign),
        "stoch_k": float(stoch_last) if pd.notna(stoch_last) else np.nan,
        "cci": float(cci_last) if pd.notna(cci_last) else np.nan,
    }

rows = []
composite = 0.0

for c in COMPONENTS:
    a, b = c["ticker_a"], c["ticker_b"]
    if a not in closes_ratio.columns or b not in closes_ratio.columns:
        rows.append([c["name"], f"{a}:{b}", np.nan, np.nan, np.nan, np.nan, "⚠️ missing"])
        continue

    ratio = safe_ratio(closes_ratio[a], closes_ratio[b])
    sig = component_signals(ratio)
    score = sig.get("score", np.nan)

    # invert stress ratio so “stress rising” becomes negative in composite
    if pd.notna(score) and c["invert"]:
        score = -score

    contrib = score * c["weight"] if pd.notna(score) else np.nan
    if pd.notna(contrib):
        composite += contrib

    rows.append([
        c["name"],
        f"{a}:{b}",
        score,
        contrib,
        sig.get("macd_sign", np.nan),
        sig.get("tsi_sign", np.nan),
        sig.get("stoch_k", np.nan),
    ])

comp_df = pd.DataFrame(
    rows,
    columns=["Component", "Ratio", "Score", "Weighted", "MACD sign", "TSI sign", "Canary Stoch %K"]
)

# ---------------------------
# BREADTH CALCULATION
# ---------------------------
breadth_score = np.nan
breadth_metrics = {}

if use_breadth:
    closes_breadth = download_closes(breadth_tickers, period=period)

    if not closes_breadth.empty:
        # compute % above SMA50, SMA200
        sma50 = closes_breadth.rolling(50).mean()
        sma200 = closes_breadth.rolling(200).mean()

        latest = closes_breadth.iloc[-1]
        above50 = (latest > sma50.iloc[-1]).mean() * 100
        above200 = (latest > sma200.iloc[-1]).mean() * 100
        above_both = ((latest > sma50.iloc[-1]) & (latest > sma200.iloc[-1])).mean() * 100

        breadth_metrics = {
            "%>50DMA": above50,
            "%>200DMA": above200,
            "%>50&200": above_both,
            "n": int(closes_breadth.shape[1])
        }

        # Breadth score: map to [-1, +1] centered around 50%
        # >60% = positive, <40% = negative
        def pct_to_score(p):
            # scale 40->-1, 50->0, 60->+1 (clip)
            return float(np.clip((p - 50) / 10, -1, 1))

        s50 = pct_to_score(above50)
        s200 = pct_to_score(above200)
        sboth = pct_to_score(above_both)

        breadth_score = 0.4 * s200 + 0.4 * s50 + 0.2 * sboth

        composite += (breadth_score * w_breadth)
    else:
        breadth_score = np.nan

# ---------------------------
# REGIME / ALERTS
# ---------------------------
regime = label_regime(composite)
alert = "—"

if composite <= risk_off_th:
    alert = "🚨 RISK-OFF ALERT: defensive posture favored"
elif composite >= risk_on_th:
    alert = "✅ RISK-ON / HEALING: risk appetite improving"
else:
    alert = "⚠️ TRANSITION: mixed signals (watch confirmations)"

# ---------------------------
# LAYOUT
# ---------------------------
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.subheader("🧠 Composite Regime")
    st.metric("Regime", regime, f"{composite:.2f}")
    st.write(f"**Alert:** {alert}")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)")

with col2:
    st.subheader("🧪 Breadth")
    if use_breadth and breadth_metrics:
        st.metric("% above 50DMA", f"{breadth_metrics['%>50DMA']:.1f}%")
        st.metric("% above 200DMA", f"{breadth_metrics['%>200DMA']:.1f}%")
        st.metric("% above 50&200", f"{breadth_metrics['%>50&200']:.1f}%")
        st.caption(f"Basket size: {breadth_metrics['n']} tickers")
        st.write(f"**Breadth score:** {breadth_score:.2f}")
    else:
        st.info("Breadth is off or not available. Toggle it on in the sidebar.")

with col3:
    st.subheader("🧩 What the score means")
    st.write(
        """
- **MACD sign + TSI sign** = core regime (confirmed over N closes)
- **Canary Stoch** adds early turning points (oversold/overbought)
- **Breadth** = structural health (% above MAs)
- Stress ratio (**SPXS:SVOL**) is **inverted**: rising stress hurts the composite
        """.strip()
    )

st.divider()

st.subheader("📦 Component Canary Table (MACD/TSI/Stoch)")
st.dataframe(comp_df, use_container_width=True)

# ---------------------------
# HISTORY: COMPOSITE + COMPONENTS
# ---------------------------
st.subheader("📈 Composite History (approx, last ~250 trading days)")
st.caption("This is a lightweight history using component scores; not a price chart.")

# Build score histories for each component (fast enough for ratios)
hist_index = closes_ratio.index
hist = pd.DataFrame(index=hist_index)

for c in COMPONENTS:
    a, b = c["ticker_a"], c["ticker_b"]
    if a not in closes_ratio.columns or b not in closes_ratio.columns:
        continue
    ratio = safe_ratio(closes_ratio[a], closes_ratio[b])
    # Build a simple daily score history using rolling recalculation
    # For speed: we compute MACD+TSI sign daily and stoch/cci zones daily
    s = ratio.dropna()
    if len(s) < 250:
        continue

    macd = ta.macd(s, fast=24, slow=52, signal=18).iloc[:, 0]
    tsi = ta.tsi(s, long=tsi_long, short=tsi_short, signal=tsi_signal)
    tsi_line = tsi.iloc[:, 0] if isinstance(tsi, pd.DataFrame) else tsi
    stoch_k = canary_stoch(s, length=stoch_len)
    cci = proxy_cci(s, length=100)

    # confirmed signs (rolling)
    macd_sign = macd.rolling(confirm_days).apply(lambda x: 1 if (x > 0).all() else (-1 if (x < 0).all() else 0), raw=False)
    tsi_sign = tsi_line.rolling(confirm_days).apply(lambda x: 1 if (x > 0).all() else (-1 if (x < 0).all() else 0), raw=False)

    stoch_zone = pd.Series(0.0, index=s.index)
    stoch_zone[stoch_k <= 20] = +0.5
    stoch_zone[stoch_k >= 80] = -0.5

    cci_zone = pd.Series(0.0, index=s.index)
    cci_zone[cci >= 100] = +0.5
    cci_zone[cci <= -100] = -0.5

    score = 0.4 * macd_sign + 0.4 * tsi_sign + 0.1 * stoch_zone + 0.1 * cci_zone
    if c["invert"]:
        score = -score

    hist[c["name"]] = score.reindex(hist_index)

# Composite history
hist["Composite"] = 0.0
for c in COMPONENTS:
    if c["name"] in hist.columns:
        hist["Composite"] += hist[c["name"]] * c["weight"]

# Breadth history (optional, expensive) -> keep it current only; not history
# To keep robust and fast on cloud, we don't recompute full breadth history here.

hist_plot = hist[["Composite"]].dropna().tail(250)
st.line_chart(hist_plot)

with st.expander("Show recent composite values"):
    st.dataframe(hist_plot.tail(60), use_container_width=True)
