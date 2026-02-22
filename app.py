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
st.title("🧭 Robust Market Regime Dashboard (5y + Predictiveness Tools)")
st.caption("Composite regime score from ratio canaries + confirmation indicators, with SPY overlay + peak markers + next-20d drawdown probability")

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
with st.sidebar:
    st.header("Settings")

    period = st.selectbox("History window", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Interval", ["1d"], index=0)

    st.subheader("Indicators")
    stoch_len = st.slider("Canary Stoch length", 10, 60, 14)
    tsi_long = st.slider("TSI long", 10, 80, 40)
    tsi_short = st.slider("TSI short", 5, 40, 20)
    tsi_signal = st.slider("TSI signal", 3, 30, 10)

    st.subheader("Confirmations")
    confirm_days = st.slider("Confirm MACD/TSI sign (days)", 1, 5, 2)
    trend_lookback = st.slider("Composite trend lookback (days)", 5, 30, 10)

    st.subheader("Predictiveness")
    fwd_days = st.slider("Forward window (days) for drawdown probability", 10, 60, 20)
    dd_threshold = st.slider("Define 'correction' as SPY drawdown ≤", -15.0, -2.0, -5.0, 0.5)
    state_bins = st.slider("State bins (composite quantiles)", 3, 10, 5)

    st.subheader("Peak markers")
    peak_window = st.slider("Local peak window (days)", 5, 30, 10)
    peak_min_move = st.slider("Min drop after peak to count (%)", 0.5, 5.0, 1.5, 0.5)

    st.subheader("Breadth (optional)")
    use_breadth = st.toggle("Compute breadth (slower)", value=True)
    breadth_max_tickers = st.slider("Breadth basket size", 20, 140, 60, step=10)

    st.subheader("Scoring weights")
    w_stress = st.slider("Stress (SPXS:SVOL)", 0.05, 0.60, 0.25, 0.05)
    w_credit = st.slider("Credit (HYG:SHY)", 0.05, 0.60, 0.25, 0.05)
    w_lead = st.slider("Leadership (SOXX:SPY)", 0.05, 0.60, 0.20, 0.05)
    w_fin = st.slider("Financials (XLF:SPY)", 0.05, 0.60, 0.20, 0.05)
    w_house = st.slider("Housing (ITB:SPY)", 0.00, 0.40, 0.10, 0.05)
    w_breadth = st.slider("Breadth weight", 0.00, 0.60, 0.20, 0.05)

    st.subheader("Alert thresholds")
    risk_off_th = st.slider("Risk-Off if Composite ≤", -1.0, 0.0, -0.35, 0.05)
    risk_on_th = st.slider("Risk-On if Composite ≥", 0.0, 1.0, 0.35, 0.05)

# Normalize weights
weights = np.array([w_stress, w_credit, w_lead, w_fin, w_house, w_breadth], dtype=float)
wsum = weights.sum()
if wsum == 0:
    st.error("All weights are zero. Increase at least one weight.")
    st.stop()
weights = weights / wsum
w_stress, w_credit, w_lead, w_fin, w_house, w_breadth = weights.tolist()

# ---------------------------
# COMPONENT DEFINITIONS
# invert=True => rising ratio = more stress (bad), so score is negated
# ---------------------------
COMPONENTS = [
    {"name": "Stress vs Carry", "a": "SPXS", "b": "SVOL", "invert": True,  "weight": w_stress},
    {"name": "Credit Gate",     "a": "HYG",  "b": "SHY",  "invert": False, "weight": w_credit},
    {"name": "Semis Lead",      "a": "SOXX", "b": "SPY",  "invert": False, "weight": w_lead},
    {"name": "Financials Lead", "a": "XLF",  "b": "SPY",  "invert": False, "weight": w_fin},
    {"name": "Housing/Cyclic",  "a": "ITB",  "b": "SPY",  "invert": False, "weight": w_house},
]

ratio_tickers = sorted({c["a"] for c in COMPONENTS} | {c["b"] for c in COMPONENTS} | {"SPY"})

# Breadth basket (liquid proxies; adjustable size)
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
    def f(arr):
        if np.all(arr > 0):
            return 1.0
        if np.all(arr < 0):
            return -1.0
        return 0.0
    return x.rolling(days).apply(f, raw=True)

def regime_label(level: float, trend: float) -> str:
    # Level buckets
    if level <= -0.50:
        lvl = "Risk-Off"
    elif level <= -0.15:
        lvl = "Deteriorating"
    elif level < 0.15:
        lvl = "Transition"
    elif level < 0.50:
        lvl = "Healing"
    else:
        lvl = "Risk-On"

    improving = trend >= 0

    if lvl in ["Healing", "Risk-On"] and improving:
        return f"✅ {lvl} + improving"
    if lvl in ["Healing", "Risk-On"] and not improving:
        return f"⚠️ {lvl} but deteriorating"
    if lvl in ["Risk-Off", "Deteriorating"] and improving:
        return f"✅ {lvl} but improving (bottoming)"
    if lvl in ["Risk-Off", "Deteriorating"] and not improving:
        return f"🚨 {lvl} and deteriorating"
    # Transition:
    return f"🟨 {lvl} (watch trend)" if improving else f"⚠️ {lvl} (weakening)"

def rebase_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)
    return (100 * (s / s.iloc[0])).reindex(series.index)

def z_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty or s.std() == 0:
        return pd.Series(index=series.index, dtype=float)
    z = (s - s.mean()) / s.std()
    return (100 + 10 * z).reindex(series.index)

def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

def canary_dial(score: float) -> int:
    # Map composite (-1..+1) to 0..100, clipped
    score = clamp(score, -1.0, 1.0)
    return int(round((score + 1) * 50))

def forward_min_return(series: pd.Series, fwd: int) -> pd.Series:
    # For each day t: min(close[t+1..t+fwd]) / close[t] - 1
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    arr = s.values
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        j = min(len(arr), i + fwd + 1)
        if i + 1 >= j:
            continue
        mn = np.nanmin(arr[i+1:j])
        out[i] = (mn / arr[i]) - 1.0
    return pd.Series(out, index=s.index)

def local_peaks(series: pd.Series, window: int = 10, min_drop_pct: float = 1.5) -> pd.Series:
    # Peak if it's the max in +/- window and the subsequent window sees at least min_drop_pct drawdown
    s = series.dropna()
    if len(s) < (2 * window + 5):
        return pd.Series(False, index=series.index)
    peaks = pd.Series(False, index=s.index)

    for i in range(window, len(s) - window):
        seg = s.iloc[i-window:i+window+1]
        if s.iloc[i] != seg.max():
            continue
        # require a drop after peak within next window
        future = s.iloc[i+1:i+window+1]
        if future.empty:
            continue
        dd = (future.min() / s.iloc[i] - 1.0) * 100.0
        if dd <= -min_drop_pct:
            peaks.iloc[i] = True
    return peaks.reindex(series.index, fill_value=False)

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
hist = pd.DataFrame(index=closes_ratio.index)
component_latest = []

for c in COMPONENTS:
    a, b = c["a"], c["b"]
    if a not in closes_ratio.columns or b not in closes_ratio.columns:
        continue

    ratio = safe_ratio(closes_ratio[a], closes_ratio[b])
    if ratio.empty or len(ratio) < 250:
        continue

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

    # Latest detail snapshot
    last_score = float(score.dropna().iloc[-1]) if score.dropna().shape[0] else np.nan
    component_latest.append({
        "Component": c["name"],
        "Ratio": f"{a}:{b}",
        "Weight": round(c["weight"], 3),
        "Latest Score": round(last_score, 3) if pd.notna(last_score) else np.nan,
        "Weighted": round(last_score * c["weight"], 3) if pd.notna(last_score) else np.nan
    })

# Composite (ratios-only history)
hist["Composite (ratios)"] = 0.0
for c in COMPONENTS:
    if c["name"] in hist.columns:
        hist["Composite (ratios)"] += hist[c["name"]] * c["weight"]

composite_series = hist["Composite (ratios)"].dropna()

# ---------------------------
# OPTIONAL: LATEST BREADTH (single-point, not full history)
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
            return float(np.clip((p - 50) / 10, -1, 1))

        breadth_score = 0.4 * pct_to_score(above200) + 0.4 * pct_to_score(above50) + 0.2 * pct_to_score(above_both)
        breadth_metrics = {"%>50DMA": above50, "%>200DMA": above200, "%>50&200": above_both, "n": int(closes_b.shape[1])}

# Latest composite incl breadth (single point)
latest_level = float(composite_series.iloc[-1]) if not composite_series.empty else np.nan
if pd.notna(breadth_score):
    latest_level = float(latest_level + breadth_score * w_breadth)

# Trend metric (based on ratios-only history to keep it stable)
trend_series = composite_series - composite_series.shift(trend_lookback)
latest_trend = float(trend_series.dropna().iloc[-1]) if trend_series.dropna().shape[0] else np.nan

label = regime_label(latest_level, latest_trend) if pd.notna(latest_level) and pd.notna(latest_trend) else "—"

alert = "—"
if pd.notna(latest_level):
    if latest_level <= risk_off_th:
        alert = "🚨 RISK-OFF ALERT: defensive posture favored"
    elif latest_level >= risk_on_th:
        alert = "✅ RISK-ON / HEALING: risk appetite improving"
    else:
        alert = "⚠️ TRANSITION: mixed signals (watch confirmations)"

# ---------------------------
# PROBABILITY: SPY ≥5% drawdown within next N days (state-conditioned)
# ---------------------------
spy_close = download_closes(["SPY"], period=period, interval=interval)["SPY"].dropna()
spy_fwd_min = forward_min_return(spy_close, fwd=fwd_days) * 100.0  # in %
event = (spy_fwd_min <= dd_threshold)  # True if future min drawdown <= threshold

# Create historical "state" from composite level + trend sign
if not composite_series.empty:
    comp_hist = composite_series.reindex(spy_close.index).dropna()
    trend_hist = (comp_hist - comp_hist.shift(trend_lookback)).dropna()
    aligned = pd.DataFrame({
        "comp": comp_hist,
        "trend": trend_hist,
        "event": event.reindex(comp_hist.index)
    }).dropna()

    if aligned.empty or aligned["event"].isna().all():
        prob_next = np.nan
        prob_note = "Not enough aligned history to compute probability."
    else:
        # Quantile bin for composite level
        try:
            aligned["bin"] = pd.qcut(aligned["comp"], q=state_bins, duplicates="drop")
        except Exception:
            aligned["bin"] = pd.cut(aligned["comp"], bins=state_bins)

        aligned["trend_sign"] = np.where(aligned["trend"] >= 0, "up", "down")

        # Current state
        cur_comp = float(comp_hist.iloc[-1])
        cur_trend_sign = "up" if latest_trend >= 0 else "down"

        # Find matching bin for current level
        cur_bin = None
        for b in aligned["bin"].cat.categories if hasattr(aligned["bin"], "cat") else sorted(aligned["bin"].dropna().unique()):
            if pd.notna(b):
                if hasattr(b, "left") and hasattr(b, "right"):
                    if (cur_comp >= b.left) and (cur_comp <= b.right):
                        cur_bin = b
                        break
        if cur_bin is None:
            # fallback: nearest by comp percentile
            cur_bin = aligned["bin"].iloc[-1]

        subset = aligned[(aligned["bin"] == cur_bin) & (aligned["trend_sign"] == cur_trend_sign)]
        if subset.shape[0] < 30:
            # broaden: ignore trend_sign if too few samples
            subset = aligned[(aligned["bin"] == cur_bin)]

        prob_next = float(subset["event"].mean()) if subset.shape[0] else np.nan
        prob_note = f"State samples used: {subset.shape[0]} (bin + trend condition; broadened if sparse)"
else:
    prob_next = np.nan
    prob_note = "No composite history."

# ---------------------------
# CANARY DIAL
# ---------------------------
dial_value = canary_dial(latest_level) if pd.notna(latest_level) else 50

# ---------------------------
# PEAK MARKERS (overlay)
# ---------------------------
spy_100 = rebase_100(spy_close)
comp_100 = z_to_100(composite_series)

spy_peaks = local_peaks(spy_100, window=peak_window, min_drop_pct=peak_min_move)
comp_peaks = local_peaks(comp_100, window=peak_window, min_drop_pct=peak_min_move)

# ---------------------------
# TOP SUMMARY
# ---------------------------
c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.2])

with c1:
    st.subheader("🧠 Composite Level + Trend")
    st.metric("Level (latest)", f"{latest_level:.2f}" if pd.notna(latest_level) else "—")
    st.metric(f"Trend ({trend_lookback}d change)", f"{latest_trend:+.2f}" if pd.notna(latest_trend) else "—")
    st.write(f"**Label:** {label}")
    st.write(f"**Alert:** {alert}")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with c2:
    st.subheader("🎛️ Canary Score Dial")
    st.progress(dial_value / 100.0, text=f"{dial_value}/100 (0=risk-off, 100=risk-on)")
    st.caption("This is a stabilized single-number proxy for regime. Use trend + components for timing.")

with c3:
    st.subheader("📉 Next move probability")
    if pd.notna(prob_next):
        st.metric(f"P(SPY drawdown ≤ {dd_threshold:.1f}% in next {fwd_days}d)", f"{prob_next*100:.1f}%")
    else:
        st.metric("Probability", "—")
    st.caption(prob_note)

with c4:
    st.subheader("🧪 Breadth (latest)")
    if use_breadth and breadth_metrics:
        st.metric("% above 50DMA", f"{breadth_metrics['%>50DMA']:.1f}%")
        st.metric("% above 200DMA", f"{breadth_metrics['%>200DMA']:.1f}%")
        st.metric("% above 50&200", f"{breadth_metrics['%>50&200']:.1f}%")
        st.write(f"**Breadth score:** {breadth_score:.2f}")
        st.caption(f"Basket: {breadth_metrics['n']} tickers")
    else:
        st.info("Breadth off or unavailable.")

st.divider()

# ---------------------------
# COMPONENT TABLE
# ---------------------------
st.subheader("📦 Component Health (latest)")
if component_latest:
    st.dataframe(pd.DataFrame(component_latest).sort_values("Weighted", ascending=False), use_container_width=True)
else:
    st.warning("No component scores computed (check tickers/data availability).")

# ---------------------------
# CHART 1: RAW COMPOSITE HISTORY
# ---------------------------
st.subheader("📈 Composite History (raw, ratios-only)")
st.caption("Daily composite regime score built from ratio canaries (full history, not capped).")
if composite_series.empty:
    st.warning("Composite series is empty.")
else:
    st.line_chart(composite_series.to_frame("Composite (ratios)"), use_container_width=True)

# ---------------------------
# CHART 2: OVERLAY COMPOSITE vs SPY + PEAK MARKERS
# ---------------------------
st.subheader("🧪 Predictiveness overlay: Composite vs SPY (rebased) + peak markers")
st.caption("SPY rebased to 100. Composite mapped to ~100-scale (z-to-100). Markers show local peaks that preceded meaningful drops.")

df_overlay = pd.DataFrame({
    "SPY_100": spy_100,
    "Composite_100": comp_100
}).dropna()

if df_overlay.empty:
    st.warning("Not enough aligned history for overlay.")
else:
    base = df_overlay.reset_index().rename(columns={"index": "Date"})

    line = (
        alt.Chart(base)
        .transform_fold(["SPY_100", "Composite_100"], as_=["Series", "Value"])
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Comparable scale (around 100)"),
            color=alt.Color("Series:N"),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=".2f")]
        )
        .properties(height=420)
        .interactive()
    )

    # Peak points
    spy_pk_df = pd.DataFrame({
        "Date": spy_peaks[spy_peaks].index,
        "Value": spy_100.reindex(spy_peaks[spy_peaks].index).values,
        "Peak": "SPY Peak"
    })
    comp_pk_df = pd.DataFrame({
        "Date": comp_peaks[comp_peaks].index,
        "Value": comp_100.reindex(comp_peaks[comp_peaks].index).values,
        "Peak": "Composite Peak"
    })
    peaks_df = pd.concat([spy_pk_df, comp_pk_df], ignore_index=True)

    if not peaks_df.empty:
        peaks = (
            alt.Chart(peaks_df)
            .mark_point(filled=True, size=70)
            .encode(
                x="Date:T",
                y="Value:Q",
                shape=alt.Shape("Peak:N"),
                tooltip=["Date:T", "Peak:N", alt.Tooltip("Value:Q", format=".2f")]
            )
        )
        st.altair_chart(line + peaks, use_container_width=True)
    else:
        st.altair_chart(line, use_container_width=True)

with st.expander("Show overlay data (last 60 rows)"):
    st.dataframe(df_overlay.tail(60), use_container_width=True)
