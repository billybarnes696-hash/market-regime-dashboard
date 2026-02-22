import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import altair as alt

# =========================
# LOCKED SETTINGS (do not tweak)
# =========================
LOOKBACK_YEARS = 5
INTERVAL = "1d"
MIN_BARS = 260  # sanity check
TREND_LOOKBACK_DAYS = 10

# Indicator params (frozen)
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 24, 52, 18
TSI_LONG, TSI_SHORT, TSI_SIGNAL = 40, 20, 10
STOCH_LEN, STOCH_SMOOTH_K, STOCH_SMOOTH_D = 14, 3, 3

# Composite thresholds (frozen)
BUY_LEVEL_TH = 0.20
SELL_LEVEL_TH = -0.20
CONF_TH = 70

st.set_page_config(page_title="Robust Market Regime Dashboard", layout="wide")

# =========================
# HELPERS
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd_hist(series: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    m = ema(series, fast) - ema(series, slow)
    sig = ema(m, signal)
    return (m - sig)

def tsi(series: pd.Series, long_len: int, short_len: int, signal_len: int) -> tuple[pd.Series, pd.Series]:
    # True Strength Index (TSI)
    # TSI = 100 * EMA(EMA(m, short), long) / EMA(EMA(|m|, short), long)
    m = series.diff()
    ema1 = ema(m, short_len)
    ema2 = ema(ema1, long_len)

    abs_m = m.abs()
    abs_ema1 = ema(abs_m, short_len)
    abs_ema2 = ema(abs_ema1, long_len)

    tsi_val = 100 * (ema2 / abs_ema2.replace(0, np.nan))
    tsi_sig = ema(tsi_val, signal_len)
    return tsi_val, tsi_sig

def stoch_kd(series: pd.Series, length: int, smooth_k: int, smooth_d: int) -> tuple[pd.Series, pd.Series]:
    # Stochastic on the ratio itself using rolling high/low of the ratio
    low = series.rolling(length).min()
    high = series.rolling(length).max()
    k = 100 * (series - low) / (high - low).replace(0, np.nan)
    k_s = k.rolling(smooth_k).mean()
    d_s = k_s.rolling(smooth_d).mean()
    return k_s, d_s

def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd.replace(0, np.nan)

def safe_last(series: pd.Series):
    return float(series.dropna().iloc[-1]) if series.dropna().shape[0] else np.nan

@st.cache_data(ttl=60 * 30)  # 30 min cache
def fetch_prices(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    # Normalize to Close-only table
    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        close_df = pd.DataFrame(closes)
    else:
        # single ticker case
        close_df = pd.DataFrame({tickers[0]: df["Close"]})
    close_df = close_df.dropna(how="all")
    return close_df

def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)

def series_score(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Returns a per-day score in [-1, +1] using frozen indicators:
      - MACD histogram sign (24,52,18)
      - TSI sign (40,20,10)
      - Stoch %K relative to 50 (14,3,3)
    """
    s = series.dropna()
    if s.shape[0] < MIN_BARS:
        # return empty aligned series
        return pd.Series(index=series.index, dtype=float)

    mh = macd_hist(s, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    tsi_val, tsi_sig = tsi(s, TSI_LONG, TSI_SHORT, TSI_SIGNAL)
    k, d = stoch_kd(s, STOCH_LEN, STOCH_SMOOTH_K, STOCH_SMOOTH_D)

    # Convert to directional signals
    macd_dir = np.where(mh > 0, 1.0, -1.0)
    tsi_dir = np.where(tsi_val > 0, 1.0, -1.0)
    stoch_dir = np.where(k > 50, 1.0, -1.0)

    # Weighted blend (fixed)
    raw = 0.4 * macd_dir + 0.4 * tsi_dir + 0.2 * stoch_dir
    raw = pd.Series(raw, index=s.index).clip(-1, 1)

    if invert:
        raw = -raw

    # Reindex to original
    out = pd.Series(index=series.index, dtype=float)
    out.loc[raw.index] = raw
    return out

def slope_dir(series: pd.Series, lookback: int) -> pd.Series:
    return series - series.shift(lookback)

def pct_above_ma(close_df: pd.DataFrame, ma_len: int) -> pd.Series:
    ma = close_df.rolling(ma_len).mean()
    above = (close_df > ma).astype(float)
    return above.mean(axis=1) * 100.0

def nh_nl(close_df: pd.DataFrame, window: int = 252) -> pd.Series:
    rolling_high = close_df.rolling(window).max()
    rolling_low = close_df.rolling(window).min()
    nh = (close_df >= rolling_high).sum(axis=1)
    nl = (close_df <= rolling_low).sum(axis=1)
    return (nh - nl).astype(float)

def adv_decl_line(close_df: pd.DataFrame) -> pd.Series:
    adv = (close_df.diff() > 0).sum(axis=1)
    dec = (close_df.diff() < 0).sum(axis=1)
    return (adv - dec).cumsum()

def normalize_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series
    base = s.iloc[0]
    if base == 0 or np.isnan(base):
        return series
    out = (series / base) * 100.0
    return out

# =========================
# TICKERS / COMPONENTS (robust clusters, not individual stocks)
# =========================
CORE_TICKERS = [
    "SPY", "SPXS", "SVOL", "VXX",
    "HYG", "SHY",
    "SMH", "SOXX", "XLF", "KBE", "KRE",
    "RSP",
    # Breadth basket additions:
    "XLK", "XLY", "XLI", "XLB", "XLE",
    "IWM", "MDY", "IJR",
    "ITB", "XHB",
    "EEM", "EWG", "EWY",
    "XOP", "OIH", "GDX",
]

# Breadth basket (approx breadth proxy using liquid ETFs & large caps)
BREADTH_BASKET = [
    "SPY","RSP","IWM","MDY","IJR",
    "XLK","XLF","XLY","XLP","XLV","XLI","XLE","XLB","XLU",
    "SMH","SOXX","XBI","IBB","XRT","ITB","XHB",
    "EEM","EWG","EWJ","EWY","FXI","EWZ",
    "GDX","SLV","GLD","USO","XOP","OIH",
    "AAPL","MSFT","AMZN","GOOGL","NVDA","AMD","META","TSLA",
    "JPM","BAC","WFC","C","GS","MS",
    "CAT","DE","FDX","DIS","NKE","MA",
    "INTC","AMAT","MU","NFLX","SBUX"
]

# Deduplicate
CORE_TICKERS = sorted(list(set(CORE_TICKERS)))
BREADTH_BASKET = sorted(list(set(BREADTH_BASKET)))

# Composite components: (name -> config)
# invert=True means "higher = worse" (risk-off)
COMPONENTS = {
    # Stress / carry
    "SPXS:SVOL (Stress/Carry)": {"type": "ratio", "a": "SPXS", "b": "SVOL", "invert": False, "weight": 0.22},

    # Vol confirmation (late confirmation)
    "SPY:VXX (Vol Confirm)": {"type": "ratio", "a": "SPY", "b": "VXX", "invert": True, "weight": 0.10},

    # Credit gate
    "HYG:SHY (Credit)": {"type": "ratio", "a": "HYG", "b": "SHY", "invert": False, "weight": 0.18},

    # Leadership clusters
    "SMH:SPY (Semis Leadership)": {"type": "ratio", "a": "SMH", "b": "SPY", "invert": False, "weight": 0.15},
    "XLF:SPY (Financials Leadership)": {"type": "ratio", "a": "XLF", "b": "SPY", "invert": False, "weight": 0.10},

    # Concentration / equal-weight
    "RSP:SPY (Equal-Weight Breadth)": {"type": "ratio", "a": "RSP", "b": "SPY", "invert": False, "weight": 0.10},

    # Cyclical context (light weight, supportive)
    "IWM:SPY (Small Caps)": {"type": "ratio", "a": "IWM", "b": "SPY", "invert": False, "weight": 0.08},
    "XLY:SPY (Discretionary)": {"type": "ratio", "a": "XLY", "b": "SPY", "invert": False, "weight": 0.07},
}

# Normalize weights to sum 1
w_sum = sum(cfg["weight"] for cfg in COMPONENTS.values())
for k in COMPONENTS:
    COMPONENTS[k]["weight"] = COMPONENTS[k]["weight"] / w_sum

# =========================
# UI (locked)
# =========================
st.title("Robust Market Regime Dashboard (Locked Model)")
st.caption(
    "Daily | 5y | MACD(24,52,18) + TSI(40,20,10) + Stoch(14,3,3) | Composite + Confidence + Action"
)

# =========================
# LOAD DATA
# =========================
period = f"{LOOKBACK_YEARS}y"
tickers_needed = sorted(list(set(CORE_TICKERS + BREADTH_BASKET)))

with st.spinner("Downloading market data (yfinance)..."):
    close_df = fetch_prices(tickers_needed, period=period, interval=INTERVAL)

if close_df.shape[0] < MIN_BARS or "SPY" not in close_df.columns:
    st.error("Not enough data loaded or SPY missing. Try again later.")
    st.stop()

# Align & forward-fill small gaps
close_df = close_df.sort_index().ffill().dropna(how="all")

# =========================
# BUILD COMPONENT SERIES + SCORES
# =========================
component_rows = []
score_series_map = {}
latest_weighted_sum = 0.0
weight_total = 0.0

for name, cfg in COMPONENTS.items():
    a = cfg["a"]
    b = cfg["b"]

    if a not in close_df.columns or b not in close_df.columns:
        continue

    s = ratio(close_df[a], close_df[b]).dropna()
    sc = series_score(s, invert=cfg["invert"])
    sc = sc.reindex(close_df.index)

    # Weighted contribution (latest)
    latest_sc = safe_last(sc)
    weighted = latest_sc * cfg["weight"] if not np.isnan(latest_sc) else np.nan

    score_series_map[name] = sc
    component_rows.append({
        "Component": name,
        "Weight": round(cfg["weight"], 3),
        "Latest Score (-1..+1)": None if np.isnan(latest_sc) else round(latest_sc, 2),
        "Weighted": None if np.isnan(weighted) else round(weighted, 2),
    })

# Composite score as weighted average of component scores
scores_df = pd.DataFrame(score_series_map)
# Weighted composite time series
weights = pd.Series({k: COMPONENTS[k]["weight"] for k in score_series_map.keys()})
composite = (scores_df * weights).sum(axis=1) / weights.sum()
composite = composite.clip(-1, 1)

composite_level = safe_last(composite)
composite_trend = safe_last(composite - composite.shift(TREND_LOOKBACK_DAYS))

# =========================
# BREADTH BLOCK (approximate breadth using your basket)
# =========================
basket_cols = [c for c in BREADTH_BASKET if c in close_df.columns]
basket = close_df[basket_cols].dropna(how="all").ffill()

pct_50 = pct_above_ma(basket, 50)
pct_150 = pct_above_ma(basket, 150)
pct_200 = pct_above_ma(basket, 200)
adl = adv_decl_line(basket)
nhnl = nh_nl(basket, 252)

# Breadth confirmation: improving over 10d
breadth_flags = {
    "%>50DMA improving": safe_last(pct_50 - pct_50.shift(TREND_LOOKBACK_DAYS)) > 0,
    "%>150DMA improving": safe_last(pct_150 - pct_150.shift(TREND_LOOKBACK_DAYS)) > 0,
    "%>200DMA improving": safe_last(pct_200 - pct_200.shift(TREND_LOOKBACK_DAYS)) > 0,
    "A/D Line improving": safe_last(adl - adl.shift(TREND_LOOKBACK_DAYS)) > 0,
    "NH-NL improving": safe_last(nhnl - nhnl.shift(TREND_LOOKBACK_DAYS)) > 0,
}
breadth_confirm = 100.0 * (sum(bool(v) for v in breadth_flags.values()) / len(breadth_flags))

# =========================
# CONFIDENCE SCORE (agreement + breadth + trend stability)
# =========================
# Agreement: component direction matches composite direction
comp_dir = 1 if composite_level > 0 else -1
agree = 0
total = 0
trend_agree = 0
trend_total = 0

for nm, sc in score_series_map.items():
    latest = safe_last(sc)
    if np.isnan(latest):
        continue
    total += 1
    if (latest > 0 and comp_dir == 1) or (latest < 0 and comp_dir == -1):
        agree += 1

    # Trend stability: component improving in same direction as composite trend sign
    d = safe_last(sc - sc.shift(TREND_LOOKBACK_DAYS))
    if not np.isnan(d):
        trend_total += 1
        if (composite_trend >= 0 and d >= 0) or (composite_trend < 0 and d < 0):
            trend_agree += 1

agreement_score = 100.0 * (agree / total) if total else np.nan
trend_stability = 100.0 * (trend_agree / trend_total) if trend_total else np.nan

# Final confidence (fixed weights)
confidence = 0.4 * agreement_score + 0.3 * breadth_confirm + 0.3 * trend_stability
confidence_score = float(confidence) if not np.isnan(confidence) else np.nan

# =========================
# GATES for ACTION (credit + stress)
# =========================
def gate_supportive(series: pd.Series) -> bool:
    s = series.dropna()
    if s.shape[0] < MIN_BARS:
        return False
    mh = macd_hist(s, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    t, _ = tsi(s, TSI_LONG, TSI_SHORT, TSI_SIGNAL)
    return (safe_last(mh) > 0) and (safe_last(t) > 0)

# Credit gate: HYG/SHY supportive?
credit_series = ratio(close_df["HYG"], close_df["SHY"])
credit_gate = gate_supportive(credit_series)

# Stress gate: SPXS/SVOL NOT expanding? (we want stress to be weakening, so supportive means inverted)
stress_series = ratio(close_df["SPXS"], close_df["SVOL"])
# supportive when MACD hist < 0 and TSI < 0 (stress fading)
stress_gate = (safe_last(macd_hist(stress_series.dropna(), MACD_FAST, MACD_SLOW, MACD_SIGNAL)) < 0) and \
              (safe_last(tsi(stress_series.dropna(), TSI_LONG, TSI_SHORT, TSI_SIGNAL)[0]) < 0)

# =========================
# ACTION SIGNAL
# =========================
action = "HOLD / NEUTRAL"
action_color = "gray"

if (composite_level > BUY_LEVEL_TH and composite_trend > 0 and confidence_score >= CONF_TH
    and credit_gate and stress_gate):
    action = "BUY (Long SPY)"
    action_color = "green"

elif (composite_level < SELL_LEVEL_TH and composite_trend < 0 and confidence_score >= CONF_TH
      and (not stress_gate) and (not credit_gate)):
    action = "SELL / SHORT (Short SPY)"
    action_color = "red"

# Regime label (your 4-state grid)
label = "Transition / Mixed"
emoji = "⚪"

if composite_level >= 0:
    if composite_trend >= 0:
        label, emoji = "Healing + improving", "✅"
    else:
        label, emoji = "Healing but deteriorating", "⚠️"
else:
    if composite_trend >= 0:
        label, emoji = "Risk-off but improving (bottoming)", "✅"
    else:
        label, emoji = "Risk-off and deteriorating", "🚨"

# =========================
# DISPLAY TOP METRICS
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Composite Level (latest)", f"{composite_level:.2f}")
c2.metric("Composite Trend (10d Δ)", f"{composite_trend:.2f}")
c3.metric("Confidence Score", f"{confidence_score:.0f}")
c4.metric("Regime Label", f"{label} {emoji}")

st.markdown(
    f"<div style='padding:14px;border-radius:12px;border:1px solid #ddd;'>"
    f"<span style='font-size:22px;font-weight:800;color:{action_color};'>{action}</span>"
    f"<div style='margin-top:6px;color:#666;'>"
    f"Credit gate={'ON' if credit_gate else 'OFF'} | Stress gate={'ON' if stress_gate else 'OFF'}"
    f"</div></div>",
    unsafe_allow_html=True
)

# =========================
# CHART: Composite + SPY overlay (normalized)
# =========================
spy_norm = normalize_to_100(close_df["SPY"])
comp_norm = normalize_to_100(composite + 2.0)  # shift up to avoid negative values; then normalize

plot_df = pd.DataFrame({
    "Date": close_df.index,
    "SPY (Normalized)": spy_norm.values,
    "Composite (Normalized)": comp_norm.reindex(close_df.index).values,
}).dropna()

base = alt.Chart(plot_df).encode(x="Date:T")

line_spy = base.mark_line().encode(y=alt.Y("SPY (Normalized):Q", title="Normalized (base=100)"))
line_comp = base.mark_line(strokeDash=[6, 3]).encode(y=alt.Y("Composite (Normalized):Q"))

st.subheader("5-Year Overlay: Composite vs SPY (Normalized)")
st.altair_chart((line_spy + line_comp).interactive(), use_container_width=True)

# =========================
# COMPONENT TABLE
# =========================
st.subheader("Components (Latest Contribution)")
comp_table = pd.DataFrame(component_rows).sort_values("Weight", ascending=False)
st.dataframe(comp_table, use_container_width=True)

# =========================
# BREADTH PANEL
# =========================
st.subheader("Breadth Health (Basket Proxies)")
b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("% > 50DMA", f"{safe_last(pct_50):.0f}%")
b2.metric("% > 150DMA", f"{safe_last(pct_150):.0f}%")
b3.metric("% > 200DMA", f"{safe_last(pct_200):.0f}%")
b4.metric("NH − NL (52w)", f"{safe_last(nhnl):.0f}")
b5.metric("Breadth Confirm", f"{breadth_confirm:.0f}")

flags_df = pd.DataFrame({
    "Breadth Check": list(breadth_flags.keys()),
    "Improving (10d)": [bool(v) for v in breadth_flags.values()]
})
st.dataframe(flags_df, use_container_width=True)

# =========================
# NOTES
# =========================
with st.expander("What this model is (and is not)"):
    st.write(
        """
        - This is a *locked* regime model: it does not optimize parameters.
        - It blends stress, credit, leadership, and breadth into a single composite score.
        - Confidence measures agreement + breadth confirmation + stability.
        - Action (BUY/SELL/HOLD) is gated to reduce whipsaws.
        """
    )
