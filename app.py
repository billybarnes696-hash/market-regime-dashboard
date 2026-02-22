import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Market Regime Dashboard", layout="wide")
st.title("📊 Market Regime Dashboard")
st.caption("Composite regime score based on ratios: stress, credit, leadership, financials, cyclicals")

# ---------------- DATA HELPERS ----------------
@st.cache_data(ttl=3600)
def get_close(symbol: str, period: str = "2y") -> pd.Series:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].dropna()
    s.name = symbol
    return s

@st.cache_data(ttl=3600)
def get_ratio(a: str, b: str, period: str = "2y") -> pd.Series:
    s1 = get_close(a, period)
    s2 = get_close(b, period)
    df = pd.concat([s1, s2], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    r = (df.iloc[:, 0] / df.iloc[:, 1]).dropna()
    r.name = f"{a}:{b}"
    return r

# ---------------- INDICATORS (ratio-safe) ----------------
def proxy_cci(x: pd.Series, length: int = 100) -> pd.Series:
    """
    CCI-like oscillator for a single series (ratios, spreads, indexes).
    Uses standard CCI formula where x acts like 'typical price'.
    """
    x = x.dropna()
    sma = x.rolling(length).mean()
    mad = (x - sma).abs().rolling(length).mean()
    cci = (x - sma) / (0.015 * mad)
    return cci

def score_series(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Score in {-1, -0.5, 0, +0.5, +1} from MACD(24,52,18) + Proxy CCI(100)
    Works on a single ratio series.
    """
    series = series.dropna()
    if len(series) < 120:
        return pd.Series(dtype=float)

    macd = ta.macd(series, fast=24, slow=52, signal=18)
    cci = proxy_cci(series, length=100)

    sc = pd.Series(0.0, index=series.index)

    # Align
    macd_line = macd.iloc[:, 0].reindex(sc.index)
    cci = cci.reindex(sc.index)

    # Scoring rules
    sc[(macd_line > 0) & (cci > 0)] = 0.5
    sc[(macd_line > 0) & (cci > 100)] = 1.0
    sc[(macd_line < 0) & (cci < 0)] = -0.5
    sc[(macd_line < 0) & (cci < -100)] = -1.0

    if invert:
        sc *= -1

    return sc.dropna()

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

# ---------------- COMPONENT DEFINITIONS ----------------
# NOTE: invert=True means "higher ratio = worse" (stress).
components = {
    "Stress vs Carry (SPXS:SVOL)": {"series": get_ratio("SPXS", "SVOL"), "invert": True,  "weight": 0.25},
    "Credit Gate (HYG:SHY)":       {"series": get_ratio("HYG",  "SHY"),  "invert": False, "weight": 0.25},
    "Semis Leadership (SOXX:SPY)": {"series": get_ratio("SOXX", "SPY"),  "invert": False, "weight": 0.20},
    "Financials Lead (XLF:SPY)":   {"series": get_ratio("XLF",  "SPY"),  "invert": False, "weight": 0.20},
    "Housing/Cyclicals (ITB:SPY)": {"series": get_ratio("ITB",  "SPY"),  "invert": False, "weight": 0.10},
}

# ---------------- BUILD DASHBOARD ----------------
rows = []
composite = 0.0
any_missing = False

for name, cfg in components.items():
    s = cfg["series"]
    if s is None or s.empty:
        any_missing = True
        rows.append([name, np.nan, cfg["weight"], np.nan, "⚠️ Data missing"])
        continue

    sc = score_series(s, invert=cfg["invert"])
    if sc.empty:
        any_missing = True
        rows.append([name, np.nan, cfg["weight"], np.nan, "⚠️ Not enough history"])
        continue

    latest = float(sc.iloc[-1])
    contrib = latest * cfg["weight"]
    composite += contrib

    rows.append([name, latest, cfg["weight"], contrib, label_regime(latest)])

df = pd.DataFrame(rows, columns=["Component", "Score (-1..1)", "Weight", "Weighted Contribution", "State"])

# ---------------- TOP SUMMARY ----------------
st.subheader("🧠 Composite Regime Score")
st.metric("Current Regime", label_regime(composite), f"{composite:.2f}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local time)")

if any_missing:
    st.warning("One or more components have missing/insufficient data. The composite may be incomplete until data loads.")

st.subheader("📦 Component Breakdown")
st.dataframe(df, use_container_width=True)

# ---------------- HISTORY ----------------
st.subheader("📈 Composite Regime Score (Daily History)")

# Create a unified history index from available series
valid_series = [cfg["series"] for cfg in components.values() if cfg["series"] is not None and not cfg["series"].empty]
if len(valid_series) == 0:
    st.error("No data could be downloaded from Yahoo Finance right now.")
    st.stop()

hist_index = valid_series[0].index
for s in valid_series[1:]:
    hist_index = hist_index.union(s.index)

hist = pd.DataFrame(index=hist_index).sort_index()

for name, cfg in components.items():
    s = cfg["series"]
    if s is None or s.empty:
        continue
    sc = score_series(s, invert=cfg["invert"])
    if sc.empty:
        continue
    hist[name] = sc.reindex(hist.index)

# Composite history
hist["Composite"] = 0.0
for name, cfg in components.items():
    if name in hist.columns:
        hist["Composite"] += hist[name] * cfg["weight"]

hist = hist.dropna(subset=["Composite"])

st.line_chart(hist["Composite"])

with st.expander("Show recent history table"):
    st.dataframe(hist.tail(60), use_container_width=True)
