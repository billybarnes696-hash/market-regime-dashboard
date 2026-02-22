import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Market Regime Dashboard",
    layout="wide"
)

st.title("📊 Market Regime Dashboard")
st.caption("Composite regime score based on stress, credit, leadership, financials, and breadth proxies")

# ---------------- DATA HELPERS ----------------
@st.cache_data(ttl=3600)
def get_price(symbol, period="2y"):
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    return df["Close"].dropna()

@st.cache_data(ttl=3600)
def get_ratio(a, b, period="2y"):
    s1 = get_price(a, period)
    s2 = get_price(b, period)
    df = pd.concat([s1, s2], axis=1).dropna()
    return df.iloc[:,0] / df.iloc[:,1]

def score_series(series, invert=False):
    macd = ta.macd(series, fast=24, slow=52, signal=18)
    cci = ta.cci(series, length=100)

    sc = pd.Series(0.0, index=series.index)
    sc[(macd.iloc[:,0] > 0) & (cci > 0)] = 0.5
    sc[(macd.iloc[:,0] > 0) & (cci > 100)] = 1.0
    sc[(macd.iloc[:,0] < 0) & (cci < 0)] = -0.5
    sc[(macd.iloc[:,0] < 0) & (cci < -100)] = -1.0

    if invert:
        sc *= -1

    return sc.dropna()

def label_state(score):
    if score <= -0.5:
        return "🟥 Risk-Off"
    elif score <= -0.15:
        return "🟧 Deteriorating"
    elif score < 0.15:
        return "🟨 Transition"
    elif score < 0.5:
        return "🟩 Healing"
    else:
        return "🟦 Risk-On"

# ---------------- COMPONENT DEFINITIONS ----------------
components = {
    "Stress vs Carry (SPXS:SVOL)": {
        "series": get_ratio("SPXS", "SVOL"),
        "invert": True,
        "weight": 0.25
    },
    "Credit Gate (HYG:SHY)": {
        "series": get_ratio("HYG", "SHY"),
        "invert": False,
        "weight": 0.25
    },
    "Semis Leadership (SOXX:SPY)": {
        "series": get_ratio("SOXX", "SPY"),
        "invert": False,
        "weight": 0.20
    },
    "Financials Lead (XLF:SPY)": {
        "series": get_ratio("XLF", "SPY"),
        "invert": False,
        "weight": 0.20
    },
    "Housing / Cyclicals (ITB:SPY)": {
        "series": get_ratio("ITB", "SPY"),
        "invert": False,
        "weight": 0.10
    }
}

# ---------------- SCORE CALCULATION ----------------
rows = []
composite_score = 0.0

for name, cfg in components.items():
    sc_series = score_series(cfg["series"], invert=cfg["invert"])
    latest = sc_series.iloc[-1]
    weighted = latest * cfg["weight"]
    composite_score += weighted

    rows.append([
        name,
        latest,
        cfg["weight"],
        weighted,
        label_state(latest)
    ])

df = pd.DataFrame(
    rows,
    columns=["Component", "Score (-1..1)", "Weight", "Weighted Contribution", "State"]
)

# ---------------- DISPLAY ----------------
st.subheader("🧠 Composite Regime Score")
st.metric(
    label="Current Regime",
    value=label_state(composite_score),
    delta=f"{composite_score:.2f}"
)

st.write("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M ET"))

st.subheader("📦 Component Breakdown")
st.dataframe(df, use_container_width=True)

# ---------------- TIME SERIES ----------------
st.subheader("📈 Composite Regime Score (Daily History)")

hist = pd.DataFrame(index=list(components.values())[0]["series"].index)

for name, cfg in components.items():
    hist[name] = score_series(cfg["series"], invert=cfg["invert"])

hist["Composite"] = sum(
    hist[name] * cfg["weight"]
    for name, cfg in components.items()
)

st.line_chart(hist["Composite"])
