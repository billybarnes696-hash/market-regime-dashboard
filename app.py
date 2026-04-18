import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from defeatbeta_api import Ticker

st.set_page_config(layout="wide")

st.title("Diamond Scanner")

# -----------------------------
# Indicator Functions
# -----------------------------

def tsi(series, r, s, signal=7):

    m = series.diff()

    ema1 = m.ewm(span=r).mean()
    ema2 = ema1.ewm(span=s).mean()

    abs_m = m.abs()

    ema3 = abs_m.ewm(span=r).mean()
    ema4 = ema3.ewm(span=s).mean()

    tsi = 100 * ema2 / ema4
    sig = tsi.ewm(span=signal).mean()

    return tsi, sig


def cci(df, n=20):

    tp = (df["high"] + df["low"] + df["close"]) / 3

    ma = tp.rolling(n).mean()

    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())))

    return (tp - ma) / (0.015 * md)


def bbpct(close):

    ma = close.rolling(20).mean()

    sd = close.rolling(20).std()

    upper = ma + 2 * sd
    lower = ma - 2 * sd

    return (close - lower) / (upper - lower)


def anchored_vwap(df):

    tp = (df["high"] + df["low"] + df["close"]) / 3

    vol = df["volume"]

    return (tp * vol).cumsum() / vol.cumsum()


# -----------------------------
# Historical Data
# -----------------------------

@st.cache_data(ttl=86400)
def get_history(symbol, years=5):

    try:

        t = Ticker(symbol)

        df = t.price("daily")

        df = df.rename(columns={
            "report_date": "date"
        })

        df["date"] = pd.to_datetime(df["date"])

        df = df.set_index("date")

        df = df.rename(columns=str.lower)

        return df.tail(252 * years)

    except:

        st.warning(f"Failed to load {symbol}")

        return None


# -----------------------------
# Feature Engineering
# -----------------------------

def build_features(df):

    df["TSI_424"], _ = tsi(df["close"], 4, 2)
    df["TSI_747"], _ = tsi(df["close"], 7, 4)
    df["TSI_1377"], _ = tsi(df["close"], 13, 7)

    df["CCI20"] = cci(df, 20)

    df["BBP"] = bbpct(df["close"])

    df["VWAP"] = anchored_vwap(df)

    df["EXT_VWAP"] = (df["close"] - df["VWAP"]) / df["VWAP"]

    df["SMA10"] = df["close"].rolling(10).mean()

    df["EXT_SMA10"] = (df["close"] - df["SMA10"]) / df["SMA10"]

    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)

    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    df["candle_score"] = (
        (df["upper_wick"] > df["lower_wick"]).astype(int)
        + (df["close"] < df["open"]).astype(int)
    )

    return df


# -----------------------------
# Forward Returns
# -----------------------------

def add_returns(df):

    df["ret1"] = df["close"].shift(-1) / df["close"] - 1

    df["ret2"] = df["close"].shift(-2) / df["close"] - 1

    df["ret5"] = df["close"].shift(-5) / df["close"] - 1

    return df


# -----------------------------
# Analog Search
# -----------------------------

def find_analogs(df):

    features = [
        "TSI_424",
        "TSI_747",
        "TSI_1377",
        "CCI20",
        "BBP",
        "EXT_VWAP",
        "EXT_SMA10",
        "candle_score"
    ]

    df = df.dropna()

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    current = X_scaled[-1]

    dist = cdist([current], X_scaled)[0]

    df["distance"] = dist

    return df.nsmallest(30, "distance")


# -----------------------------
# Forecast
# -----------------------------

def forecast(analogs):

    stats = {}

    for col in ["ret1","ret2","ret5"]:

        r = analogs[col]

        stats[col] = {
            "prob": (r < 0).mean(),
            "median": r.median()
        }

    confidence = min(len(analogs)/30,1)

    return stats, confidence


# -----------------------------
# Options Chain
# -----------------------------

def get_options(symbol):

    t = yf.Ticker(symbol)

    if len(t.options) == 0:
        return None

    exp = t.options[0]

    chain = t.option_chain(exp)

    puts = chain.puts

    puts["spread"] = puts["ask"] - puts["bid"]

    return puts.sort_values("volume", ascending=False).head(10)


# -----------------------------
# UI
# -----------------------------

symbols = st.text_input(
    "Tickers",
    "QQQ,SMH,NVDA,MSFT,AMZN"
)

symbols = [x.strip().upper() for x in symbols.split(",")]

rows = []

for s in symbols:

    df = get_history(s)

    if df is None:
        continue

    df = build_features(df)

    df = add_returns(df)

    analogs = find_analogs(df)

    stats, conf = forecast(analogs)

    rows.append({

        "symbol": s,

        "DipProb_1d": stats["ret1"]["prob"],

        "DipProb_2d": stats["ret2"]["prob"],

        "DipProb_5d": stats["ret5"]["prob"],

        "ExpRet_1d": stats["ret1"]["median"],

        "ExpRet_5d": stats["ret5"]["median"],

        "Confidence": conf

    })

results = pd.DataFrame(rows)

st.dataframe(results.sort_values("DipProb_1d", ascending=False))

selected = st.selectbox("Inspect ticker", results["symbol"])

if selected:

    st.subheader("Option Candidates")

    opt = get_options(selected)

    if opt is not None:
        st.dataframe(opt)
