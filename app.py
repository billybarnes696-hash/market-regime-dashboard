import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from defeatbeta import Ticker
import yfinance as yf

st.set_page_config(layout="wide")

# -----------------------------
# Indicators
# -----------------------------

def tsi(series, r, s, signal=7):
    momentum = series.diff()
    ema1 = momentum.ewm(span=r).mean()
    ema2 = ema1.ewm(span=s).mean()
    abs_mom = momentum.abs()
    ema3 = abs_mom.ewm(span=r).mean()
    ema4 = ema3.ewm(span=s).mean()
    tsi_val = 100 * ema2 / ema4
    tsi_signal = tsi_val.ewm(span=signal).mean()
    return tsi_val, tsi_signal


def cci(df, n=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * md)


def bb_pct(close, n=20):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    return (close - lower) / (upper - lower)


def anchored_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume']
    return (tp * vol).cumsum() / vol.cumsum()


# -----------------------------
# Historical Data
# -----------------------------

@st.cache_data(ttl=86400)
def get_historical(symbol, years=5):
    t = Ticker(symbol)
    df = t.price("daily")
    df = df.rename(columns={
        "report_date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index("date")
    return df.tail(252 * years)


# -----------------------------
# Feature Engineering
# -----------------------------

def compute_features(df):

    df["TSI_424"], df["TSI_424_sig"] = tsi(df["close"], 4, 2)
    df["TSI_747"], df["TSI_747_sig"] = tsi(df["close"], 7, 4)
    df["TSI_1377"], df["TSI_1377_sig"] = tsi(df["close"], 13, 7)

    df["CCI_20"] = cci(df, 20)
    df["CCI_15"] = cci(df, 15)

    df["BBP"] = bb_pct(df["close"])

    df["VWAP"] = anchored_vwap(df)
    df["EXT_VWAP"] = (df["close"] - df["VWAP"]) / df["VWAP"]

    df["SMA10"] = df["close"].rolling(10).mean()
    df["EXT_SMA10"] = (df["close"] - df["SMA10"]) / df["SMA10"]

    # Candle score
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["candle_score"] = (
        (df["upper_wick"] > df["lower_wick"]).astype(int) +
        (df["close"] < df["open"]).astype(int)
    )

    return df


# -----------------------------
# Forward returns
# -----------------------------

def add_forward_returns(df):

    df["fwd_1d"] = df["close"].shift(-1) / df["close"] - 1
    df["fwd_2d"] = df["close"].shift(-2) / df["close"] - 1
    df["fwd_5d"] = df["close"].shift(-5) / df["close"] - 1

    return df


# -----------------------------
# Analog Search
# -----------------------------

def find_analogs(df, current_row, n=30):

    features = [
        "TSI_424",
        "TSI_747",
        "TSI_1377",
        "CCI_20",
        "BBP",
        "EXT_VWAP",
        "EXT_SMA10",
        "candle_score"
    ]

    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    current_vec = scaler.transform([current_row[features]])

    dist = cdist(current_vec, X_scaled)[0]

    df_sub = df.loc[X.index].copy()
    df_sub["distance"] = dist

    analogs = df_sub.nsmallest(n, "distance")

    return analogs


# -----------------------------
# Forecast
# -----------------------------

def forecast(analogs):

    result = {}

    for horizon in ["fwd_1d", "fwd_2d", "fwd_5d"]:

        returns = analogs[horizon].dropna()

        prob = (returns < 0).mean()
        mean = returns.mean()
        median = returns.median()

        result[horizon] = {
            "prob": prob,
            "mean": mean,
            "median": median
        }

    confidence = min(len(analogs) / 30, 1)

    return result, confidence


# -----------------------------
# Options Chain
# -----------------------------

def get_options(symbol):

    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    if len(expirations) == 0:
        return None

    exp = expirations[0]
    chain = ticker.option_chain(exp)

    puts = chain.puts

    puts["spread"] = puts["ask"] - puts["bid"]

    return puts.sort_values("volume", ascending=False).head(10)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Diamond Scanner v1")

symbols = st.text_input(
    "Tickers (comma separated)",
    "QQQ,SMH,NVDA,MSFT,AMZN"
)

symbols = [s.strip().upper() for s in symbols.split(",")]

results = []

for symbol in symbols:

    df = get_historical(symbol)
    df = compute_features(df)
    df = add_forward_returns(df)

    current = df.iloc[-1]

    analogs = find_analogs(df, current)

    forecast_stats, confidence = forecast(analogs)

    results.append({
        "symbol": symbol,
        "DipProb_1d": forecast_stats["fwd_1d"]["prob"],
        "DipProb_2d": forecast_stats["fwd_2d"]["prob"],
        "DipProb_5d": forecast_stats["fwd_5d"]["prob"],
        "ExpRet_1d": forecast_stats["fwd_1d"]["median"],
        "ExpRet_5d": forecast_stats["fwd_5d"]["median"],
        "Confidence": confidence
    })

results_df = pd.DataFrame(results)

st.dataframe(
    results_df.sort_values("DipProb_1d", ascending=False),
    use_container_width=True
)

selected = st.selectbox("Inspect ticker", results_df["symbol"])

if selected:

    options = get_options(selected)

    st.subheader("Best Put Candidates")

    st.dataframe(options)
