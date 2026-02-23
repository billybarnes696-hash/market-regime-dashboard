import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRADING_DAYS = 252

# =========================
# DATA
# =========================
@st.cache_data(ttl=3600)
def fetch_prices(tickers, start="2010-01-01"):
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
    else:
        close = df[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(how="all").ffill().dropna(how="all")
    close.index = pd.to_datetime(close.index)
    return close

# =========================
# INDICATORS
# =========================
def ema(s, length):
    return s.ewm(span=length, adjust=False).mean()

def ppo(price, fast, slow, signal=9):
    ef = ema(price, fast)
    es = ema(price, slow)
    p = 100.0 * (ef - es) / es.replace(0, np.nan)
    sig = ema(p, signal)
    hist = p - sig
    return p, sig, hist

# =========================
# PERFORMANCE
# =========================
def perf_stats(equity):
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "Total Return %": (eq.iloc[-1] / eq.iloc[0] - 1) * 100,
        "CAGR %": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100,
        "Max DD %": (eq / eq.cummax() - 1).min() * 100,
        "Sharpe": (rets.mean() / rets.std()) * np.sqrt(TRADING_DAYS),
    }

def backtest_spy_shy(spy, shy, position, cost_bps=5.0):
    idx = spy.index.intersection(shy.index).intersection(position.index)
    spy = spy.loc[idx]
    shy = shy.loc[idx]
    pos = position.loc[idx]

    rets = pd.DataFrame({
        "SPY": spy.pct_change().fillna(0.0),
        "SHY": shy.pct_change().fillna(0.0),
    })

    pos_lag = pos.shift(1).fillna(1)
    asset = pos_lag.map({1: "SPY", 0: "SHY"})
    strat_ret = rets.to_numpy()[np.arange(len(rets)), rets.columns.get_indexer(asset)]
    strat_ret = pd.Series(strat_ret, index=idx)

    turnover = (pos.diff().abs().fillna(0) > 0).astype(int)
    strat_ret -= turnover * (cost_bps / 10000)

    equity = (1 + strat_ret).cumprod()
    return equity, turnover.sum()

# =========================
# POSITIONS FROM PPO ZERO CROSS
# =========================
def pos_from_ppo(ppo_series):
    pos = (ppo_series > 0).astype(int)
    return pos.ffill().fillna(1)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="PPO Zero-Cross SPY/SHY", layout="wide")
st.title("📊 PPO Zero-Cross Rotation: SPY vs SHY")

start_date = st.sidebar.text_input("Start date", "2010-01-01")
cost_bps = st.sidebar.slider("Trading cost (bps)", 0.0, 25.0, 5.0, 1.0)

with st.spinner("Downloading data..."):
    close = fetch_prices(["SPY", "SHY"], start=start_date)

spy = close["SPY"]
shy = close["SHY"]

# =========================
# PPOs
# =========================
ppo_slow, _, _ = ppo(spy, 105, 170)
ppo_mid,  _, _ = ppo(spy, 21, 55)
ppo_fast, _, _ = ppo(spy, 3, 10)

# =========================
# POSITIONS
# =========================
pos_slow = pos_from_ppo(ppo_slow)
pos_mid  = pos_from_ppo(ppo_mid)
pos_fast = pos_from_ppo(ppo_fast)

# =========================
# BACKTESTS
# =========================
eq_slow, trades_slow = backtest_spy_shy(spy, shy, pos_slow, cost_bps)
eq_mid,  trades_mid  = backtest_spy_shy(spy, shy, pos_mid,  cost_bps)
eq_fast, trades_fast = backtest_spy_shy(spy, shy, pos_fast, cost_bps)

idx = spy.index.intersection(eq_slow.index)
eq_bh = (1 + spy.loc[idx].pct_change().fillna(0)).cumprod()

# =========================
# RESULTS TABLE
# =========================
rows = []
for name, eq, tr in [
    ("PPO 105/170 > 0", eq_slow, trades_slow),
    ("PPO 21/55 > 0",  eq_mid,  trades_mid),
    ("PPO 3/10 > 0",   eq_fast, trades_fast),
    ("Buy & Hold SPY", eq_bh,   0),
]:
    s = perf_stats(eq)
    s["Strategy"] = name
    s["Trades"] = tr
    rows.append(s)

st.subheader("Performance Comparison")
st.dataframe(
    pd.DataFrame(rows)[
        ["Strategy", "Total Return %", "CAGR %", "Max DD %", "Sharpe", "Trades"]
    ].style.format({
        "Total Return %": "{:.1f}",
        "CAGR %": "{:.2f}",
        "Max DD %": "{:.1f}",
        "Sharpe": "{:.2f}",
    }),
    use_container_width=True
)

# =========================
# EQUITY CURVES
# =========================
st.subheader("Equity Curves ($10k start)")
fig = plt.figure(figsize=(14, 6))
plt.plot(eq_slow.index, eq_slow * 10000, label="PPO 105/170")
plt.plot(eq_mid.index,  eq_mid * 10000,  label="PPO 21/55")
plt.plot(eq_fast.index, eq_fast * 10000, label="PPO 3/10")
plt.plot(eq_bh.index,   eq_bh * 10000,   label="Buy & Hold SPY", linestyle="--")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylabel("Equity ($)")
plt.xlabel("Date")
st.pyplot(fig)

# =========================
# DIAGNOSTICS
# =========================
st.subheader("Recent PPO Values")
diag = pd.DataFrame({
    "PPO 105/170": ppo_slow,
    "PPO 21/55": ppo_mid,
    "PPO 3/10": ppo_fast,
    "SPY": spy,
}).tail(50)
st.dataframe(diag, use_container_width=True)
