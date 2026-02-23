import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# DAILY VERSION OF YOUR STOCKCHARTS SETUP
# - EMA stack (weekly 5/13/21/34) -> daily ~21/55/100/170
# - PPO(21,34), PPO(5,13), PPO(1,5) weekly -> daily ~ (105,170), (21,55), (3,10)
# - SPY vs SH regime switching
# - Backtest 2000-present (SH inception ~2006; pre-inception we synthesize -1x SPY)
# - Plots: Price+EMAs + 3 PPO panels + Equity curve
# =========================================

TRADING_DAYS = 252

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_prices(tickers, start="1999-01-01"):
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
        if "Close" not in df.columns.get_level_values(0):
            return pd.DataFrame()
        close = df.xs("Close", axis=1, level=0)
    else:
        # single ticker
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df[["Close"]].copy()
        close.columns = [tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)]

    close = close.dropna(how="all").ffill().dropna(how="all")
    close.index = pd.to_datetime(close.index)
    return close

def ema(s, length):
    return s.ewm(span=length, adjust=False).mean()

def ppo(price, fast, slow, signal=9):
    """
    StockCharts-style PPO:
      PPO = 100 * (EMA(fast) - EMA(slow)) / EMA(slow)
      Signal = EMA(PPO, signal)
      Hist = PPO - Signal
    """
    ef = ema(price, fast)
    es = ema(price, slow)
    p = 100.0 * (ef - es) / es.replace(0, np.nan)
    sig = ema(p, signal)
    hist = p - sig
    return p.rename(f"PPO_{fast}_{slow}"), sig.rename(f"PPOsig_{fast}_{slow}"), hist.rename(f"PPOhist_{fast}_{slow}")

def perf_stats(equity):
    eq = equity.dropna()
    if len(eq) < 5:
        return {"Total Return": np.nan, "CAGR": np.nan, "Max Drawdown": np.nan, "Sharpe": np.nan}
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    sharpe = (rets.mean() / rets.std()) * np.sqrt(TRADING_DAYS) if rets.std() != 0 else np.nan
    return {
        "Total Return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "CAGR": float(cagr),
        "Max Drawdown": float(dd.min()),
        "Sharpe": float(sharpe),
    }

def synthesize_sh_from_spy(spy_close, annual_drag=0.009):
    """
    SH doesn't exist before ~2006. To backtest from 2000, we synthesize a -1x SPY series:
      SH_ret ≈ -SPY_ret - daily_drag
    where daily_drag approximates expense/decay.
    """
    spy_ret = spy_close.pct_change().fillna(0.0)
    daily_drag = annual_drag / TRADING_DAYS
    sh_ret = (-spy_ret - daily_drag).clip(lower=-0.99)  # avoid <= -100%
    sh_close = (1.0 + sh_ret).cumprod() * 100.0
    sh_close.name = "SH_SYNTH"
    return sh_close

def build_signals_daily(spy_close,
                        ema_fast=21, ema_13=55, ema_21=100, ema_34=170,
                        ppo_slow_fast=105, ppo_slow_slow=170,
                        ppo_sig=9,
                        ppo_entry_buffer=0.0,
                        confirm_days=2):
    """
    Regime logic (daily, "weekly-like"):
      LONG SPY when:
        - PPO(105,170) > +buffer
        - EMA55 > EMA170
        - EMA21 slope > 0
      LONG SH when:
        - PPO(105,170) < -buffer
        - EMA55 < EMA170
        - EMA21 slope < 0
    confirm_days: require N consecutive days to flip (hysteresis).
    """
    df = pd.DataFrame({"SPY": spy_close}).dropna()

    df["EMA_21"]  = ema(df["SPY"], ema_fast)
    df["EMA_55"]  = ema(df["SPY"], ema_13)
    df["EMA_100"] = ema(df["SPY"], ema_21)
    df["EMA_170"] = ema(df["SPY"], ema_34)

    df["ema21_slope"] = df["EMA_21"].diff()

    ppo_line, ppo_signal, ppo_hist = ppo(df["SPY"], ppo_slow_fast, ppo_slow_slow, signal=ppo_sig)
    df["PPO_slow"] = ppo_line
    df["PPO_slow_sig"] = ppo_signal
    df["PPO_slow_hist"] = ppo_hist

    # also compute the other two PPO panels for plotting (not for regime flips)
    ppo_mid, ppo_mid_sig, ppo_mid_hist = ppo(df["SPY"], 21, 55, signal=ppo_sig)
    df["PPO_mid"] = ppo_mid
    df["PPO_mid_sig"] = ppo_mid_sig
    df["PPO_mid_hist"] = ppo_mid_hist

    ppo_fast, ppo_fast_sig, ppo_fast_hist = ppo(df["SPY"], 3, 10, signal=ppo_sig)
    df["PPO_fast"] = ppo_fast
    df["PPO_fast_sig"] = ppo_fast_sig
    df["PPO_fast_hist"] = ppo_fast_hist

    # Conditions
    long_spy_raw = (
        (df["PPO_slow"] > +ppo_entry_buffer) &
        (df["EMA_55"] > df["EMA_170"]) &
        (df["ema21_slope"] > 0)
    )
    long_sh_raw = (
        (df["PPO_slow"] < -ppo_entry_buffer) &
        (df["EMA_55"] < df["EMA_170"]) &
        (df["ema21_slope"] < 0)
    )

    # Hysteresis via consecutive-day confirmation
    def consec_true(x, n):
        return x.rolling(n).sum() >= n

    long_spy = consec_true(long_spy_raw.astype(int), confirm_days)
    long_sh  = consec_true(long_sh_raw.astype(int), confirm_days)

    # Position: +1 SPY, -1 SH
    pos = pd.Series(np.nan, index=df.index, dtype=float)
    pos[long_spy] = 1
    pos[long_sh] = -1
    pos = pos.ffill().fillna(1)  # start long SPY

    df["position"] = pos.astype(int)
    df["asset"] = df["position"].map({1: "SPY", -1: "SH"})
    df["long_spy_raw"] = long_spy_raw
    df["long_sh_raw"] = long_sh_raw
    return df

def backtest_spy_sh(spy_close, sh_close, position, cost_bps=5.0):
    """
    No-lookahead: use yesterday's position for today's return.
    """
    idx = spy_close.index.intersection(sh_close.index).intersection(position.index)
    spy = spy_close.loc[idx].copy()
    sh = sh_close.loc[idx].copy()
    pos = position.loc[idx].copy()

    rets = pd.DataFrame({
        "SPY": spy.pct_change().fillna(0.0),
        "SH":  sh.pct_change().fillna(0.0),
    }, index=idx)

    pos_lag = pos.shift(1).fillna(pos.iloc[0])
    asset = pos_lag.map({1: "SPY", -1: "SH"})

    strat_ret = rets.to_numpy()[np.arange(len(rets)), rets.columns.get_indexer(asset)]
    strat_ret = pd.Series(strat_ret, index=idx)

    turnover = (pos.diff().abs().fillna(0) > 0).astype(int)
    strat_ret -= turnover * (cost_bps / 10000.0)

    equity = (1.0 + strat_ret).cumprod()
    return equity, int(turnover.sum()), strat_ret

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Daily Regime: SPY vs SH (EMA + PPO)", layout="wide")
st.title("📈 Daily Regime Engine: SPY vs SH (EMA stack + PPO at 3 speeds)")
st.caption("Daily indicators scaled to behave like your weekly StockCharts setup: EMAs ~21/55/100/170 and PPOs ~105/170, 21/55, 3/10.")

st.sidebar.header("Backtest Range")
start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="2000-01-01")

st.sidebar.header("Execution / Friction")
cost_bps = st.sidebar.slider("Trading cost (bps per switch)", 0.0, 50.0, 5.0, 1.0)

st.sidebar.header("Hysteresis / Noise Control")
confirm_days = st.sidebar.slider("Confirm days to flip", 1, 5, 2, 1)
ppo_buffer = st.sidebar.slider("PPO buffer (avoid tiny flips)", 0.0, 1.0, 0.10, 0.05)

st.sidebar.header("PPO Signal")
ppo_signal_len = st.sidebar.slider("PPO signal EMA length", 3, 21, 9, 1)

st.sidebar.header("Synthetic SH (pre-inception)")
use_synth_pre = st.sidebar.checkbox("Synthesize SH before it exists", value=True)
annual_drag = st.sidebar.slider("Synthetic SH annual drag", 0.0, 0.03, 0.009, 0.001)

st.sidebar.header("Chart Window")
years_show = st.sidebar.slider("Show last N years (charts)", 1, 15, 8, 1)

TICKERS = ["SPY", "SH"]

with st.spinner("Downloading SPY/SH..."):
    close = fetch_prices(TICKERS, start=start_date)

if close.empty or "SPY" not in close.columns:
    st.error("Could not download SPY data. Try again / different start date.")
    st.stop()

spy = close["SPY"].dropna()

# SH can be missing pre-inception; handle it
if "SH" in close.columns and close["SH"].dropna().shape[0] > 50:
    sh_raw = close["SH"].dropna()
else:
    sh_raw = pd.Series(dtype=float)

if use_synth_pre:
    sh_synth = synthesize_sh_from_spy(spy, annual_drag=annual_drag)
    if not sh_raw.empty:
        # stitch: use real SH when available
        sh = sh_synth.copy()
        sh.loc[sh_raw.index] = sh_raw
        sh = sh.sort_index().ffill()
    else:
        sh = sh_synth
else:
    if sh_raw.empty:
        st.error("SH data is missing and synth is OFF. Turn synth ON or pick a later start date.")
        st.stop()
    sh = sh_raw

# Build daily signals (weekly-like)
sig_df = build_signals_daily(
    spy_close=spy,
    ppo_sig=ppo_signal_len,
    ppo_entry_buffer=ppo_buffer,
    confirm_days=confirm_days,
)

# Backtest
eq_strat, trades, strat_ret = backtest_spy_sh(spy, sh, sig_df["position"], cost_bps=cost_bps)

# Buy & hold SPY
idx_bh = spy.index.intersection(eq_strat.index)
eq_bh = (1.0 + spy.loc[idx_bh].pct_change().fillna(0.0)).cumprod()

# Stats
s1 = perf_stats(eq_strat)
s2 = perf_stats(eq_bh)

st.subheader("Performance (2000-present where possible)")
stats_df = pd.DataFrame([
    {"Portfolio": "Strategy: SPY/SH (daily EMA+PPO)", "Total Return %": s1["Total Return"]*100, "CAGR %": s1["CAGR"]*100, "Max DD %": s1["Max Drawdown"]*100, "Sharpe": s1["Sharpe"], "Trades": trades},
    {"Portfolio": "Buy & Hold: SPY",                 "Total Return %": s2["Total Return"]*100, "CAGR %": s2["CAGR"]*100, "Max DD %": s2["Max Drawdown"]*100, "Sharpe": s2["Sharpe"], "Trades": 0},
])
st.dataframe(
    stats_df.style.format({
        "Total Return %": "{:.1f}",
        "CAGR %": "{:.2f}",
        "Max DD %": "{:.1f}",
        "Sharpe": "{:.2f}",
    }),
    use_container_width=True
)

# ---------------------------
# Plot panels (StockCharts-like)
# ---------------------------
st.subheader("StockCharts-like Indicator Panels (Daily)")

# limit chart window
end = sig_df.index.max()
start_show = end - pd.Timedelta(days=int(years_show * 365.25))
plot_df = sig_df.loc[sig_df.index >= start_show].copy()
plot_df = plot_df.dropna(subset=["SPY", "EMA_21", "EMA_55", "EMA_100", "EMA_170", "PPO_slow", "PPO_mid", "PPO_fast"])

fig = plt.figure(figsize=(14, 10))

# Panel 1: Price + EMA stack
ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4)
ax1.plot(plot_df.index, plot_df["SPY"], label="SPY")
ax1.plot(plot_df.index, plot_df["EMA_21"], label="EMA 21")
ax1.plot(plot_df.index, plot_df["EMA_55"], label="EMA 55")
ax1.plot(plot_df.index, plot_df["EMA_100"], label="EMA 100")
ax1.plot(plot_df.index, plot_df["EMA_170"], label="EMA 170")
ax1.set_title("SPY (Daily) + EMA Stack (21/55/100/170)")
ax1.grid(True, alpha=0.25)
ax1.legend(loc="upper left", ncol=3)

# Shade regime
# (green-ish when long SPY, red-ish when long SH; uses default colors if any)
pos_plot = plot_df["position"]
ax1.fill_between(plot_df.index, ax1.get_ylim()[0], ax1.get_ylim()[1], where=(pos_plot == 1), alpha=0.08, step="pre")
ax1.fill_between(plot_df.index, ax1.get_ylim()[0], ax1.get_ylim()[1], where=(pos_plot == -1), alpha=0.08, step="pre")

# Panel 2: PPO slow (105/170)
ax2 = plt.subplot2grid((10, 1), (4, 0), rowspan=2, sharex=ax1)
ax2.plot(plot_df.index, plot_df["PPO_slow"], label="PPO 105/170")
ax2.plot(plot_df.index, plot_df["PPO_slow_sig"], label=f"Signal {ppo_signal_len}")
ax2.bar(plot_df.index, plot_df["PPO_slow_hist"], label="Hist", alpha=0.35)
ax2.axhline(0, linewidth=1)
ax2.set_title("PPO (105/170) + Signal + Histogram")
ax2.grid(True, alpha=0.25)
ax2.legend(loc="upper left", ncol=3)

# Panel 3: PPO mid (21/55)
ax3 = plt.subplot2grid((10, 1), (6, 0), rowspan=2, sharex=ax1)
ax3.plot(plot_df.index, plot_df["PPO_mid"], label="PPO 21/55")
ax3.plot(plot_df.index, plot_df["PPO_mid_sig"], label=f"Signal {ppo_signal_len}")
ax3.bar(plot_df.index, plot_df["PPO_mid_hist"], label="Hist", alpha=0.35)
ax3.axhline(0, linewidth=1)
ax3.set_title("PPO (21/55) + Signal + Histogram")
ax3.grid(True, alpha=0.25)
ax3.legend(loc="upper left", ncol=3)

# Panel 4: PPO fast (3/10)
ax4 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, sharex=ax1)
ax4.plot(plot_df.index, plot_df["PPO_fast"], label="PPO 3/10")
ax4.plot(plot_df.index, plot_df["PPO_fast_sig"], label=f"Signal {ppo_signal_len}")
ax4.bar(plot_df.index, plot_df["PPO_fast_hist"], label="Hist", alpha=0.35)
ax4.axhline(0, linewidth=1)
ax4.set_title("PPO (3/10) + Signal + Histogram")
ax4.grid(True, alpha=0.25)
ax4.legend(loc="upper left", ncol=3)

plt.tight_layout()
st.pyplot(fig)

# Equity curve panel
st.subheader("Equity Curve ($10k start)")
eq_plot = pd.DataFrame({
    "Strategy": (eq_strat * 10000).reindex(idx_bh).ffill(),
    "SPY_BH": (eq_bh * 10000),
}).dropna()

fig2 = plt.figure(figsize=(14, 5))
plt.plot(eq_plot.index, eq_plot["Strategy"], label="Strategy SPY/SH")
plt.plot(eq_plot.index, eq_plot["SPY_BH"], label="Buy & Hold SPY", linestyle="--")
plt.grid(True, alpha=0.25)
plt.legend()
plt.ylabel("Equity ($)")
plt.xlabel("Date")
st.pyplot(fig2)

# Diagnostics table
st.subheader("Last 60 days diagnostics")
diag = sig_df[["PPO_slow", "EMA_55", "EMA_170", "ema21_slope", "position", "asset"]].tail(60).copy()
st.dataframe(diag, use_container_width=True)

with st.expander("Notes / Reality checks"):
    st.write(
        "- This uses *daily* indicators scaled to mimic your *weekly* StockCharts setup.\n"
        "- SH did not exist prior to ~2006; if synth is ON, pre-inception performance uses a synthetic -1x SPY series with adjustable annual drag.\n"
        "- No lookahead: returns use yesterday's position.\n"
        "- If you want fewer flips, increase confirm days and/or PPO buffer."
    )
