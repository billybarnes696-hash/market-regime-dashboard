import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 24, 52, 18
TSI_R, TSI_S, TSI_SIGNAL = 40, 20, 10
STOCH_LEN, STOCH_SMOOTHK, STOCH_SMOOTHD = 14, 3, 3
CCI_LEN = 100
TRADING_DAYS = 252

# =========================
# DATA
# =========================
@st.cache_data(ttl=3600)
def fetch_adjclose(tickers, years=5):
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(years * 365.25) + 30)
    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        st.error(f"Data download failed: {e}")
        return pd.DataFrame()

    # Normalize output into a simple Close dataframe
    if isinstance(tickers, str) or len(tickers) == 1:
        t = tickers if isinstance(tickers, str) else tickers[0]
        if "Close" in data.columns:
            close = data["Close"].to_frame(t)
        else:
            close = data["Close"].to_frame(t)  # fallback
        return close.dropna(how="all").ffill().dropna()

    close_cols = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                if t in data.columns.get_level_values(0) and "Close" in data[t].columns:
                    close_cols[t] = data[t]["Close"]
            except Exception:
                continue
    else:
        # sometimes yfinance returns flat columns for single ticker even if list passed
        if "Close" in data.columns:
            # can't reliably split per ticker; return empty to be safe
            return pd.DataFrame()

    close = pd.DataFrame(close_cols)
    return close.dropna(how="all").ffill().dropna()

# =========================
# INDICATORS (close-only helpers)
# =========================
def close_only_stoch(close, length=14, smoothk=3, smoothd=3):
    lo = close.rolling(length).min()
    hi = close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = (100.0 * (close - lo) / denom).rolling(smoothk).mean()
    d = k.rolling(smoothd).mean()
    return k, d

def close_only_cci(close, length=100):
    sma = close.rolling(length).mean()
    mad = (close - sma).abs().rolling(length).mean()
    return (close - sma) / (0.015 * mad.replace(0, np.nan))

def zscore(s, win=252):
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std()
    z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-3, 3)

def indicator_pack_continuous(close):
    close = close.dropna()
    min_req = max(MACD_SLOW, TSI_R, CCI_LEN, 252) + 50
    if len(close) < min_req:
        return pd.DataFrame()

    # MACD
    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is None or macd.empty:
        return pd.DataFrame()

    macdh_candidates = [c for c in macd.columns if "MACDh" in str(c) or "HIST" in str(c).upper()]
    if macdh_candidates:
        macdh = macd[macdh_candidates[0]].rename("macdh")
    else:
        # fallback attempt: macd - signal if both exist
        macd_line = [c for c in macd.columns if "MACD_" in str(c) and "MACDh" not in str(c)]
        sig_line = [c for c in macd.columns if "MACDs" in str(c) or "SIGNAL" in str(c).upper()]
        if macd_line and sig_line:
            macdh = (macd[macd_line[0]] - macd[sig_line[0]]).rename("macdh")
        else:
            return pd.DataFrame()

    # TSI (pandas_ta sometimes returns Series or DataFrame)
    tsi_raw = ta.tsi(close, fast=TSI_S, slow=TSI_R, signal=TSI_SIGNAL)
    if tsi_raw is None or len(tsi_raw) == 0:
        return pd.DataFrame()
    if isinstance(tsi_raw, pd.Series):
        tsi = tsi_raw.rename("tsi")
    else:
        cols = list(tsi_raw.columns)
        prefer = [c for c in cols if "TSI" in str(c) and "sig" not in str(c).lower()]
        tsi = tsi_raw[prefer[0]].rename("tsi") if prefer else tsi_raw.iloc[:, 0].rename("tsi")

    # Stoch + CCI from close only
    stoch_k, _ = close_only_stoch(close, length=STOCH_LEN, smoothk=STOCH_SMOOTHK, smoothd=STOCH_SMOOTHD)
    cci = close_only_cci(close, length=CCI_LEN)

    ind = pd.concat([macdh, tsi, stoch_k.rename("stochk"), cci.rename("cci")], axis=1).dropna()
    return ind

def score_from_indicators_continuous(ind):
    """
    Continuous score ~ [-1, +1]
    """
    if ind.empty:
        return pd.Series(dtype=float)

    macdh_z = zscore(ind["macdh"])
    tsi_z = zscore(ind["tsi"])
    stoch_z = zscore(ind["stochk"] - 50.0)
    cci_z = zscore(ind["cci"])

    raw = (macdh_z + tsi_z + stoch_z + cci_z) / 4.0
    score = np.tanh(raw / 1.25)
    return pd.Series(score, index=ind.index, name="score")

# =========================
# COMPOSITE ENGINE
# =========================
def make_ratio(close_df, num, den):
    if num not in close_df.columns or den not in close_df.columns:
        return pd.Series(dtype=float)
    return (close_df[num] / close_df[den]).replace([np.inf, -np.inf], np.nan).rename(f"{num}:{den}")

def build_composite_scores(close):
    ratios = {
        "SPXS:SVOL": {"series": make_ratio(close, "SPXS", "SVOL"), "invert": True,  "weight": 0.22},
        "HYG:SHY":   {"series": make_ratio(close, "HYG",  "SHY"),  "invert": False, "weight": 0.18},
        "SMH:SPY":   {"series": make_ratio(close, "SMH",  "SPY"),  "invert": False, "weight": 0.14},
        "SPY:VXX":   {"series": make_ratio(close, "SPY",  "VXX"),  "invert": False, "weight": 0.08},
        "XLF:SPY":   {"series": make_ratio(close, "XLF",  "SPY"),  "invert": False, "weight": 0.10},
        "RSP:SPY":   {"series": make_ratio(close, "RSP",  "SPY"),  "invert": False, "weight": 0.08},
        "IWM:SPY":   {"series": make_ratio(close, "IWM",  "SPY"),  "invert": False, "weight": 0.08},
        "XLY:SPY":   {"series": make_ratio(close, "XLY",  "SPY"),  "invert": False, "weight": 0.06},
        "SOXX:SPY":  {"series": make_ratio(close, "SOXX", "SPY"),  "invert": False, "weight": 0.06},
    }

    scores = {}
    for name, cfg in ratios.items():
        s = cfg["series"].dropna()
        if s.empty:
            continue
        ind = indicator_pack_continuous(s)
        if ind.empty:
            continue
        sc = score_from_indicators_continuous(ind)
        if sc.empty:
            continue
        scores[name] = (-sc if cfg["invert"] else sc)

    if not scores:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=bool), pd.Series(dtype=bool)

    # align index
    all_idx = None
    for sc in scores.values():
        all_idx = sc.index if all_idx is None else all_idx.intersection(sc.index)

    scores_df = pd.DataFrame({k: v.loc[all_idx] for k, v in scores.items()}).dropna()
    if scores_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=bool), pd.Series(dtype=bool)

    w = pd.Series({k: ratios[k]["weight"] for k in scores_df.columns})
    w = w / w.sum()
    composite = (scores_df * w).sum(axis=1)

    # confidence: agreement + magnitude (meaningful now because score is continuous)
    comp_sign = np.sign(composite.replace(0, np.nan))
    align = (np.sign(scores_df).replace(0, np.nan).eq(comp_sign, axis=0)).mean(axis=1).fillna(0)
    magnitude = scores_df.abs().mean(axis=1).clip(0, 1)
    confidence = (0.5 * align + 0.5 * magnitude) * 100.0

    credit_gate = (scores_df["HYG:SHY"] > 0) if "HYG:SHY" in scores_df.columns else pd.Series(False, index=composite.index)
    stress_gate = (scores_df["SPXS:SVOL"] > 0) if "SPXS:SVOL" in scores_df.columns else pd.Series(False, index=composite.index)

    return scores_df, composite, confidence, credit_gate, stress_gate

# =========================
# POSITIONS: FASTEST PATH = SPY/SHY (NO SH DRAG) + HYSTERESIS
# =========================
def build_positions_spy_shy(
    composite,
    confidence,
    spy_close,
    riskoff_thr=-0.08,
    riskon_thr=0.12,
    conf_thr=45,
    use_sma_exit=True,
    use_gates=True,
    credit_gate=None,
    stress_gate=None
):
    df = pd.DataFrame({
        "comp": composite,
        "conf": confidence,
        "spy": spy_close.reindex(composite.index).ffill()
    }).dropna()
    if df.empty:
        return pd.DataFrame()

    if use_sma_exit:
        df["sma_200"] = df["spy"].rolling(200).mean()
        df["below_200"] = (df["spy"] < df["sma_200"]).astype(int)
    else:
        df["below_200"] = 0

    if use_gates and credit_gate is not None and stress_gate is not None:
        df["credit_ok"] = credit_gate.reindex(df.index).fillna(False).astype(int)
        df["stress_ok"] = stress_gate.reindex(df.index).fillna(False).astype(int)
    else:
        df["credit_ok"] = 1
        df["stress_ok"] = 1

    pos = pd.Series(1, index=df.index)  # start in SPY

    go_riskoff = (df["comp"] <= riskoff_thr) & (df["conf"] >= conf_thr)
    go_riskon  = (df["comp"] >= riskon_thr)  & (df["conf"] >= conf_thr)

    if use_gates:
        go_riskoff |= (df["credit_ok"] == 0) | (df["stress_ok"] == 0)
        go_riskon  &= (df["credit_ok"] == 1) & (df["stress_ok"] == 1)

    if use_sma_exit:
        go_riskoff |= (df["below_200"] == 1)

    pos[go_riskoff] = 0
    pos[go_riskon] = 1
    pos = pos.ffill().astype(int)

    df["position"] = pos
    df["asset"] = pos.map({1: "SPY", 0: "SHY"})
    return df

# =========================
# BACKTEST (NO LOOKAHEAD)
# =========================
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
        "Total Return": (eq.iloc[-1] / eq.iloc[0]) - 1.0,
        "CAGR": cagr,
        "Max Drawdown": float(dd.min()),
        "Sharpe": float(sharpe),
    }

def backtest_spy_shy(spy, shy, pos_df, cost_bps=5.0):
    idx = spy.index.intersection(pos_df.index)
    df = pd.DataFrame({
        "SPY": spy.loc[idx],
        "SHY": shy.loc[idx],
        "pos": pos_df["position"].loc[idx],
    }).dropna()

    if df.empty:
        return pd.Series(dtype=float), 0

    rets = pd.DataFrame({
        "SPY": df["SPY"].pct_change().fillna(0.0),
        "SHY": df["SHY"].pct_change().fillna(0.0),
    }, index=df.index)

    # use yesterday's position for today's return
    pos_lag = df["pos"].shift(1).fillna(df["pos"].iloc[0])
    asset = pos_lag.map({1: "SPY", 0: "SHY"})

    strat_ret = rets.to_numpy()[np.arange(len(rets)), rets.columns.get_indexer(asset)]
    strat_ret = pd.Series(strat_ret, index=df.index)

    turnover = (df["pos"].diff().abs().fillna(0) > 0).astype(int)
    strat_ret -= turnover * (cost_bps / 10000.0)

    equity = (1.0 + strat_ret).cumprod()
    return equity, int(turnover.sum())

# =========================
# UI
# =========================
st.set_page_config(page_title="Regime Turbo (Fixed): SPY/SHY", layout="wide")
st.title("🚀 Regime Turbo (Fixed): SPY/SHY Overlay")
st.caption("Fast path: continuous scoring + hysteresis + SMA as exit + gates + no lookahead.")

st.sidebar.header("Data")
years = st.sidebar.slider("History (years)", 3, 10, 5)

st.sidebar.header("Risk On/Off Controls (Hysteresis)")
riskoff_thr = st.sidebar.slider("Risk-OFF (to SHY) if comp <=", -0.60, 0.00, -0.08, 0.02)
riskon_thr  = st.sidebar.slider("Risk-ON (to SPY) if comp >=",  0.00, 0.60,  0.12, 0.02)
conf_thr    = st.sidebar.slider("Min Confidence", 10, 95, 45, 5)

st.sidebar.header("Filters")
use_sma_exit = st.sidebar.checkbox("Use 200-SMA as EXIT (force SHY if below)", value=True)
use_gates    = st.sidebar.checkbox("Use Credit/Stress Gates", value=True)

st.sidebar.header("Costs")
cost_bps = st.sidebar.slider("Trading Cost (bps)", 0.0, 50.0, 5.0, 1.0)

TICKERS = ["SPY", "SHY", "SPXS", "SVOL", "VXX", "HYG", "SMH", "SOXX", "XLF", "RSP", "IWM", "XLY"]

with st.spinner("📥 Fetching data..."):
    close = fetch_adjclose(TICKERS, years=years)

missing = [t for t in ["SPY", "SHY"] if t not in close.columns]
if missing:
    st.error(f"❌ Missing required data: {missing}")
    st.stop()

with st.spinner("🔧 Building signals..."):
    scores_df, composite, confidence, credit_gate, stress_gate = build_composite_scores(close)

if composite.empty:
    st.error("❌ Failed to build signals. Try increasing history or check tickers.")
    st.stop()

pos_df = build_positions_spy_shy(
    composite=composite,
    confidence=confidence,
    spy_close=close["SPY"],
    riskoff_thr=riskoff_thr,
    riskon_thr=riskon_thr,
    conf_thr=conf_thr,
    use_sma_exit=use_sma_exit,
    use_gates=use_gates,
    credit_gate=credit_gate,
    stress_gate=stress_gate,
)

if pos_df.empty:
    st.error("❌ No positions generated. Lower thresholds / confidence or increase history.")
    st.stop()

# Backtest
eq_strat, trades = backtest_spy_shy(close["SPY"], close["SHY"], pos_df, cost_bps)
idx = close["SPY"].index.intersection(pos_df.index)
bh = (1.0 + close["SPY"].loc[idx].pct_change().fillna(0.0)).cumprod()

# Stats table
stats = {
    "Strategy: SPY/SHY Overlay": perf_stats(eq_strat),
    "Buy&Hold: SPY": perf_stats(bh),
}
rows = []
for name, s in stats.items():
    rows.append({
        "Portfolio": name,
        "Total Return %": f"{s['Total Return']*100:.1f}" if not np.isnan(s["Total Return"]) else "N/A",
        "CAGR %": f"{s['CAGR']*100:.2f}" if not np.isnan(s["CAGR"]) else "N/A",
        "Max DD %": f"{s['Max Drawdown']*100:.1f}" if not np.isnan(s["Max Drawdown"]) else "N/A",
        "Sharpe": f"{s['Sharpe']:.2f}" if not np.isnan(s["Sharpe"]) else "N/A",
        "Trades": int(trades) if "Strategy" in name else 0,
    })

st.subheader("Performance Comparison")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# Equity curves
st.subheader("Equity Curves ($10k Start)")
fig = plt.figure(figsize=(12, 6))
plt.plot(eq_strat.index, eq_strat * 10000, label="SPY/SHY Overlay", linewidth=2)
plt.plot(bh.index, bh * 10000, label="Buy&Hold SPY", linestyle="--", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(True, alpha=0.3)
st.pyplot(fig)

# Diagnostics
st.subheader("🔍 Signal Diagnostics (Last 80 Bars)")
diag_cols = ["comp", "conf", "position", "asset"]
if use_sma_exit:
    diag_cols.insert(2, "below_200")
if use_gates:
    diag_cols.extend(["credit_ok", "stress_ok"])

st.dataframe(pos_df.tail(80)[diag_cols], use_container_width=True)

st.caption("Position: 1=SPY, 0=SHY. Uses yesterday's position for today's return (no lookahead).")

# Debug
with st.expander("🔧 Debug"):
    st.write(f"Composite Range: {composite.min():.3f} to {composite.max():.3f}")
    st.write(f"Confidence Range: {confidence.min():.1f} to {confidence.max():.1f}")
    st.write(f"Trades: {trades}")
    st.write(f"Current Asset: {pos_df['asset'].iloc[-1]}")
    st.write("Included ratio scores:", list(scores_df.columns))
