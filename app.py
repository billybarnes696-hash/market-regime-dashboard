import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import altair as alt

# ============================================================
# LOCKED MODEL SETTINGS (frozen)
# ============================================================
LOOKBACK_YEARS = 5
INTERVAL = "1d"
MIN_BARS = 260
TREND_LOOKBACK_DAYS = 10

# Frozen indicators
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 24, 52, 18
TSI_LONG, TSI_SHORT, TSI_SIGNAL = 40, 20, 10
STOCH_LEN, STOCH_SMOOTH_K, STOCH_SMOOTH_D = 14, 3, 3

# Frozen action thresholds
BUY_LEVEL_TH = 0.20
SELL_LEVEL_TH = -0.20
CONF_TH = 70

st.set_page_config(page_title="Robust Market Regime Dashboard", layout="wide")


# ============================================================
# INDICATOR HELPERS (no external TA libs)
# ============================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd_hist(series: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    m = ema(series, fast) - ema(series, slow)
    sig = ema(m, signal)
    return (m - sig)

def tsi(series: pd.Series, long_len: int, short_len: int, signal_len: int):
    m = series.diff()
    ema1 = ema(m, short_len)
    ema2 = ema(ema1, long_len)

    abs_m = m.abs()
    abs_ema1 = ema(abs_m, short_len)
    abs_ema2 = ema(abs_ema1, long_len)

    tsi_val = 100 * (ema2 / abs_ema2.replace(0, np.nan))
    tsi_sig = ema(tsi_val, signal_len)
    return tsi_val, tsi_sig

def stoch_kd(series: pd.Series, length: int, smooth_k: int, smooth_d: int):
    low = series.rolling(length).min()
    high = series.rolling(length).max()
    k = 100 * (series - low) / (high - low).replace(0, np.nan)
    k_s = k.rolling(smooth_k).mean()
    d_s = k_s.rolling(smooth_d).mean()
    return k_s, d_s

def safe_last(series: pd.Series):
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def normalize_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series
    base = s.iloc[0]
    if base == 0 or np.isnan(base):
        return series
    return (series / base) * 100.0

def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


# ============================================================
# PERFORMANCE HELPERS (backtest stats)
# ============================================================
def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return np.nan
    total = eq.iloc[-1] / eq.iloc[0]
    years = (len(eq) - 1) / periods_per_year
    if years <= 0:
        return np.nan
    return float(total ** (1 / years) - 1)

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float(r.std() * math.sqrt(periods_per_year))

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    excess = r - (rf / periods_per_year)
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(excess.mean() / vol * math.sqrt(periods_per_year))


# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=60 * 30)
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
    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        out = pd.DataFrame(closes)
    else:
        out = pd.DataFrame({tickers[0]: df["Close"]})
    out = out.dropna(how="all").sort_index()
    return out


# ============================================================
# MODEL SCORING
# ============================================================
def series_score(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Per-day score in [-1, +1] using frozen:
      - MACD histogram sign (24,52,18)
      - TSI sign (40,20,10)
      - Stoch %K relative to 50 (14,3,3)
    """
    s = series.dropna()
    if s.shape[0] < MIN_BARS:
        return pd.Series(index=series.index, dtype=float)

    mh = macd_hist(s, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    tval, _ = tsi(s, TSI_LONG, TSI_SHORT, TSI_SIGNAL)
    k, _ = stoch_kd(s, STOCH_LEN, STOCH_SMOOTH_K, STOCH_SMOOTH_D)

    macd_dir = np.where(mh > 0, 1.0, -1.0)
    tsi_dir = np.where(tval > 0, 1.0, -1.0)
    stoch_dir = np.where(k > 50, 1.0, -1.0)

    raw = 0.4 * macd_dir + 0.4 * tsi_dir + 0.2 * stoch_dir
    raw = pd.Series(raw, index=s.index).clip(-1, 1)

    if invert:
        raw = -raw

    out = pd.Series(index=series.index, dtype=float)
    out.loc[raw.index] = raw
    return out


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


# ============================================================
# COMPONENTS (robust: ratios + breadth + credit + leadership)
# ============================================================
CORE_TICKERS = [
    "SPY", "SPXS", "SVOL", "VXX",
    "HYG", "SHY",
    "SMH", "SOXX", "XLF", "RSP",
    "IWM", "XLY",
]

# Breadth basket proxy (works everywhere via yfinance)
BREADTH_BASKET = sorted(list(set([
    "SPY","RSP","IWM","MDY","IJR",
    "XLK","XLF","XLY","XLP","XLV","XLI","XLE","XLB","XLU",
    "SMH","SOXX","XBI","IBB","XRT","ITB","XHB",
    "EEM","EWG","EWJ","EWY","FXI","EWZ",
    "GDX","SLV","GLD","USO","XOP","OIH",
    "AAPL","MSFT","AMZN","GOOGL","NVDA","AMD","META","TSLA",
    "JPM","BAC","WFC","C","GS","MS",
    "CAT","DE","FDX","DIS","NKE","MA",
    "INTC","AMAT","MU","NFLX","SBUX"
])))

# Composite components
COMPONENTS = {
    # Stress / carry
    "SPXS:SVOL (Stress/Carry)": {"a": "SPXS", "b": "SVOL", "invert": False, "weight": 0.22},

    # Vol confirmation (late by design)
    "SPY:VXX (Vol Confirm)": {"a": "SPY", "b": "VXX", "invert": True, "weight": 0.10},

    # Credit gate
    "HYG:SHY (Credit)": {"a": "HYG", "b": "SHY", "invert": False, "weight": 0.18},

    # Leadership
    "SMH:SPY (Semis Leadership)": {"a": "SMH", "b": "SPY", "invert": False, "weight": 0.15},
    "XLF:SPY (Financials Leadership)": {"a": "XLF", "b": "SPY", "invert": False, "weight": 0.10},

    # Breadth / concentration
    "RSP:SPY (Equal-Weight)": {"a": "RSP", "b": "SPY", "invert": False, "weight": 0.10},

    # Cyclical support (light)
    "IWM:SPY (Small Caps)": {"a": "IWM", "b": "SPY", "invert": False, "weight": 0.08},
    "XLY:SPY (Discretionary)": {"a": "XLY", "b": "SPY", "invert": False, "weight": 0.07},
}

# Normalize weights to sum to 1.0
w_sum = sum(v["weight"] for v in COMPONENTS.values())
for k in COMPONENTS:
    COMPONENTS[k]["weight"] = COMPONENTS[k]["weight"] / w_sum


# ============================================================
# UI HEADER
# ============================================================
st.title("Robust Market Regime Dashboard (Locked)")
st.caption(
    "Daily | 5 years | MACD(24,52,18) + TSI(40,20,10) + Stoch(14,3,3) | Composite + Trend + Confidence + Backtest"
)

with st.sidebar:
    st.header("Backtest controls")
    start_equity = st.number_input("Starting equity ($)", min_value=1000, value=10000, step=1000)
    cost_bps = st.number_input("Trading cost (bps per position change)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
    hold_last = st.checkbox("Hold last position until new signal", value=True)
    st.divider()
    st.write("Model thresholds are locked:")
    st.write(f"BUY if Composite>{BUY_LEVEL_TH}, Trend>0, Confidence≥{CONF_TH}, CreditGate ON, StressGate ON")
    st.write(f"SELL if Composite<{SELL_LEVEL_TH}, Trend<0, Confidence≥{CONF_TH}, CreditGate OFF, StressGate OFF")


# ============================================================
# LOAD DATA
# ============================================================
period = f"{LOOKBACK_YEARS}y"
tickers_needed = sorted(list(set(CORE_TICKERS + BREADTH_BASKET)))

with st.spinner("Downloading market data (yfinance)..."):
    close_df = fetch_prices(tickers_needed, period=period, interval=INTERVAL)

if close_df.shape[0] < MIN_BARS or "SPY" not in close_df.columns:
    st.error("Not enough data loaded or SPY missing. Try again later.")
    st.stop()

close_df = close_df.ffill().dropna(how="all")

spy_close = close_df["SPY"].dropna()
idx = spy_close.index  # primary index


# ============================================================
# BUILD COMPONENT SCORES (series)
# ============================================================
score_series_map = {}
component_rows = []

for name, cfg in COMPONENTS.items():
    a, b = cfg["a"], cfg["b"]
    if a not in close_df.columns or b not in close_df.columns:
        continue

    s = ratio(close_df[a], close_df[b]).reindex(idx).ffill()
    sc = series_score(s, invert=cfg["invert"]).reindex(idx)

    score_series_map[name] = sc

    latest_sc = safe_last(sc)
    weighted_latest = latest_sc * cfg["weight"] if not np.isnan(latest_sc) else np.nan
    component_rows.append({
        "Component": name,
        "Weight": round(cfg["weight"], 3),
        "Latest Score (-1..+1)": None if np.isnan(latest_sc) else round(latest_sc, 2),
        "Weighted (latest)": None if np.isnan(weighted_latest) else round(weighted_latest, 2),
    })

scores_df = pd.DataFrame(score_series_map).reindex(idx)

weights = pd.Series({k: COMPONENTS[k]["weight"] for k in score_series_map.keys()})
composite = (scores_df * weights).sum(axis=1) / weights.sum()
composite = composite.clip(-1, 1)

composite_level = safe_last(composite)
composite_trend = safe_last(composite - composite.shift(TREND_LOOKBACK_DAYS))

# ============================================================
# BREADTH (basket proxy)
# ============================================================
basket_cols = [c for c in BREADTH_BASKET if c in close_df.columns]
basket = close_df[basket_cols].reindex(idx).ffill()

pct_50 = pct_above_ma(basket, 50)
pct_150 = pct_above_ma(basket, 150)
pct_200 = pct_above_ma(basket, 200)
adl = adv_decl_line(basket)
nhnl = nh_nl(basket, 252)

breadth_flags = pd.DataFrame({
    "%>50DMA improving": (pct_50 - pct_50.shift(TREND_LOOKBACK_DAYS)) > 0,
    "%>150DMA improving": (pct_150 - pct_150.shift(TREND_LOOKBACK_DAYS)) > 0,
    "%>200DMA improving": (pct_200 - pct_200.shift(TREND_LOOKBACK_DAYS)) > 0,
    "A/D Line improving": (adl - adl.shift(TREND_LOOKBACK_DAYS)) > 0,
    "NH-NL improving": (nhnl - nhnl.shift(TREND_LOOKBACK_DAYS)) > 0,
}).reindex(idx)

breadth_confirm = 100.0 * breadth_flags.mean(axis=1)
breadth_confirm_latest = safe_last(breadth_confirm)

# ============================================================
# CONFIDENCE SERIES (agreement + breadth + trend stability)
# ============================================================
comp_dir = np.where(composite > 0, 1, -1)
comp_dir = pd.Series(comp_dir, index=idx)

agreement_pct = pd.Series(index=idx, dtype=float)
trend_stability_pct = pd.Series(index=idx, dtype=float)

scores_delta = scores_df - scores_df.shift(TREND_LOOKBACK_DAYS)
comp_trend_series = composite - composite.shift(TREND_LOOKBACK_DAYS)

for dt in idx:
    row = scores_df.loc[dt].dropna()
    if row.empty:
        agreement_pct.loc[dt] = np.nan
    else:
        cd = comp_dir.loc[dt]
        agree = ((row > 0) & (cd == 1)).sum() + ((row < 0) & (cd == -1)).sum()
        agreement_pct.loc[dt] = 100.0 * (agree / len(row))

    drow = scores_delta.loc[dt].dropna()
    if drow.empty or np.isnan(comp_trend_series.loc[dt]):
        trend_stability_pct.loc[dt] = np.nan
    else:
        ct = comp_trend_series.loc[dt]
        same = ((ct >= 0) & (drow >= 0)).sum() + ((ct < 0) & (drow < 0)).sum()
        trend_stability_pct.loc[dt] = 100.0 * (same / len(drow))

confidence = 0.4 * agreement_pct + 0.3 * breadth_confirm + 0.3 * trend_stability_pct
confidence_latest = safe_last(confidence)

# ============================================================
# GATES (series)
# ============================================================
hyg_shy = ratio(close_df["HYG"], close_df["SHY"]).reindex(idx).ffill()
spxs_svol = ratio(close_df["SPXS"], close_df["SVOL"]).reindex(idx).ffill()

hyg_macd = macd_hist(hyg_shy, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
hyg_tsi, _ = tsi(hyg_shy, TSI_LONG, TSI_SHORT, TSI_SIGNAL)
credit_gate = (hyg_macd > 0) & (hyg_tsi > 0)

stress_macd = macd_hist(spxs_svol, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
stress_tsi, _ = tsi(spxs_svol, TSI_LONG, TSI_SHORT, TSI_SIGNAL)
# supportive = stress fading (both negative)
stress_gate = (stress_macd < 0) & (stress_tsi < 0)

credit_gate_latest = bool(credit_gate.dropna().iloc[-1]) if credit_gate.dropna().shape[0] else False
stress_gate_latest = bool(stress_gate.dropna().iloc[-1]) if stress_gate.dropna().shape[0] else False

# ============================================================
# ACTION + LABEL (latest)
# ============================================================
action = "HOLD / NEUTRAL"
action_color = "gray"

if (composite_level > BUY_LEVEL_TH and composite_trend > 0 and confidence_latest >= CONF_TH
    and credit_gate_latest and stress_gate_latest):
    action = "BUY (Long SPY)"
    action_color = "green"
elif (composite_level < SELL_LEVEL_TH and composite_trend < 0 and confidence_latest >= CONF_TH
      and (not credit_gate_latest) and (not stress_gate_latest)):
    action = "SELL / SHORT (Short SPY)"
    action_color = "red"

# 4-state regime label (your grid)
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

# ============================================================
# TOP METRICS
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Composite Level (latest)", f"{composite_level:.2f}")
c2.metric("Composite Trend (10d Δ)", f"{composite_trend:.2f}")
c3.metric("Confidence Score (latest)", f"{confidence_latest:.0f}")
c4.metric("Regime Label", f"{label} {emoji}")

st.markdown(
    f"<div style='padding:14px;border-radius:12px;border:1px solid #ddd;'>"
    f"<span style='font-size:22px;font-weight:800;color:{action_color};'>{action}</span>"
    f"<div style='margin-top:6px;color:#666;'>"
    f"Credit gate={'ON' if credit_gate_latest else 'OFF'} | Stress gate={'ON' if stress_gate_latest else 'OFF'}"
    f"</div></div>",
    unsafe_allow_html=True
)

# ============================================================
# CHART: Composite (normalized) + SPY overlay (normalized)
# ============================================================
st.subheader("5-Year Overlay: Composite vs SPY (Normalized)")

spy_norm = normalize_to_100(spy_close)
# shift composite up so normalization works and doesn't flip sign; this is only for plotting
comp_plot = (composite + 2.0).reindex(idx)
comp_norm = normalize_to_100(comp_plot)

plot_df = pd.DataFrame({
    "Date": idx,
    "SPY (Normalized)": spy_norm.values,
    "Composite (Normalized)": comp_norm.values
}).dropna()

base = alt.Chart(plot_df).encode(x="Date:T")
line_spy = base.mark_line().encode(y=alt.Y("SPY (Normalized):Q", title="Normalized (base=100)"))
line_comp = base.mark_line(strokeDash=[6, 3]).encode(y=alt.Y("Composite (Normalized):Q"))
st.altair_chart((line_spy + line_comp).interactive(), use_container_width=True)

# ============================================================
# COMPONENT TABLE
# ============================================================
st.subheader("Components (Latest)")
comp_table = pd.DataFrame(component_rows).sort_values("Weight", ascending=False)
st.dataframe(comp_table, use_container_width=True)

# ============================================================
# BREADTH PANEL
# ============================================================
st.subheader("Breadth Health (Basket Proxies)")
b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("% > 50DMA", f"{safe_last(pct_50):.0f}%")
b2.metric("% > 150DMA", f"{safe_last(pct_150):.0f}%")
b3.metric("% > 200DMA", f"{safe_last(pct_200):.0f}%")
b4.metric("NH − NL (52w)", f"{safe_last(nhnl):.0f}")
b5.metric("Breadth Confirm", f"{breadth_confirm_latest:.0f}")

flags_latest = breadth_flags.tail(1).T.reset_index()
flags_latest.columns = ["Breadth Check", "Improving (10d)"]
st.dataframe(flags_latest, use_container_width=True)

# ============================================================
# BACKTEST: Turn the rules into a position series and compute P&L
# ============================================================
st.subheader("Backtest (Long/Short/Cash on SPY using these exact rules)")

BUY = (composite > BUY_LEVEL_TH) & (comp_trend_series > 0) & (confidence >= CONF_TH) & credit_gate & stress_gate
SELL = (composite < SELL_LEVEL_TH) & (comp_trend_series < 0) & (confidence >= CONF_TH) & (~credit_gate) & (~stress_gate)

position = pd.Series(0.0, index=idx)
position[BUY] = 1.0
position[SELL] = -1.0

if hold_last:
    position = position.replace(0.0, np.nan).ffill().fillna(0.0)

# Next-day execution to avoid lookahead bias
position_exec = position.shift(1).fillna(0.0)

spy_ret = spy_close.pct_change().fillna(0.0)
strategy_ret = position_exec * spy_ret

# Trading costs: charged on position changes
cost = (cost_bps / 10000.0) * position_exec.diff().abs().fillna(0.0)
strategy_ret_net = strategy_ret - cost

equity = (1.0 + strategy_ret_net).cumprod() * float(start_equity)
equity_spy = (1.0 + spy_ret).cumprod() * float(start_equity)

total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
cagr = annualized_return(equity / equity.iloc[0])
mdd = max_drawdown(equity)
vol = annualized_vol(strategy_ret_net)
sharpe = sharpe_ratio(strategy_ret_net)

pos_change = position_exec.diff().fillna(0.0)
trade_count = int((pos_change != 0).sum())

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Strategy Total Return", f"{total_return*100:.1f}%")
m2.metric("Strategy CAGR", f"{cagr*100:.1f}%" if not np.isnan(cagr) else "n/a")
m3.metric("Max Drawdown", f"{mdd*100:.1f}%" if not np.isnan(mdd) else "n/a")
m4.metric("Ann. Volatility", f"{vol*100:.1f}%" if not np.isnan(vol) else "n/a")
m5.metric("Sharpe (rf=0)", f"{sharpe:.2f}" if not np.isnan(sharpe) else "n/a")
m6.metric("# Position Changes", f"{trade_count}")

plot_bt = pd.DataFrame({
    "Date": idx,
    "Strategy Equity": equity.values,
    "SPY Buy&Hold Equity": equity_spy.values
}).dropna()

base_bt = alt.Chart(plot_bt).encode(x="Date:T")
line_strat = base_bt.mark_line().encode(y=alt.Y("Strategy Equity:Q", title="Equity ($)"))
line_bh = base_bt.mark_line(strokeDash=[6, 3]).encode(y="SPY Buy&Hold Equity:Q")
st.altair_chart((line_strat + line_bh).interactive(), use_container_width=True)

st.subheader("Recent Signals (last 80 bars)")
recent = pd.DataFrame({
    "SPY Close": spy_close.reindex(idx),
    "Composite": composite,
    "Composite Trend (10d)": comp_trend_series,
    "Confidence": confidence,
    "Credit Gate": credit_gate.astype(int),
    "Stress Gate": stress_gate.astype(int),
    "BUY Signal": BUY.astype(int),
    "SELL Signal": SELL.astype(int),
    "Position (exec)": position_exec
}).tail(80)
st.dataframe(recent, use_container_width=True)

with st.expander("What this is / is not"):
    st.write(
        """
        - This is a locked, rules-based regime model (not optimized).
        - Composite uses ratios + credit + leadership + breadth proxy.
        - BUY/SELL are gated to reduce whipsaws.
        - Backtest uses next-day execution (signal today -> position tomorrow) to reduce lookahead.
        """
    )
