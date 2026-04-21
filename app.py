
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="Stable Market Engine · TSI Heat Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Stable Market Engine · TSI Heat Dashboard")
st.caption(
    "Alpaca-only feed. Same visual framework, with a TSI-centered heat engine using "
    "CCI, RSI, BB%, VWAP stretch, and price behavior."
)


# ----------------------------- Indicator helpers -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = ema(up, n) / ema(down, n).replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def tsi(close: pd.Series, long: int = 25, short: int = 13, signal: int = 7) -> Tuple[pd.Series, pd.Series]:
    delta = close.diff()
    m1 = ema(delta, long)
    m2 = ema(m1, short)
    a1 = ema(delta.abs(), long)
    a2 = ema(a1, short)
    tsi_val = 100 * m2 / a2.replace(0, np.nan)
    sig = ema(tsi_val, signal)
    return tsi_val, sig


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = tp.rolling(n, min_periods=max(5, n // 2)).mean()
    mad = (tp - sma).abs().rolling(n, min_periods=max(5, n // 2)).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def bb_pct(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=max(5, n // 2)).mean()
    sd = close.rolling(n, min_periods=max(5, n // 2)).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def normalize_rolling(series: pd.Series, window: int) -> pd.Series:
    lo = series.rolling(window, min_periods=max(20, window // 5)).min()
    hi = series.rolling(window, min_periods=max(20, window // 5)).max()
    out = 100 * (series - lo) / (hi - lo).replace(0, np.nan)
    return out.clip(0, 100)


def anchored_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    local_idx = pd.to_datetime(df.index).tz_convert("America/New_York")
    day_keys = pd.Series(local_idx.date, index=df.index)
    for _, part in df.groupby(day_keys):
        tp = (part["High"] + part["Low"] + part["Close"]) / 3.0
        pv = tp * part["Volume"].fillna(0)
        cum_vol = part["Volume"].fillna(0).cumsum().replace(0, np.nan)
        out.loc[part.index] = pv.cumsum() / cum_vol
    return out


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"].fillna(0)
    return pv.rolling(n, min_periods=max(5, n // 2)).sum() / df["Volume"].fillna(0).rolling(
        n, min_periods=max(5, n // 2)
    ).sum().replace(0, np.nan)


def slope(series: pd.Series, n: int = 3) -> pd.Series:
    return series - series.shift(n)


def recent_divergence(price: pd.Series, osc: pd.Series, lookback: int = 10) -> pd.Series:
    hh_price = price == price.rolling(lookback, min_periods=max(5, lookback // 2)).max()
    hh_osc = osc == osc.rolling(lookback, min_periods=max(5, lookback // 2)).max()
    ll_price = price == price.rolling(lookback, min_periods=max(5, lookback // 2)).min()
    ll_osc = osc == osc.rolling(lookback, min_periods=max(5, lookback // 2)).min()

    bear = hh_price & (~hh_osc) & (price > price.shift(max(2, lookback // 2))) & (osc < osc.shift(max(2, lookback // 2)))
    bull = ll_price & (~ll_osc) & (price < price.shift(max(2, lookback // 2))) & (osc > osc.shift(max(2, lookback // 2)))

    out = pd.Series("None", index=price.index)
    out.loc[bear] = "Bearish"
    out.loc[bull] = "Bullish"
    return out


def regime_bucket(tsi_heat: float, stretch_heat: float) -> str:
    if pd.isna(tsi_heat):
        return "Neutral"
    if tsi_heat >= 70 and stretch_heat >= 70:
        return "Strong & Extended"
    if tsi_heat >= 60:
        return "Strong"
    if tsi_heat <= 35 and stretch_heat <= 45:
        return "Weak"
    if tsi_heat < 50:
        return "Fading"
    return "Neutral"


def state_from_row(row: pd.Series) -> str:
    heat = row["Heat_osc"]
    hs = row["Heat_slope"]
    ts = row["TSI_slope"]
    price_chg = row["Price_lookback_ret"]

    if pd.isna(heat):
        return "Neutral"
    if heat >= 85:
        if hs < 0 and ts < 0:
            if abs(price_chg) < 0.0035:
                return "Overheated · Rolling, No Price Damage"
            return "Overheated · Rolling"
        if hs >= 0:
            return "Overheated · Supported"
        return "Overheated"
    if heat >= 68:
        if hs < 0:
            return "Warm · Fading"
        return "Warm"
    if heat <= 15:
        if hs > 0 and ts > 0:
            return "Oversold · Bull Turn"
        return "Oversold"
    if heat <= 32:
        if hs > 0:
            return "Cooling · Improving"
        return "Cooling"
    return "Neutral"


def traffic_emoji(state: str) -> str:
    if "Overheated" in state:
        return "🔴"
    if "Warm" in state:
        return "🟠"
    if "Oversold" in state or "Cooling" in state:
        return "🟢"
    return "🟡"


# ----------------------------- Alpaca loading -----------------------------
def _normalize_bars_df(bars_df: pd.DataFrame) -> pd.DataFrame:
    if bars_df is None or len(bars_df) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if isinstance(bars_df.index, pd.MultiIndex):
        bars_df = bars_df.reset_index(level=0, drop=True)

    bars_df = bars_df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in bars_df.columns]
    bars_df = bars_df[keep].copy()

    idx = pd.to_datetime(bars_df.index, utc=True)
    bars_df.index = idx.tz_convert("America/New_York")
    return bars_df.sort_index().dropna(subset=["Close"])


@st.cache_data(ttl=60, show_spinner=False)
def fetch_alpaca_bars(symbol: str, timeframe_name: str, years_back: int, api_key: str, secret_key: str, feed: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years_back * 366))

    tf_map = {
        "1H": TimeFrame.Hour,
        "1D": TimeFrame.Day,
    }
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe_name],
        start=start,
        end=end,
        feed=feed,
        adjustment="all",
    )
    bars = client.get_stock_bars(req).df
    return _normalize_bars_df(bars)


def to_2h(df_1h: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Open"] = df_1h["Open"].resample("2h").first()
    out["High"] = df_1h["High"].resample("2h").max()
    out["Low"] = df_1h["Low"].resample("2h").min()
    out["Close"] = df_1h["Close"].resample("2h").last()
    out["Volume"] = df_1h["Volume"].resample("2h").sum(min_count=1)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Open"] = df_daily["Open"].resample("W-FRI").first()
    out["High"] = df_daily["High"].resample("W-FRI").max()
    out["Low"] = df_daily["Low"].resample("W-FRI").min()
    out["Close"] = df_daily["Close"].resample("W-FRI").last()
    out["Volume"] = df_daily["Volume"].resample("W-FRI").sum(min_count=1)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


# ----------------------------- Feature engine -----------------------------
def compute_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    x = df.copy()

    x["EMA10"] = ema(x["Close"], 10)
    x["EMA20"] = ema(x["Close"], 20)
    x["SMA50"] = x["Close"].rolling(50, min_periods=10).mean()

    x["TSI"], x["TSI_signal"] = tsi(x["Close"], 25, 13, 7)
    x["RSI"] = rsi(x["Close"], 14)
    x["CCI"] = cci(x, 20)
    x["BBPct"] = bb_pct(x["Close"], 20, 2.0)

    if tf in ["1H", "2H"]:
        x["VWAP"] = anchored_intraday_vwap(x)
        roll_window = 140
        div_lb = 10
        price_lb = 4
    else:
        x["VWAP"] = rolling_vwap(x, 20)
        roll_window = 252
        div_lb = 6
        price_lb = 3

    x["Dist_EMA10"] = 100 * (x["Close"] / x["EMA10"] - 1)
    x["Dist_VWAP"] = 100 * (x["Close"] / x["VWAP"] - 1)
    x["Price_lookback_ret"] = x["Close"] / x["Close"].shift(price_lb) - 1

    x["TSI_heat"] = normalize_rolling(x["TSI"], roll_window)
    x["CCI_heat"] = normalize_rolling(x["CCI"], roll_window)
    x["RSI_heat"] = x["RSI"].clip(0, 100)
    x["BB_heat"] = (x["BBPct"] * 100).clip(0, 100)
    x["Stretch_heat"] = normalize_rolling(x["Dist_EMA10"] + 0.5 * x["Dist_VWAP"].fillna(0), roll_window)

    x["Heat_osc"] = (
        0.40 * x["TSI_heat"]
        + 0.18 * x["CCI_heat"]
        + 0.14 * x["RSI_heat"]
        + 0.16 * x["BB_heat"]
        + 0.12 * x["Stretch_heat"]
    )
    x["Heat_signal"] = ema(x["Heat_osc"], 5)
    x["Heat_slope"] = slope(x["Heat_osc"], 3)
    x["TSI_slope"] = slope(x["TSI"], 3)
    x["Divergence"] = recent_divergence(x["Close"], x["Heat_osc"], div_lb)
    x["State"] = x.apply(state_from_row, axis=1)
    x["Regime_bucket"] = [regime_bucket(a, b) for a, b in zip(x["TSI_heat"], x["Stretch_heat"])]

    return x


def merge_higher_state(lower_df: pd.DataFrame, higher_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["State", "Regime_bucket", "Heat_osc", "Heat_slope", "TSI_slope"]
    tmp = higher_df[cols].copy().sort_index()
    out = pd.merge_asof(
        lower_df.sort_index(),
        tmp,
        left_index=True,
        right_index=True,
        direction="backward",
        suffixes=("", "_higher"),
    )
    out = out.rename(columns={
        "State_higher": "Higher_State",
        "Regime_bucket_higher": "Higher_Regime",
        "Heat_osc_higher": "Higher_Heat",
        "Heat_slope_higher": "Higher_Heat_slope",
        "TSI_slope_higher": "Higher_TSI_slope",
    })
    return out


# ----------------------------- Analogs -----------------------------
FWD_BARS = {"1H": [1, 3, 6, 12], "2H": [1, 2, 4, 8], "Daily": [1, 2, 5, 10], "Weekly": [1, 2, 4, 8]}

def analog_summary(df: pd.DataFrame, tf: str) -> Tuple[dict, pd.DataFrame]:
    max_fwd = max(FWD_BARS[tf])
    if len(df) <= max_fwd + 20:
        return {}, pd.DataFrame()

    base = df.iloc[:-max_fwd].copy()
    cur = df.iloc[-1]

    mask = base["State"].eq(cur["State"]) & base["Higher_Regime"].eq(cur["Higher_Regime"])
    if cur["Divergence"] in ["Bearish", "Bullish"]:
        mask &= base["Divergence"].eq(cur["Divergence"])

    matches = base.loc[mask].copy()
    if len(matches) < 15:
        matches = base.loc[base["State"].eq(cur["State"]) & base["Higher_Regime"].eq(cur["Higher_Regime"])].copy()

    if matches.empty:
        return {}, pd.DataFrame()

    fwd = pd.DataFrame(index=matches.index)
    for h in FWD_BARS[tf]:
        fwd[f"ret_{h}"] = df["Close"].shift(-h).reindex(matches.index) / df["Close"].reindex(matches.index) - 1

    path_labels = []
    mid_h = FWD_BARS[tf][min(2, len(FWD_BARS[tf]) - 1)]
    for idx in matches.index:
        path = df.loc[idx:].iloc[: mid_h + 1]
        if len(path) < mid_h + 1:
            path_labels.append(np.nan)
            continue
        ret_end = path["Close"].iloc[-1] / path["Close"].iloc[0] - 1
        max_dd = (path["Close"] / path["Close"].cummax() - 1).min()
        if max_dd > -0.004 and abs(ret_end) < 0.003:
            path_labels.append("Sideways cool")
        elif ret_end > 0 and max_dd < -0.01:
            path_labels.append("Corrective dip then resume")
        elif ret_end < 0:
            path_labels.append("Correction extends")
        else:
            path_labels.append("Mixed")
    fwd["Path"] = path_labels

    stats = {"sample": int(len(matches))}
    for h in FWD_BARS[tf]:
        s = fwd[f"ret_{h}"].dropna()
        if len(s):
            stats[f"median_{h}"] = float(s.median())
            stats[f"up_pct_{h}"] = float((s > 0).mean())
    mix = fwd["Path"].value_counts(normalize=True)
    for k, v in mix.items():
        stats[f"path_{k}"] = float(v)

    return stats, fwd


def prediction_text(tf: str, row: pd.Series, stats: dict) -> str:
    if not stats:
        return f"{tf}: not enough conditioned analogs yet."

    horizon = FWD_BARS[tf][min(1, len(FWD_BARS[tf]) - 1)]
    med = stats.get(f"median_{horizon}", np.nan)
    up = stats.get(f"up_pct_{horizon}", np.nan)
    resume = stats.get("path_Corrective dip then resume", 0.0)
    extend = stats.get("path_Correction extends", 0.0)
    sideways = stats.get("path_Sideways cool", 0.0)

    if resume >= max(extend, sideways):
        path = "corrective dip then resume"
    elif extend >= sideways:
        path = "correction extends"
    else:
        path = "sideways cool"

    return (
        f"{tf}: {row['State']}. Higher timeframe regime is {row['Higher_Regime']}. "
        f"Conditioned analogs show median move over the next {horizon} bars of {med:.2%}, "
        f"up-rate {up:.0%}, and the most common path is '{path}'."
    )


# ----------------------------- Charts / interpretation -----------------------------
def make_panel(df: pd.DataFrame, bars: int) -> go.Figure:
    d = df.tail(bars).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            name="Price",
            yaxis="y",
        )
    )
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA20"], name="EMA20", yaxis="y"))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], name="SMA50", yaxis="y"))
    fig.add_trace(go.Scatter(x=d.index, y=d["Heat_osc"], name="TSI Heat", yaxis="y2"))
    fig.add_trace(go.Scatter(x=d.index, y=d["Heat_signal"], name="Signal", yaxis="y2"))
    fig.add_trace(go.Scatter(x=d.index, y=d["TSI_heat"], name="TSI Core", yaxis="y2", line={"dash": "dot"}, opacity=0.55))

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(domain=[0, 1], rangeslider_visible=False),
        yaxis=dict(domain=[0.36, 1.0], title="Price"),
        yaxis2=dict(domain=[0, 0.28], title="Heat", range=[0, 100]),
        legend=dict(orientation="h"),
    )
    for y in [15, 32, 68, 85]:
        fig.add_hline(y=y, yref="y2", line_dash="dash", opacity=0.35)
    return fig


def stacked_message(rows: Dict[str, pd.Series]) -> str:
    h1, h2, d, w = rows["1H"], rows["2H"], rows["Daily"], rows["Weekly"]

    if "Overheated" in h1["State"] and h2["Heat_slope"] >= 0 and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return "1H is overheated, but 2H and Daily remain supportive. That is tactical heat, not confirmed regression yet."
    if "Overheated" in h2["State"] and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return "2H overheat is showing inside a still-strong Daily regime. Corrective regression risk is up, but bounce/resume is still a common path."
    if "Overheated" in h2["State"] and d["Regime_bucket"] in ["Fading", "Weak"]:
        return "2H overheat is aligned with a fading Daily regime. That raises the odds that weakness spreads upward in timeframe."
    if "Bull Turn" in h1["State"] and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return "1H has made a bull turn while Daily still holds. Watch for the lower timeframe dip to get absorbed by the broader trend."
    if d["Regime_bucket"] in ["Fading", "Weak"] and w["Regime_bucket"] in ["Fading", "Weak"]:
        return "Daily and Weekly are both deteriorating. This is no longer just hourly exhaustion; it is closer to regime fragility."
    return "Mixed stack. Respect the higher timeframe before assuming the hourly signal will produce durable price regression."


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Credentials")
    api_key = st.text_input("Alpaca API key", type="password", value=os.getenv("ALPACA_API_KEY", ""))
    secret_key = st.text_input("Alpaca API secret", type="password", value=os.getenv("ALPACA_SECRET_KEY", ""))
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], index=0)

    st.header("Input")
    symbol = st.text_input("Select symbol", value="QQQ").strip().upper()

    st.header("Settings")
    intraday_years = st.slider("Intraday years (1H/2H)", 1, 2, 2)
    higher_years = st.slider("Daily/Weekly years", 3, 15, 10)
    bars_to_show = st.slider("Bars to show", 100, 400, 180, 20)
    force_refresh = st.checkbox("Force refresh data (clear cache)", value=False)

if force_refresh:
    st.cache_data.clear()

if not symbol:
    st.stop()
if not api_key or not secret_key:
    st.warning("Enter your Alpaca API key and secret to use Alpaca bars.")
    st.stop()

# ----------------------------- Load data -----------------------------
with st.spinner(f"Loading Alpaca 1H bars for {symbol}..."):
    intraday_1h = fetch_alpaca_bars(symbol, "1H", intraday_years, api_key, secret_key, feed)

with st.spinner(f"Loading Alpaca daily bars for {symbol}..."):
    daily = fetch_alpaca_bars(symbol, "1D", higher_years, api_key, secret_key, feed)

if intraday_1h.empty or daily.empty:
    st.error("No data returned from Alpaca. Check symbol, credentials, and feed.")
    st.stop()

intraday_2h = to_2h(intraday_1h)
weekly = to_weekly(daily)

feat_1h = compute_features(intraday_1h, "1H")
feat_2h = compute_features(intraday_2h, "2H")
feat_daily = compute_features(daily, "Daily")
feat_weekly = compute_features(weekly, "Weekly")

feat_1h = merge_higher_state(feat_1h, feat_2h)
feat_2h = merge_higher_state(feat_2h, feat_daily)
feat_daily = merge_higher_state(feat_daily, feat_weekly)
feat_weekly["Higher_State"] = feat_weekly["State"]
feat_weekly["Higher_Regime"] = feat_weekly["Regime_bucket"]

frames = {"1H": feat_1h, "2H": feat_2h, "Daily": feat_daily, "Weekly": feat_weekly}
latest = {k: v.iloc[-1] for k, v in frames.items()}

# ----------------------------- Header / traffic lights -----------------------------
st.subheader("Traffic lights")
c1, c2, c3, c4 = st.columns(4)
for col, tf in zip([c1, c2, c3, c4], ["1H", "2H", "Daily", "Weekly"]):
    row = latest[tf]
    with col:
        st.metric(tf, f"{traffic_emoji(row['State'])} {row['State']}")
        st.caption(f"Regime: {row['Regime_bucket']}")
        st.caption(f"Divergence: {row['Divergence']}")

st.subheader("Combined")
st.info(stacked_message(latest))

# ----------------------------- Tabs -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["1-Hour (Real)", "2-Hour (Real)", "Daily", "Weekly"])

for tab, tf, label in [
    (tab1, "1H", "1-Hour (Real)"),
    (tab2, "2H", "2-Hour (Real)"),
    (tab3, "Daily", "Daily"),
    (tab4, "Weekly", "Weekly"),
]:
    with tab:
        df = frames[tf]
        row = df.iloc[-1]
        stats, paths = analog_summary(df, tf)

        left, right = st.columns([2.2, 1.2], vertical_alignment="top")
        with left:
            st.plotly_chart(make_panel(df, bars_to_show), use_container_width=True)

        with right:
            st.markdown(f"### {label}")
            st.write(f"State: **{row['State']}**")
            st.write(f"Higher timeframe regime: **{row['Higher_Regime']}**")
            st.write(f"Heat oscillator: **{row['Heat_osc']:.1f}**")
            st.write(f"Signal gap: **{(row['Heat_osc'] - row['Heat_signal']):.1f}**")
            st.write(f"TSI(25,13,7): **{row['TSI']:.1f}**")
            st.write(f"RSI(14): **{row['RSI']:.1f}**")
            st.write(f"CCI(20): **{row['CCI']:.1f}**")
            st.write(f"%B(20,2): **{row['BBPct']:.2f}**")
            st.write(f"Dist EMA10: **{row['Dist_EMA10']:.2f}%**")
            if pd.notna(row["Dist_VWAP"]):
                st.write(f"Dist VWAP: **{row['Dist_VWAP']:.2f}%**")
            st.write(f"Heat slope: **{'Falling' if row['Heat_slope'] < 0 else 'Rising'}**")
            st.write(f"TSI slope: **{'Falling' if row['TSI_slope'] < 0 else 'Rising'}**")
            st.write(f"Divergence: **{row['Divergence']}**")

            st.markdown("### Historical analogs")
            if stats:
                st.write(f"Sample size: **{stats['sample']}**")
                for h in FWD_BARS[tf]:
                    if f"median_{h}" in stats:
                        st.write(f"{h} bars ahead: median **{stats[f'median_{h}']:.2%}**, up-rate **{stats[f'up_pct_{h}']:.0%}**")
                if "path_Corrective dip then resume" in stats:
                    st.write(f"Corrective dip then resume: **{stats['path_Corrective dip then resume']:.0%}**")
                if "path_Correction extends" in stats:
                    st.write(f"Correction extends: **{stats['path_Correction extends']:.0%}**")
                if "path_Sideways cool" in stats:
                    st.write(f"Sideways cool: **{stats['path_Sideways cool']:.0%}**")
                st.success(prediction_text(tf, row, stats))
            else:
                st.warning("Not enough conditioned analogs for the current state bucket yet.")

        if not paths.empty:
            st.markdown("#### Analog path mix")
            path_mix = paths["Path"].value_counts(normalize=True).rename("Share").reset_index()
            path_mix.columns = ["Path", "Share"]
            st.dataframe(path_mix, use_container_width=True)

with st.expander("How this version works"):
    st.markdown(
        """
- **Alpaca-only bars** for 1H, 2H, Daily, and Weekly.
- **Same style dashboard**: traffic lights, tabs, price + oscillator, summary panel, analogs.
- **TSI-centered engine** with CCI / RSI / Bollinger %B / VWAP stretch layered in.
- Exhaustion is determined from:
  - TSI level and slope,
  - price stretch,
  - price behavior,
  - and higher timeframe confirmation.
- The dashboard is designed to say things like:
  - **1H overheated but no price damage yet**
  - **2H overheated inside strong Daily**
  - **1H bull turn while Daily holds**
  - **weakness spreading upward in timeframe**
"""
    )
