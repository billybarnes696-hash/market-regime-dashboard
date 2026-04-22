
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


st.set_page_config(
    page_title="Stable Market Engine · TSI Cross Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Stable Market Engine · TSI Cross Dashboard")
st.caption(
    "Alpaca-only feed. Raw TSI(25,13,7) cross is the primary trigger. "
    "CCI, RSI, BB%, VWAP stretch, and price behavior qualify the signal. "
    "1H/2H can be viewed as Real, Proxy-from-Daily, or both."
)


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


def _to_ny_index(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize("America/New_York")
    return idx.tz_convert("America/New_York")


def normalize_frame_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _to_ny_index(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def anchored_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    local_idx = _to_ny_index(df.index)
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


def traffic_emoji(state: str) -> str:
    if state.startswith("PUT"):
        return "🔴"
    if state.startswith("CALL"):
        return "🟢"
    return "🟡"


def ensure_nonempty_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def safe_last_row(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    return df.iloc[-1]


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

    tf_map = {"1H": TimeFrame.Hour, "1D": TimeFrame.Day}
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe_name],
        start=start,
        end=end,
        feed=feed,
        adjustment="all",
    )
    return _normalize_bars_df(client.get_stock_bars(req).df)


def filter_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    local = _to_ny_index(x.index)
    minutes = local.hour * 60 + local.minute
    keep = (minutes >= 570) & (minutes <= 930)
    return x.loc[keep]


def to_2h_session(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h.copy()
    x = filter_regular_hours(df_1h).copy()
    local_idx = _to_ny_index(x.index)
    days = pd.Series(local_idx.date, index=x.index)
    frames = []
    for _, part in x.groupby(days):
        part = part.sort_index().copy()
        local_part = _to_ny_index(part.index)
        minutes = local_part.hour * 60 + local_part.minute
        slot_map = {570: 0, 630: 0, 690: 1, 750: 1, 810: 2, 870: 2, 930: 3}
        part["slot"] = [slot_map.get(m, 99) for m in minutes]
        for _, grp in part.groupby("slot"):
            grp = grp.sort_index()
            if grp.empty or grp["slot"].iloc[0] == 99:
                continue
            row = pd.DataFrame({
                "Open": [grp["Open"].iloc[0]],
                "High": [grp["High"].max()],
                "Low": [grp["Low"].min()],
                "Close": [grp["Close"].iloc[-1]],
                "Volume": [grp["Volume"].sum()],
            }, index=[grp.index[-1]])
            frames.append(row)
    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return pd.concat(frames).sort_index()


def to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Open"] = df_daily["Open"].resample("W-FRI").first()
    out["High"] = df_daily["High"].resample("W-FRI").max()
    out["Low"] = df_daily["Low"].resample("W-FRI").min()
    out["Close"] = df_daily["Close"].resample("W-FRI").last()
    out["Volume"] = df_daily["Volume"].resample("W-FRI").sum(min_count=1)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def build_proxy_intraday_from_daily(daily: pd.DataFrame, mode: str) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    records = []
    anchors = ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"] if mode == "1H" else ["11:30", "13:30", "15:30"]
    reps = len(anchors)
    for ts, row in daily.iterrows():
        base_ts = ts.tz_convert("America/New_York")
        for a in anchors:
            hh, mm = map(int, a.split(":"))
            new_ts = pd.Timestamp(year=base_ts.year, month=base_ts.month, day=base_ts.day, hour=hh, minute=mm, tz="America/New_York")
            records.append({
                "timestamp": new_ts,
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": row["Volume"] / reps if pd.notna(row["Volume"]) else np.nan,
            })
    return pd.DataFrame(records).set_index("timestamp").sort_index()


def classify_state(row: pd.Series) -> str:
    tsi_val, tsi_sig = row["TSI"], row["TSI_signal"]
    tsi_slope, gap = row["TSI_slope"], row["TSI_gap"]
    heat, price_chg = row["Exhaustion_score"], row["Price_lookback_ret"]
    if tsi_val < tsi_sig:
        if heat >= 72:
            return "PUT · Exhausted, No Price Damage" if abs(price_chg) < 0.004 else "PUT · Exhausted"
        return "PUT · Bearish" if (tsi_slope < 0 or gap < 0) else "NEUTRAL · Transition"
    if tsi_val > tsi_sig:
        if heat <= 28 and tsi_slope > 0:
            return "CALL · Oversold Bull Turn"
        return "CALL · Bullish" if (tsi_slope > 0 or gap > 0) else "NEUTRAL · Transition"
    return "NEUTRAL · Transition"


def compute_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    x = df.copy()
    x["EMA10"] = ema(x["Close"], 10)
    x["EMA20"] = ema(x["Close"], 20)
    x["SMA50"] = x["Close"].rolling(50, min_periods=10).mean()
    x["TSI"], x["TSI_signal"] = tsi(x["Close"], 25, 13, 7)
    x["TSI_gap"] = x["TSI"] - x["TSI_signal"]
    x["RSI"] = rsi(x["Close"], 14)
    x["CCI"] = cci(x, 20)
    x["BBPct"] = bb_pct(x["Close"], 20, 2.0)

    if tf in ["1H", "2H", "1H_PROXY", "2H_PROXY"]:
        x["VWAP"] = anchored_intraday_vwap(x)
        roll_window, div_lb, price_lb = 140, 10, 4
    else:
        x["VWAP"] = rolling_vwap(x, 20)
        roll_window, div_lb, price_lb = 252, 6, 3

    x["Dist_EMA10"] = 100 * (x["Close"] / x["EMA10"] - 1)
    x["Dist_VWAP"] = 100 * (x["Close"] / x["VWAP"] - 1)
    x["Price_lookback_ret"] = x["Close"] / x["Close"].shift(price_lb) - 1

    x["TSI_heat"] = normalize_rolling(x["TSI"], roll_window)
    x["CCI_heat"] = normalize_rolling(x["CCI"], roll_window)
    x["RSI_heat"] = x["RSI"].clip(0, 100)
    x["BB_heat"] = (x["BBPct"] * 100).clip(0, 100)
    x["Stretch_heat"] = normalize_rolling(x["Dist_EMA10"] + 0.5 * x["Dist_VWAP"].fillna(0), roll_window)
    raw_exhaust = 0.25*x["TSI_heat"] + 0.25*x["CCI_heat"] + 0.15*x["RSI_heat"] + 0.20*x["BB_heat"] + 0.15*x["Stretch_heat"]
    x["Exhaustion_score"] = ema(raw_exhaust, 4)

    x["TSI_slope"] = slope(x["TSI"], 3)
    x["Exhaustion_slope"] = slope(x["Exhaustion_score"], 3)
    x["Divergence"] = recent_divergence(x["Close"], x["TSI"], div_lb)
    x["Regime_bucket"] = [regime_bucket(a, b) for a, b in zip(x["TSI_heat"], x["Stretch_heat"])]
    x["State"] = x.apply(classify_state, axis=1)
    return x


def merge_higher_state(lower_df: pd.DataFrame, higher_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["State", "Regime_bucket", "TSI", "TSI_signal", "TSI_gap", "TSI_slope"]
    lower = normalize_frame_index(lower_df)
    higher = normalize_frame_index(higher_df[cols].copy())

    # Much more stable than merge_asof for these dashboard joins:
    # align higher-timeframe state to the lower-timeframe index and carry
    # the last known higher-timeframe value forward.
    aligned = higher.reindex(lower.index, method="ffill")

    aligned = aligned.rename(columns={
        "State": "Higher_State",
        "Regime_bucket": "Higher_Regime",
        "TSI": "Higher_TSI",
        "TSI_signal": "Higher_TSI_signal",
        "TSI_gap": "Higher_TSI_gap",
        "TSI_slope": "Higher_TSI_slope",
    })

    return lower.join(aligned)


FWD_BARS = {"1H": [1, 3, 6, 12], "2H": [1, 2, 4, 8], "Daily": [1, 2, 5, 10], "Weekly": [1, 2, 4, 8]}

def analog_summary(df: pd.DataFrame, tf_key: str) -> Tuple[dict, pd.DataFrame]:
    max_fwd = max(FWD_BARS[tf_key])
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
    for h in FWD_BARS[tf_key]:
        fwd[f"ret_{h}"] = df["Close"].shift(-h).reindex(matches.index) / df["Close"].reindex(matches.index) - 1
    return {"sample": int(len(matches))}, fwd


def prediction_text(tf_key: str, row: pd.Series, stats: dict) -> str:
    if not stats:
        return f"{tf_key}: not enough conditioned analogs yet."
    return f"{tf_key}: {row['State']}. Higher timeframe regime is {row['Higher_Regime']}. Sample size {stats['sample']}."


def make_panel(df: pd.DataFrame, bars: int, label: str, overlay_df: pd.DataFrame = None, overlay_label: str = None) -> go.Figure:
    d = df.tail(bars).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price", yaxis="y"))
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA20"], name="EMA20", yaxis="y"))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], name="SMA50", yaxis="y"))
    fig.add_trace(go.Scatter(x=d.index, y=d["TSI"], name=f"{label} TSI", yaxis="y2"))
    fig.add_trace(go.Scatter(x=d.index, y=d["TSI_signal"], name=f"{label} Signal", yaxis="y2"))
    if overlay_df is not None and overlay_label is not None:
        od = overlay_df.tail(bars).copy()
        fig.add_trace(go.Scatter(x=od.index, y=od["TSI"], name=f"{overlay_label} TSI", yaxis="y2", line={"dash": "dot"}))
        fig.add_trace(go.Scatter(x=od.index, y=od["TSI_signal"], name=f"{overlay_label} Signal", yaxis="y2", line={"dash": "dash"}))
    fig.add_trace(go.Scatter(x=d.index, y=d["Exhaustion_score"], name="Exhaustion", yaxis="y3", opacity=0.55))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(domain=[0,1], rangeslider_visible=False),
                      yaxis=dict(domain=[0.42,1.0], title="Price"), yaxis2=dict(domain=[0.18,0.38], title="TSI"),
                      yaxis3=dict(domain=[0.0,0.12], title="Exhaust", range=[0,100]), legend=dict(orientation="h"))
    fig.add_hline(y=0, yref="y2", line_dash="dash", opacity=0.3)
    for y in [15, 32, 68, 85]:
        fig.add_hline(y=y, yref="y3", line_dash="dash", opacity=0.25)
    return fig


def stacked_message(rows: Dict[str, pd.Series]) -> str:
    h1, h2, d, w = rows["1H"], rows["2H"], rows["Daily"], rows["Weekly"]
    if h1["State"].startswith("PUT") and h2["State"].startswith("CALL") and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return "1H has rolled bearish, but 2H and Daily are still supportive."
    if h2["State"].startswith("PUT") and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return "2H TSI is below signal inside a still-strong Daily regime."
    if h2["State"].startswith("PUT") and d["Regime_bucket"] in ["Fading", "Weak"]:
        return "2H bearish cross is aligned with a fading Daily regime."
    if h1["State"].startswith("CALL") and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return "1H has made a bullish turn while Daily still holds."
    if d["State"].startswith("PUT") and w["Regime_bucket"] in ["Fading", "Weak"]:
        return "Daily is bearish and Weekly is no longer supportive."
    return "Mixed stack. Respect the higher timeframe."

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
    intraday_source = st.selectbox("Intraday source", ["Real Alpaca", "Proxy from Daily", "Both (overlay)"], index=0)
    force_refresh = st.checkbox("Force refresh data (clear cache)", value=False)

if force_refresh:
    st.cache_data.clear()
if not symbol:
    st.stop()
if not api_key or not secret_key:
    st.warning("Enter your Alpaca API key and secret to use Alpaca bars.")
    st.stop()

with st.spinner(f"Loading Alpaca 1H bars for {symbol}..."):
    intraday_1h_real = fetch_alpaca_bars(symbol, "1H", intraday_years, api_key, secret_key, feed)
with st.spinner(f"Loading Alpaca daily bars for {symbol}..."):
    daily = fetch_alpaca_bars(symbol, "1D", higher_years, api_key, secret_key, feed)

if intraday_1h_real.empty or daily.empty:
    st.error("No data returned from Alpaca. Check symbol, credentials, and feed.")
    st.stop()

intraday_1h_real = normalize_frame_index(filter_regular_hours(intraday_1h_real))
intraday_2h_real = normalize_frame_index(to_2h_session(intraday_1h_real))
intraday_1h_proxy = normalize_frame_index(build_proxy_intraday_from_daily(daily, "1H"))
intraday_2h_proxy = normalize_frame_index(build_proxy_intraday_from_daily(daily, "2H"))
daily = normalize_frame_index(daily)
weekly = normalize_frame_index(to_weekly(daily))

feat_1h_real = compute_features(intraday_1h_real, "1H")
feat_2h_real = compute_features(intraday_2h_real, "2H")
feat_1h_proxy = compute_features(intraday_1h_proxy, "1H_PROXY")
feat_2h_proxy = compute_features(intraday_2h_proxy, "2H_PROXY")
feat_daily = compute_features(daily, "Daily")
feat_weekly = compute_features(weekly, "Weekly")

if intraday_source == "Real Alpaca":
    feat_1h, feat_2h = feat_1h_real.copy(), feat_2h_real.copy()
    overlay_1h = overlay_2h = None
    active_1h_label, active_2h_label = "Real 1H", "Real 2H"
elif intraday_source == "Proxy from Daily":
    feat_1h, feat_2h = feat_1h_proxy.copy(), feat_2h_proxy.copy()
    overlay_1h = overlay_2h = None
    active_1h_label, active_2h_label = "Proxy 1H", "Proxy 2H"
else:
    feat_1h, feat_2h = feat_1h_real.copy(), feat_2h_real.copy()
    overlay_1h, overlay_2h = feat_1h_proxy.copy(), feat_2h_proxy.copy()
    active_1h_label, active_2h_label = "Real 1H", "Real 2H"

feat_1h = merge_higher_state(feat_1h, feat_2h)
feat_2h = merge_higher_state(feat_2h, feat_daily)
feat_daily = merge_higher_state(feat_daily, feat_weekly)
feat_weekly["Higher_State"] = feat_weekly["State"]
feat_weekly["Higher_Regime"] = feat_weekly["Regime_bucket"]

frames = {
    "1H": ensure_nonempty_frame(feat_1h, "1H"),
    "2H": ensure_nonempty_frame(feat_2h, "2H"),
    "Daily": ensure_nonempty_frame(feat_daily, "Daily"),
    "Weekly": ensure_nonempty_frame(feat_weekly, "Weekly"),
}
latest = {k: safe_last_row(v) for k, v in frames.items()}

missing = [k for k, v in frames.items() if v.empty]
if missing:
    st.warning("Missing or empty frame(s): " + ", ".join(missing) + ". The dashboard will show available frames only.")

st.subheader("Traffic lights")
c1, c2, c3, c4 = st.columns(4)
for col, tf in zip([c1, c2, c3, c4], ["1H", "2H", "Daily", "Weekly"]):
    row = latest[tf]
    with col:
        label_prefix = "Proxy " if (tf in ["1H","2H"] and intraday_source == "Proxy from Daily") else ""
        if row is None:
            st.metric(f"{label_prefix}{tf}", "⚪ No data")
            st.caption("Regime: n/a")
            st.caption("Divergence: n/a")
        else:
            st.metric(f"{label_prefix}{tf}", f"{traffic_emoji(row['State'])} {row['State']}")
            st.caption(f"Regime: {row['Regime_bucket']}")
            st.caption(f"Divergence: {row['Divergence']}")

st.subheader("Combined")
if all(latest[k] is not None for k in ["1H", "2H", "Daily", "Weekly"]):
    st.info(stacked_message(latest))
else:
    st.info("Combined interpretation unavailable until all required frames have data.")

if intraday_source == "Both (overlay)":
    st.markdown("#### Real vs Proxy quick compare")
    c1, c2 = st.columns(2)
    with c1:
        r, p = safe_last_row(feat_1h_real), safe_last_row(feat_1h_proxy)
        if r is not None and p is not None:
            st.write(f"**1H Real:** {r['State']}")
            st.write(f"**1H Proxy:** {p['State']}")
            st.write(f"TSI gap Real / Proxy: **{r['TSI_gap']:.2f} / {p['TSI_gap']:.2f}**")
        else:
            st.write("1H compare unavailable.")
    with c2:
        r, p = safe_last_row(feat_2h_real), safe_last_row(feat_2h_proxy)
        if r is not None and p is not None:
            st.write(f"**2H Real:** {r['State']}")
            st.write(f"**2H Proxy:** {p['State']}")
            st.write(f"TSI gap Real / Proxy: **{r['TSI_gap']:.2f} / {p['TSI_gap']:.2f}**")
        else:
            st.write("2H compare unavailable.")

tab1, tab2, tab3, tab4 = st.tabs(["1-Hour", "2-Hour", "Daily", "Weekly"])
for tab, tf, label, tf_key in [(tab1,"1H",active_1h_label,"1H"),(tab2,"2H",active_2h_label,"2H"),(tab3,"Daily","Daily","Daily"),(tab4,"Weekly","Weekly","Weekly")]:
    with tab:
        df = frames[tf]
        row = safe_last_row(df)
        left, right = st.columns([2.2,1.2], vertical_alignment="top")
        if row is None:
            with left:
                st.info(f"{label}: no data available.")
            with right:
                st.markdown(f"### {label}")
                st.write("No data available for this frame.")
        else:
            stats, paths = analog_summary(df, tf_key)
            with left:
                overlay_df = overlay_1h if (intraday_source=="Both (overlay)" and tf=="1H") else overlay_2h if (intraday_source=="Both (overlay)" and tf=="2H") else None
                overlay_label = "Proxy 1H" if (intraday_source=="Both (overlay)" and tf=="1H") else "Proxy 2H" if (intraday_source=="Both (overlay)" and tf=="2H") else None
                st.plotly_chart(make_panel(df, bars_to_show, label, overlay_df, overlay_label), use_container_width=True)
            with right:
                st.markdown(f"### {label}")
                st.write(f"State: **{row['State']}**")
                st.write(f"Higher timeframe regime: **{row['Higher_Regime']}**")
                st.write(f"TSI(25,13,7): **{row['TSI']:.1f}**")
                st.write(f"TSI Signal: **{row['TSI_signal']:.1f}**")
                st.write(f"TSI Gap: **{row['TSI_gap']:.2f}**")
                st.write(f"Exhaustion score: **{row['Exhaustion_score']:.1f}**")
                if intraday_source == "Both (overlay)" and tf in ["1H","2H"]:
                    comp = safe_last_row(overlay_1h if tf=="1H" else overlay_2h)
                    if comp is not None:
                        st.markdown("### Proxy compare")
                        st.write(f"Proxy state: **{comp['State']}**")
                        st.write(f"Proxy TSI gap: **{comp['TSI_gap']:.2f}**")
                st.markdown("### Historical analogs")
                if stats:
                    st.write(f"Sample size: **{stats['sample']}**")
                    st.success(prediction_text(tf_key, row, stats))
                else:
                    st.warning("Not enough conditioned analogs for the current state bucket yet.")

with st.expander("How this version works"):
    st.markdown("""
- **Primary trigger = raw TSI(25,13,7) cross**
- **Alpaca-only bars** for real 1H, 2H, Daily, and Weekly
- **Proxy 1H / 2H from Daily** can be viewed instead of real intraday, or overlaid for comparison
- **CCI / RSI / BB% / VWAP stretch** qualify whether the cross is exhausted, supported, or weak
- In **Both (overlay)** mode:
  - Real 1H/2H remains the active decision engine
  - Proxy 1H/2H appears as a comparison overlay so you can compare structure vs real-time motion
""")
