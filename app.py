
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="Multi-Timeframe Heat & Regime Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Multi-Timeframe Heat & Regime Dashboard")
st.caption(
    "Hourly / 2-Hour / Daily / Weekly overheating, divergence, stacked context, and conditioned historical analogs."
)


# ----------------------------- Config -----------------------------
TIMEFRAME_CONFIG = {
    "1H": {"yf_interval": "60m", "period": "730d", "forward_bars": [1, 3, 6, 12]},
    "2H": {"yf_interval": "60m", "period": "730d", "forward_bars": [1, 2, 4, 8]},
    "Daily": {"yf_interval": "1d", "period": "20y", "forward_bars": [1, 2, 5, 10]},
    "Weekly": {"yf_interval": "1d", "period": "20y", "forward_bars": [1, 2, 4, 8]},
}

STATE_ORDER = [
    "Oversold",
    "Cooling",
    "Neutral",
    "Warm",
    "Overheated",
    "Overheated & Rolling",
    "Overheated but Supported",
]

REGIME_ORDER = [
    "Weak",
    "Fading",
    "Neutral",
    "Strong",
    "Strong & Extended",
]

TF_TO_HIGHER = {"1H": "2H", "2H": "Daily", "Daily": "Weekly", "Weekly": None}


# ----------------------------- Utility functions -----------------------------
def safe_series(x, idx):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        return x["Close"] if "Close" in x.columns else x.iloc[:, 0]
    if x is None:
        return pd.Series(index=idx, dtype=float)
    return pd.Series(x, index=idx) if not isinstance(x, pd.Series) else x


def normalize_rolling(series: pd.Series, window: int = 252) -> pd.Series:
    roll_min = series.rolling(window, min_periods=max(20, window // 5)).min()
    roll_max = series.rolling(window, min_periods=max(20, window // 5)).max()
    out = 100 * (series - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    return out.clip(0, 100)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


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
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(n, min_periods=max(5, n // 2)).mean()
    mad = (tp - sma).abs().rolling(n, min_periods=max(5, n // 2)).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def bb_pct(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=max(5, n // 2)).mean()
    sd = close.rolling(n, min_periods=max(5, n // 2)).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def anchored_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    dates = pd.to_datetime(df.index).tz_localize(None).date
    for d, part in df.groupby(dates):
        typical = (part["High"] + part["Low"] + part["Close"]) / 3.0
        pv = typical * part["Volume"].fillna(0)
        out.loc[part.index] = pv.cumsum() / part["Volume"].fillna(0).cumsum().replace(0, np.nan)
    return out


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical * df["Volume"].fillna(0)
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

    bear_div = hh_price & (~hh_osc) & (price > price.shift(lookback // 2)) & (osc < osc.shift(lookback // 2))
    bull_div = ll_price & (~ll_osc) & (price < price.shift(lookback // 2)) & (osc > osc.shift(lookback // 2))
    out = pd.Series("None", index=price.index)
    out.loc[bear_div] = "Bearish"
    out.loc[bull_div] = "Bullish"
    return out


def bucket_heat(x: float) -> str:
    if pd.isna(x):
        return "Neutral"
    if x >= 85:
        return "Overheated"
    if x >= 68:
        return "Warm"
    if x <= 15:
        return "Oversold"
    if x <= 32:
        return "Cooling"
    return "Neutral"


def bucket_regime(x: float, ext: float) -> str:
    if pd.isna(x):
        return "Neutral"
    if x >= 70 and ext >= 70:
        return "Strong & Extended"
    if x >= 60:
        return "Strong"
    if x <= 35:
        return "Weak"
    if x < 50:
        return "Fading"
    return "Neutral"


def traffic_emoji(state: str) -> str:
    mapping = {
        "Overheated": "🔴",
        "Overheated & Rolling": "🔴",
        "Overheated but Supported": "🟠",
        "Warm": "🟡",
        "Neutral": "⚪",
        "Cooling": "🟢",
        "Oversold": "🟢",
        "Strong": "🟢",
        "Strong & Extended": "🟡",
        "Fading": "🟡",
        "Weak": "🔴",
    }
    return mapping.get(state, "⚪")


# ----------------------------- Data loading -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"]).copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    return df


def to_2h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h.copy()
    out = pd.DataFrame()
    out["Open"] = df_1h["Open"].resample("2h").first()
    out["High"] = df_1h["High"].resample("2h").max()
    out["Low"] = df_1h["Low"].resample("2h").min()
    out["Close"] = df_1h["Close"].resample("2h").last()
    out["Volume"] = df_1h["Volume"].resample("2h").sum(min_count=1)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily.empty:
        return df_daily.copy()
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
    if x.empty:
        return x

    x["EMA10"] = ema(x["Close"], 10)
    x["EMA20"] = ema(x["Close"], 20)
    x["SMA50"] = x["Close"].rolling(50, min_periods=10).mean()

    x["TSI"], x["TSI_signal"] = tsi(x["Close"], 25, 13, 7)
    x["TSI_short"], _ = tsi(x["Close"], 7, 4, 7)
    x["CCI"] = cci(x, 20)
    x["BBPct"] = bb_pct(x["Close"], 20, 2.0)

    if tf in ["1H", "2H"]:
        vwap = anchored_intraday_vwap(x)
    else:
        vwap = rolling_vwap(x, 20)
    x["VWAP"] = vwap

    x["Dist_EMA10"] = 100 * (x["Close"] / x["EMA10"] - 1)
    x["Dist_VWAP"] = 100 * (x["Close"] / x["VWAP"] - 1)
    x["ROC12"] = 100 * x["Close"].pct_change(12)
    x["ATR14"] = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - x["Close"].shift()).abs(),
            (x["Low"] - x["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1).rolling(14, min_periods=5).mean()

    # Normalize to 0..100 heat components
    window = 252 if tf in ["Daily", "Weekly"] else 140
    x["TSI_heat"] = normalize_rolling(x["TSI"], window)
    x["CCI_heat"] = normalize_rolling(x["CCI"], window)
    x["BB_heat"] = (x["BBPct"] * 100).clip(0, 100)
    x["Stretch_heat"] = normalize_rolling(x["Dist_EMA10"] + 0.5 * x["Dist_VWAP"].fillna(0), window)

    # Core oscillator: keep TSI shape visible, add stretch/heat context
    x["Heat_osc"] = (
        0.50 * x["TSI_heat"]
        + 0.20 * x["CCI_heat"]
        + 0.20 * x["BB_heat"]
        + 0.10 * x["Stretch_heat"]
    )
    x["Heat_signal"] = ema(x["Heat_osc"], 5)
    x["Heat_slope"] = slope(x["Heat_osc"], 3)
    x["TSI_slope"] = slope(x["TSI"], 3)
    x["Divergence"] = recent_divergence(x["Close"], x["Heat_osc"], 10 if tf in ["1H", "2H"] else 6)

    x["Heat_bucket"] = x["Heat_osc"].apply(bucket_heat)
    x["Regime_bucket"] = [
        bucket_regime(h, e) for h, e in zip(x["TSI_heat"].fillna(50), x["Stretch_heat"].fillna(50))
    ]

    def classify(row):
        hb = row["Heat_bucket"]
        if hb == "Overheated":
            if row["Heat_slope"] < 0 and row["TSI_slope"] < 0:
                return "Overheated & Rolling"
            if row["Heat_signal"] <= row["Heat_osc"]:
                return "Overheated but Supported"
            return "Overheated"
        return hb

    x["State"] = x.apply(classify, axis=1)
    x["Return_1"] = x["Close"].pct_change()
    return x


def align_higher_state(lower_df: pd.DataFrame, higher_df: pd.DataFrame, higher_cols: List[str]) -> pd.DataFrame:
    if higher_df is None or higher_df.empty or lower_df.empty:
        for c in higher_cols:
            lower_df[f"Higher_{c}"] = np.nan
        return lower_df
    tmp = higher_df[higher_cols].copy()
    out = pd.merge_asof(
        lower_df.sort_index(),
        tmp.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    for c in higher_cols:
        out.rename(columns={c: f"Higher_{c}"}, inplace=True)
    return out


# ----------------------------- Analog engine -----------------------------
def analog_summary(df: pd.DataFrame, tf: str, current_idx=None) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    if current_idx is None:
        current_idx = df.index[-1]
    cur = df.loc[current_idx]

    # Keep buckets broad enough to preserve sample size
    mask = (
        (df["Heat_bucket"] == cur["Heat_bucket"])
        & (df["Higher_Regime_bucket"] == cur["Higher_Regime_bucket"])
    )

    # Optional direction filter
    cur_dir = "falling" if cur["Heat_slope"] < 0 else "rising"
    mask &= ((df["Heat_slope"] < 0) if cur_dir == "falling" else (df["Heat_slope"] >= 0))

    # Divergence helps but don't over-filter; only apply if current bar has one
    if cur["Divergence"] in ["Bearish", "Bullish"]:
        mask &= df["Divergence"].eq(cur["Divergence"])

    # Exclude very recent tail bars and current point
    max_forward = max(TIMEFRAME_CONFIG[tf]["forward_bars"])
    base = df.iloc[:-max_forward].copy()
    mask = mask.reindex(base.index).fillna(False)

    matches = base.loc[mask].copy()
    if len(matches) < 20:
        # relax divergence and direction if sample too small
        mask = (
            (base["Heat_bucket"] == cur["Heat_bucket"])
            & (base["Higher_Regime_bucket"] == cur["Higher_Regime_bucket"])
        )
        matches = base.loc[mask].copy()

    stats = {}
    if matches.empty:
        return matches, stats, pd.DataFrame()

    fwd = pd.DataFrame(index=matches.index)
    for h in TIMEFRAME_CONFIG[tf]["forward_bars"]:
        fwd[f"ret_{h}"] = df["Close"].shift(-h).reindex(matches.index) / df["Close"].reindex(matches.index) - 1

    # crude path labels
    labels = []
    for idx in matches.index:
        h1 = TIMEFRAME_CONFIG[tf]["forward_bars"][0]
        hm = TIMEFRAME_CONFIG[tf]["forward_bars"][min(2, len(TIMEFRAME_CONFIG[tf]["forward_bars"]) - 1)]
        path = df.loc[idx:].iloc[: hm + 1]
        if len(path) < hm + 1:
            labels.append(np.nan)
            continue
        ret_end = path["Close"].iloc[-1] / path["Close"].iloc[0] - 1
        max_dd = (path["Close"] / path["Close"].cummax() - 1).min()
        if max_dd > -0.004 and abs(ret_end) < 0.003:
            labels.append("Sideways cool")
        elif ret_end > 0 and max_dd < -0.01:
            labels.append("Corrective dip then resume")
        elif ret_end < 0:
            labels.append("Correction extends")
        else:
            labels.append("Mixed")
    fwd["path"] = labels

    for h in TIMEFRAME_CONFIG[tf]["forward_bars"]:
        s = fwd[f"ret_{h}"].dropna()
        if len(s):
            stats[f"median_{h}"] = float(s.median())
            stats[f"up_pct_{h}"] = float((s > 0).mean())
    stats["sample"] = int(len(fwd))
    path_counts = fwd["path"].value_counts(normalize=True, dropna=True)
    for k, v in path_counts.items():
        stats[f"path_{k}"] = float(v)

    return matches, stats, fwd


def build_prediction_text(tf: str, row: pd.Series, stats: Dict[str, float]) -> str:
    higher = TF_TO_HIGHER.get(tf)
    if not stats:
        return f"{tf}: not enough conditioned analogs yet."

    horizons = TIMEFRAME_CONFIG[tf]["forward_bars"]
    short_h = horizons[min(1, len(horizons) - 1)]
    med = stats.get(f"median_{short_h}", np.nan)
    up = stats.get(f"up_pct_{short_h}", np.nan)

    path_resume = stats.get("path_Corrective dip then resume", 0.0)
    path_extend = stats.get("path_Correction extends", 0.0)
    path_side = stats.get("path_Sideways cool", 0.0)

    direction = "corrective regression likely" if med < 0 else "no strong regression edge"
    path_desc = max(
        [("corrective dip then resume", path_resume), ("correction extends", path_extend), ("sideways cooling", path_side)],
        key=lambda x: x[1],
    )[0]

    return (
        f"{tf}: {row['State'].lower()} with {higher.lower() if higher else 'same-frame'} regime "
        f"'{row['Higher_Regime_bucket'] if 'Higher_Regime_bucket' in row else row['Regime_bucket']}'. "
        f"Conditioned analogs suggest {direction}; median move over the next {short_h} bars is "
        f"{med:.2%}, up-rate {up:.0%}, and the most common path is '{path_desc}'."
    )


# ----------------------------- Plotting -----------------------------
def make_panel(df: pd.DataFrame, tf: str, bars: int = 180) -> go.Figure:
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

    fig.add_trace(go.Scatter(x=d.index, y=d["Heat_osc"], name="Heat Osc", yaxis="y2"))
    fig.add_trace(go.Scatter(x=d.index, y=d["Heat_signal"], name="Signal", yaxis="y2"))
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["TSI_heat"],
            name="TSI Heat",
            yaxis="y2",
            line={"dash": "dot"},
            opacity=0.55,
        )
    )

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(domain=[0, 1], rangeslider_visible=False),
        yaxis=dict(domain=[0.36, 1.0], title="Price"),
        yaxis2=dict(domain=[0, 0.28], title="Heat", range=[0, 100]),
        legend=dict(orientation="h"),
    )
    fig.add_hline(y=85, line_dash="dash", opacity=0.5, yref="y2")
    fig.add_hline(y=68, line_dash="dot", opacity=0.35, yref="y2")
    fig.add_hline(y=32, line_dash="dot", opacity=0.35, yref="y2")
    fig.add_hline(y=15, line_dash="dash", opacity=0.5, yref="y2")
    return fig


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    symbol = st.text_input("Symbol", value="QQQ").strip().upper()
    show_weekly = st.checkbox("Show weekly tab", value=True)
    bars_to_show = st.slider("Bars to show in chart", 80, 400, 180, 20)
    st.markdown("---")
    st.caption(
        "Heat Osc = TSI shape + CCI + Bollinger %B + stretch. "
        "Raw TSI is preserved in the internals, but the decision layer now classifies heat, support, and regression risk."
    )

if not symbol:
    st.stop()

# ----------------------------- Data prep -----------------------------
with st.spinner(f"Loading {symbol} data..."):
    raw_1h = fetch_ohlcv(symbol, "60m", TIMEFRAME_CONFIG["1H"]["period"])
    raw_daily = fetch_ohlcv(symbol, "1d", TIMEFRAME_CONFIG["Daily"]["period"])

if raw_1h.empty or raw_daily.empty:
    st.error("Unable to load data for this symbol.")
    st.stop()

raw_2h = to_2h(raw_1h)
raw_weekly = to_weekly(raw_daily)

feat_1h = compute_features(raw_1h, "1H")
feat_2h = compute_features(raw_2h, "2H")
feat_daily = compute_features(raw_daily, "Daily")
feat_weekly = compute_features(raw_weekly, "Weekly")

feat_1h = align_higher_state(feat_1h, feat_2h, ["Regime_bucket", "State", "Heat_osc", "Heat_slope"])
feat_2h = align_higher_state(feat_2h, feat_daily, ["Regime_bucket", "State", "Heat_osc", "Heat_slope"])
feat_daily = align_higher_state(feat_daily, feat_weekly, ["Regime_bucket", "State", "Heat_osc", "Heat_slope"])
feat_weekly["Higher_Regime_bucket"] = feat_weekly["Regime_bucket"]

frames = {"1H": feat_1h, "2H": feat_2h, "Daily": feat_daily, "Weekly": feat_weekly}
latest = {k: v.iloc[-1] for k, v in frames.items() if not v.empty}

# ----------------------------- Summary cards -----------------------------
st.subheader(f"Current State · {symbol}")

cols = st.columns(4 if show_weekly else 3)
for i, tf in enumerate(["1H", "2H", "Daily"] + (["Weekly"] if show_weekly else [])):
    row = latest[tf]
    with cols[i]:
        st.metric(f"{tf}", f"{traffic_emoji(row['State'])} {row['State']}")
        st.write(f"Heat: **{row['Heat_osc']:.1f}**")
        st.write(f"TSI: **{row['TSI']:.1f}**")
        st.write(f"Slope: **{'Falling' if row['Heat_slope'] < 0 else 'Rising'}**")
        st.write(f"Divergence: **{row['Divergence']}**")

# ----------------------------- Stacked interpretation -----------------------------
st.subheader("Stacked Interpretation")

def stacked_message(latest_rows: Dict[str, pd.Series]) -> str:
    h1 = latest_rows["1H"]
    h2 = latest_rows["2H"]
    d = latest_rows["Daily"]
    w = latest_rows["Weekly"]

    if "Overheated" in h1["State"] and h2["Heat_slope"] >= 0 and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return (
            "1H is overheated, but 2H and Daily remain supportive. "
            "That usually means tactical scalp heat, not confirmed price regression yet."
        )
    if "Overheated" in h2["State"] and d["Regime_bucket"] in ["Strong", "Strong & Extended"]:
        return (
            "2H overheat is showing inside a still-strong Daily regime. "
            "Expect corrective regression risk, but history often favors a bounce/resume path rather than immediate Daily breakdown."
        )
    if "Overheated" in h2["State"] and d["Regime_bucket"] in ["Fading", "Weak"]:
        return (
            "2H overheat is aligned with a fading Daily regime. "
            "That is more dangerous: corrective weakness has a higher chance of spreading upward in timeframe."
        )
    if d["Regime_bucket"] in ["Fading", "Weak"] and w["Regime_bucket"] in ["Fading", "Weak"]:
        return "Daily and Weekly are both deteriorating. This is beyond hourly heat and closer to a regime-fragile state."
    return "Mixed stack. Respect the higher timeframe before assuming the hourly move will produce a durable regression."

st.info(stacked_message(latest))

# ----------------------------- Tabs -----------------------------
tab_names = ["1H", "2H", "Daily"] + (["Weekly"] if show_weekly else [])
tabs = st.tabs(tab_names)

for tab, tf in zip(tabs, tab_names):
    with tab:
        df = frames[tf]
        row = df.iloc[-1]

        left, right = st.columns([2.2, 1.2], vertical_alignment="top")
        with left:
            st.plotly_chart(make_panel(df, tf, bars_to_show), use_container_width=True)
        with right:
            st.markdown(f"### {tf} summary")
            st.write(f"State: **{row['State']}**")
            st.write(f"Higher timeframe regime: **{row.get('Higher_Regime_bucket', row['Regime_bucket'])}**")
            st.write(f"Heat Osc: **{row['Heat_osc']:.1f}**")
            st.write(f"Signal gap: **{row['Heat_osc'] - row['Heat_signal']:.1f}**")
            st.write(f"TSI(25,13,7): **{row['TSI']:.1f}**")
            st.write(f"CCI(20): **{row['CCI']:.1f}**")
            st.write(f"%B(20,2): **{row['BBPct']:.2f}**")
            st.write(f"Dist EMA10: **{row['Dist_EMA10']:.2f}%**")
            st.write(f"Dist VWAP: **{row['Dist_VWAP']:.2f}%**" if pd.notna(row["Dist_VWAP"]) else "Dist VWAP: **n/a**")
            st.write(f"Heat slope: **{'Falling' if row['Heat_slope'] < 0 else 'Rising'}**")
            st.write(f"Divergence: **{row['Divergence']}**")

            matches, stats, fwd = analog_summary(df, tf)
            st.markdown("### Conditioned historical analogs")
            if stats:
                st.write(f"Sample size: **{stats['sample']}**")
                for h in TIMEFRAME_CONFIG[tf]["forward_bars"]:
                    if f"median_{h}" in stats:
                        st.write(
                            f"{h} bars ahead: median **{stats[f'median_{h}']:.2%}**, up-rate **{stats[f'up_pct_{h}']:.0%}**"
                        )
                if "path_Corrective dip then resume" in stats:
                    st.write(f"Corrective dip then resume: **{stats['path_Corrective dip then resume']:.0%}**")
                if "path_Correction extends" in stats:
                    st.write(f"Correction extends: **{stats['path_Correction extends']:.0%}**")
                if "path_Sideways cool" in stats:
                    st.write(f"Sideways cool: **{stats['path_Sideways cool']:.0%}**")
                st.success(build_prediction_text(tf, row, stats))
            else:
                st.warning("Not enough conditioned analogs for the current state bucket yet.")

        if stats:
            hist = fwd.copy()
            hist = hist.dropna(how="all")
            if not hist.empty:
                st.markdown("#### Analog path mix")
                path_mix = hist["path"].value_counts(normalize=True).rename("share").reset_index()
                path_mix.columns = ["Path", "Share"]
                st.dataframe(path_mix, use_container_width=True)

# ----------------------------- Footer notes -----------------------------
with st.expander("How this version differs"):
    st.markdown(
        """
- The chart layout stays familiar.
- The oscillator is no longer a generic blended line trying to do everything.
- It preserves **TSI shape** but adds **CCI, Bollinger %B, and stretch** so the engine can classify:
  - overheat,
  - overheat without confirmed regression,
  - overheat with spreading weakness,
  - higher-timeframe support or deterioration.
- Historical analogs are **conditioned** on the current frame heat bucket plus the higher-timeframe regime bucket.
"""
    )
