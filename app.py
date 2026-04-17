import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Breadth Oscillator Dashboard", layout="wide")

SERIES = ["NYAD", "NYSI", "NYHL", "NYMO", "RSP"]

def normalize_name(name: str) -> str:
    return str(name).strip().replace("$", "").replace("^", "").replace("-", "").replace("_", "").replace(" ", "").upper()

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(1, int(span)), adjust=False).mean()

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(10, window // 4)).mean()
    std = series.rolling(window, min_periods=max(10, window // 4)).std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).clip(-5, 5)

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / max(1, length), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / max(1, length), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def tsi(series: pd.Series, long_len: int = 25, short_len: int = 13, signal_len: int = 7):
    momentum = series.diff()
    abs_momentum = momentum.abs()
    ds_m = ema(ema(momentum, long_len), short_len)
    ds_a = ema(ema(abs_momentum, long_len), short_len).replace(0, np.nan)
    out = 100 * (ds_m / ds_a)
    sig = ema(out, signal_len)
    return out, sig

def cci(series: pd.Series, length: int = 20) -> pd.Series:
    sma = series.rolling(length, min_periods=max(5, length // 3)).mean()
    mad = series.rolling(length, min_periods=max(5, length // 3)).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    mad = pd.Series(mad, index=series.index).replace(0, np.nan)
    return (series - sma) / (0.015 * mad)

def bb_percent(series: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(length, min_periods=max(5, length // 3)).mean()
    sd = series.rolling(length, min_periods=max(5, length // 3)).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = (upper - lower).replace(0, np.nan)
    return (series - lower) / width

def detect_symbol_table(df: pd.DataFrame) -> bool:
    cols = [normalize_name(c) for c in df.columns]
    return any(c in cols for c in ["SYMBOL", "TICKER"]) and any(c in cols for c in ["CLOSE", "VALUE", "LAST"])

def parse_symbol_table(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    temp = df.copy()
    rename_map = {}
    for c in temp.columns:
        nc = normalize_name(c)
        if nc in ["SYMBOL", "TICKER"]:
            rename_map[c] = "symbol"
        elif nc in ["CLOSE", "VALUE", "LAST"]:
            rename_map[c] = "value"
    temp = temp.rename(columns=rename_map)
    temp["symbol"] = temp["symbol"].astype(str).str.upper().str.replace("$", "", regex=False).str.strip()
    temp["value"] = to_num(temp["value"])
    keep = temp[temp["symbol"].isin(SERIES)][["symbol", "value"]].dropna()
    if keep.empty:
        return pd.DataFrame()
    row = {"Date": pd.to_datetime(snapshot_date)}
    for _, rec in keep.iterrows():
        row[rec["symbol"]] = float(rec["value"])
    return pd.DataFrame([row])

def parse_csv_bytes(raw: bytes, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    if detect_symbol_table(df):
        if snapshot_date is None:
            snapshot_date = pd.Timestamp.today().normalize()
        return parse_symbol_table(df, snapshot_date)

    rename_map = {}
    for c in df.columns:
        nc = normalize_name(c)
        if nc == "DATE":
            rename_map[c] = "Date"
        elif nc in SERIES:
            rename_map[c] = nc
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Date" not in df.columns and len(df.columns) > 0:
        maybe_date = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if maybe_date.notna().sum() > 0:
            df["Date"] = maybe_date

    if "Date" not in df.columns:
        return pd.DataFrame()

    keep_cols = ["Date"] + [c for c in SERIES if c in df.columns]
    out = df[keep_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in SERIES:
        if c in out.columns:
            out[c] = to_num(out[c])
    return out.dropna(subset=["Date"]).sort_values("Date")

def load_uploaded(uploaded_file, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    parts = []

    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    try:
                        piece = parse_csv_bytes(zf.read(name), snapshot_date=snapshot_date)
                        if not piece.empty:
                            parts.append(piece)
                    except Exception:
                        continue
    elif suffix == ".csv":
        piece = parse_csv_bytes(raw, snapshot_date=snapshot_date)
        if not piece.empty:
            parts.append(piece)

    if not parts:
        return pd.DataFrame()

    merged = parts[0]
    for piece in parts[1:]:
        merged = pd.merge(merged, piece, on="Date", how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        for dc in dup_cols:
            base = dc[:-4]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[dc])
            else:
                merged[base] = merged[dc]
        merged = merged.drop(columns=dup_cols)

    return merged.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

def append_snapshot(history: pd.DataFrame, snapshot_file, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    if snapshot_file is None:
        return history.copy()
    snap = load_uploaded(snapshot_file, snapshot_date=snapshot_date)
    if snap.empty:
        return history.copy()
    snap["Date"] = pd.to_datetime(snapshot_date)
    out = pd.concat([history, snap], ignore_index=True)
    return out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

def prepare_series(df: pd.DataFrame, nyad_cumulative: bool, nyhl_cumulative: bool) -> pd.DataFrame:
    work = df.copy().sort_values("Date").reset_index(drop=True)
    for c in SERIES:
        if c in work.columns:
            work[c] = to_num(work[c])
    if "NYAD" in work.columns and not nyad_cumulative:
        work["NYAD"] = work["NYAD"].fillna(0).cumsum()
    if "NYHL" in work.columns and not nyhl_cumulative:
        work["NYHL"] = work["NYHL"].fillna(0).cumsum()
    return work

def build_composite(df: pd.DataFrame, weights: dict[str, float], z_window: int, smooth_span: int) -> pd.DataFrame:
    work = df.copy()
    normed = {}
    for key, wt in weights.items():
        if wt > 0 and key in work.columns:
            normed[key] = rolling_zscore(work[key], z_window)
    if not normed:
        return work

    norm_df = pd.DataFrame(normed, index=work.index)
    active_weight_sum = sum(w for k, w in weights.items() if k in norm_df.columns and w > 0)
    composite_raw = pd.Series(0.0, index=work.index)
    for key, wt in weights.items():
        if key in norm_df.columns and wt > 0:
            composite_raw = composite_raw.add(norm_df[key] * (wt / active_weight_sum), fill_value=0.0)

    work["Breadth_Composite_Raw"] = composite_raw
    work["Breadth_Composite"] = ema(composite_raw, smooth_span)
    return work

def add_oscillators(df: pd.DataFrame, rsi_len: int, tsi_long: int, tsi_short: int, tsi_signal: int, cci_len: int, bb_len: int, bb_std: float) -> pd.DataFrame:
    work = df.copy()
    base = work["Breadth_Composite"]
    work["Composite_RSI"] = rsi(base, rsi_len)
    work["Composite_TSI"], work["Composite_TSI_Signal"] = tsi(base, tsi_long, tsi_short, tsi_signal)
    work["Composite_CCI"] = cci(base, cci_len)
    work["Composite_BBP"] = bb_percent(base, bb_len, bb_std)
    return work

def regime_label(row: pd.Series) -> str:
    rsi_v = row.get("Composite_RSI", np.nan)
    tsi_v = row.get("Composite_TSI", np.nan)
    tsi_sig = row.get("Composite_TSI_Signal", np.nan)
    if pd.isna(rsi_v) or pd.isna(tsi_v) or pd.isna(tsi_sig):
        return "Insufficient data"
    improving = tsi_v > tsi_sig
    if rsi_v < 35:
        return "Repair / Improving" if improving else "Washout / Weakening"
    if rsi_v < 50:
        return "Neutral / Improving" if improving else "Neutral / Regressing"
    if rsi_v < 70:
        return "Constructive / Trending" if improving else "Constructive / Fading"
    return "Expansion / Strong" if improving else "Exhaustion Risk"

def trim_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    end = df["Date"].max()
    start = end - pd.DateOffset(years=years)
    return df[df["Date"] >= start].copy()

st.title("NYAD + NYSI + NYHL + NYMO Breadth Oscillator")
st.caption("Holistic breadth composite with adjustable weights and default oscillator intervals, tracked against RSP.")

with st.sidebar:
    st.header("Uploads")
    historical_file = st.file_uploader("Historical ZIP or CSV", type=["zip", "csv"])
    snapshot_date = st.date_input("Daily snapshot date", value=pd.Timestamp.today().date())
    snapshot_file = st.file_uploader("Daily snapshot (optional)", type=["zip", "csv"])

    st.header("Series handling")
    st.caption("Use these if uploaded NYAD/NYHL are daily values rather than already-cumulative lines.")
    nyad_cumulative = st.checkbox("NYAD already cumulative", value=True)
    nyhl_cumulative = st.checkbox("NYHL already cumulative", value=True)

    st.header("Weights")
    nyad_w = st.slider("NYAD %", 0, 100, 25, 1)
    nysi_w = st.slider("NYSI %", 0, 100, 25, 1)
    nyhl_w = st.slider("NYHL %", 0, 100, 25, 1)
    nymo_w = st.slider("NYMO %", 0, 100, 25, 1)

    st.header("Composite settings")
    z_window = st.slider("Normalization window", 20, 252, 126, 1)
    smooth_span = st.slider("Composite EMA smoothing", 1, 20, 5, 1)

    st.header("Oscillator settings")
    rsi_len = st.slider("RSI length", 2, 50, 14, 1)
    tsi_long = st.slider("TSI long", 2, 60, 25, 1)
    tsi_short = st.slider("TSI short", 2, 40, 13, 1)
    tsi_signal = st.slider("TSI signal", 1, 20, 7, 1)
    cci_len = st.slider("CCI length", 2, 60, 20, 1)
    bb_len = st.slider("BB% length", 2, 60, 20, 1)
    bb_std = st.slider("BB% std dev", 0.5, 4.0, 2.0, 0.1)

    st.header("Display")
    lookback_years = st.slider("Chart lookback (years)", 1, 20, 2, 1)

if historical_file is None:
    st.info("Upload a historical ZIP or CSV to begin.")
    st.stop()

history = load_uploaded(historical_file, snapshot_date=pd.Timestamp(snapshot_date))
if history.empty:
    st.error("Could not parse the uploaded file. Include Date and at least one of NYAD, NYSI, NYHL, NYMO, or RSP.")
    st.stop()

merged = append_snapshot(history, snapshot_file, pd.Timestamp(snapshot_date))
merged = prepare_series(merged, nyad_cumulative=nyad_cumulative, nyhl_cumulative=nyhl_cumulative)

weights = {"NYAD": float(nyad_w), "NYSI": float(nysi_w), "NYHL": float(nyhl_w), "NYMO": float(nymo_w)}
model = build_composite(merged, weights=weights, z_window=z_window, smooth_span=smooth_span)

if "Breadth_Composite" not in model.columns:
    st.error("No composite could be built. Check that at least one breadth series was found and weighted above 0%.")
    st.stop()

model = add_oscillators(model, rsi_len, tsi_long, tsi_short, tsi_signal, cci_len, bb_len, bb_std)
model["State"] = model.apply(regime_label, axis=1)

latest = model.iloc[-1]
view = trim_years(model, lookback_years)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Composite RSI", f"{latest['Composite_RSI']:.1f}" if pd.notna(latest["Composite_RSI"]) else "n/a")
m2.metric("Composite TSI", f"{latest['Composite_TSI']:.1f}" if pd.notna(latest["Composite_TSI"]) else "n/a")
m3.metric("TSI Signal", f"{latest['Composite_TSI_Signal']:.1f}" if pd.notna(latest["Composite_TSI_Signal"]) else "n/a")
m4.metric("Composite CCI", f"{latest['Composite_CCI']:.1f}" if pd.notna(latest["Composite_CCI"]) else "n/a")
m5.metric("Composite BB%", f"{latest['Composite_BBP']:.2f}" if pd.notna(latest["Composite_BBP"]) else "n/a")

st.markdown(f"**Current state:** {latest['State']}")

fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=view["Date"], y=view["Breadth_Composite"], mode="lines", name="Breadth Composite"))
if "RSP" in view.columns:
    fig_main.add_trace(go.Scatter(x=view["Date"], y=view["RSP"], mode="lines", name="RSP", yaxis="y2"))
fig_main.update_layout(
    title="Breadth Composite vs RSP",
    height=460,
    xaxis_title="Date",
    yaxis_title="Breadth Composite",
    yaxis2=dict(title="RSP", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig_main, width="stretch")

left, right = st.columns(2)
with left:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=view["Date"], y=view["Composite_RSI"], mode="lines", name="Composite RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash")
    fig_rsi.add_hline(y=50, line_dash="dot")
    fig_rsi.add_hline(y=30, line_dash="dash")
    fig_rsi.update_layout(title="Composite RSI", height=300)
    st.plotly_chart(fig_rsi, width="stretch")

    fig_cci = go.Figure()
    fig_cci.add_trace(go.Scatter(x=view["Date"], y=view["Composite_CCI"], mode="lines", name="Composite CCI"))
    fig_cci.add_hline(y=100, line_dash="dash")
    fig_cci.add_hline(y=0, line_dash="dot")
    fig_cci.add_hline(y=-100, line_dash="dash")
    fig_cci.update_layout(title="Composite CCI", height=300)
    st.plotly_chart(fig_cci, width="stretch")

with right:
    fig_tsi = go.Figure()
    fig_tsi.add_trace(go.Scatter(x=view["Date"], y=view["Composite_TSI"], mode="lines", name="Composite TSI"))
    fig_tsi.add_trace(go.Scatter(x=view["Date"], y=view["Composite_TSI_Signal"], mode="lines", name="TSI Signal"))
    fig_tsi.add_hline(y=0, line_dash="dot")
    fig_tsi.update_layout(title="Composite TSI", height=300)
    st.plotly_chart(fig_tsi, width="stretch")

    fig_bbp = go.Figure()
    fig_bbp.add_trace(go.Scatter(x=view["Date"], y=view["Composite_BBP"], mode="lines", name="Composite BB%"))
    fig_bbp.add_hline(y=1.0, line_dash="dash")
    fig_bbp.add_hline(y=0.5, line_dash="dot")
    fig_bbp.add_hline(y=0.0, line_dash="dash")
    fig_bbp.update_layout(title="Composite BB%", height=300)
    st.plotly_chart(fig_bbp, width="stretch")

st.subheader("Latest breadth values and weights")
display_df = pd.DataFrame({
    "Series": ["NYAD", "NYSI", "NYHL", "NYMO", "RSP"],
    "Latest Value": [
        latest.get("NYAD", np.nan),
        latest.get("NYSI", np.nan),
        latest.get("NYHL", np.nan),
        latest.get("NYMO", np.nan),
        latest.get("RSP", np.nan),
    ],
    "Weight %": [nyad_w, nysi_w, nyhl_w, nymo_w, np.nan],
})
st.dataframe(display_df, width="stretch")

st.markdown("""
**Requirements / assumptions**
- `NYSI` is treated as already cumulative.
- `NYMO` is treated as daily, non-cumulative.
- `NYAD` and `NYHL` can be either already cumulative or daily values to cumulate, using the sidebar toggles.
- Default weights are 25% each.
- `RSP` is used for comparison on the main chart and is not blended into the breadth composite.
""")
