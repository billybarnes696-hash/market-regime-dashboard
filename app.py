import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Breadth Ultimate Oscillator", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
EXPECTED_SERIES = ["NYAD", "NYSI", "NYHL", "BPSPX", "SPXA50R", "RSP"]
CUMULATIVE_CORE_SERIES = ["NYAD", "NYSI", "NYHL"]
OPTIONAL_RAW_SERIES = ["BPSPX", "SPXA50R"]
OSCILLATOR_SERIES = CUMULATIVE_CORE_SERIES + OPTIONAL_RAW_SERIES
DEFAULT_WEIGHTS = {"NYAD": 0.26, "NYSI": 0.26, "NYHL": 0.24, "BPSPX": 0.12, "SPXA50R": 0.12}


@dataclass
class SimilarityResult:
    anchor_date: pd.Timestamp
    similarity: float
    forward_5d: float
    forward_10d: float
    forward_20d: float


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_col_name(col: str) -> str:
    c = str(col).strip().upper().replace("$", "")
    c = c.replace(" ", "_")
    if c in {"DATE", "DATETIME", "TIME"}:
        return "Date"
    aliases = {
        "NYAD": "NYAD",
        "ADLINE": "NYAD",
        "ADVANCE_DECLINE": "NYAD",
        "NYSI": "NYSI",
        "SUMMATION": "NYSI",
        "NYHL": "NYHL",
        "NHNL": "NYHL",
        "NEW_HIGHS_NEW_LOWS": "NYHL",
        "RSP": "RSP",
        "RSP_CLOSE": "RSP",
        "BPSPX": "BPSPX",
        "BULLISH_PERCENT": "BPSPX",
        "SPXA50R": "SPXA50R",
        "SPXA_50R": "SPXA50R",
        "PCT_ABOVE_50DMA": "SPXA50R",
        "CLOSE": "Value",
        "LAST": "Value",
        "VALUE": "Value",
    }
    return aliases.get(c, c)


def infer_series_name(df: pd.DataFrame, fallback: str) -> Optional[str]:
    cols = {normalize_col_name(c): c for c in df.columns}
    for series in EXPECTED_SERIES:
        if series in cols:
            return series
    stem = fallback.upper().replace("$", "")
    for series in EXPECTED_SERIES:
        if series in stem:
            return series
    return None


def extract_single_series_from_df(df: pd.DataFrame, fallback_name: str) -> Optional[pd.DataFrame]:
    if df.empty:
        return None

    renamed = {c: normalize_col_name(c) for c in df.columns}
    df = df.rename(columns=renamed).copy()

    date_col = None
    for c in df.columns:
        if c == "Date":
            date_col = c
            break
    if date_col is None:
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    series_name = infer_series_name(df, fallback_name)
    value_col = None

    if series_name and series_name in df.columns:
        value_col = series_name
    elif "Value" in df.columns:
        value_col = "Value"
    else:
        candidates = [c for c in df.columns if c != "Date"]
        numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_candidates and candidates:
            # try coercion
            for c in candidates:
                coerced = safe_numeric(df[c])
                if coerced.notna().sum() > 0:
                    df[c] = coerced
                    numeric_candidates.append(c)
        if len(numeric_candidates) == 1:
            value_col = numeric_candidates[0]
        elif series_name and series_name in candidates:
            value_col = series_name
        elif candidates:
            value_col = candidates[0]

    if value_col is None:
        return None

    if series_name is None:
        series_name = fallback_name.upper()
        if series_name not in EXPECTED_SERIES:
            return None

    out = df[["Date", value_col]].copy()
    out.columns = ["Date", series_name]
    out[series_name] = safe_numeric(out[series_name])
    out = out.dropna(subset=[series_name]).drop_duplicates(subset=["Date"], keep="last")
    return out


def read_csv_like(blob: bytes, filename: str) -> Optional[pd.DataFrame]:
    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            return pd.read_csv(io.BytesIO(blob))
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(blob))
        if lower.endswith(".txt"):
            return pd.read_csv(io.BytesIO(blob), sep=None, engine="python")
    except Exception:
        return None
    return None


def load_series_from_zip(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            blob = zf.read(name)
            df = read_csv_like(blob, name)
            if df is None:
                continue
            parsed = extract_single_series_from_df(df, fallback_name=name)
            if parsed is None:
                continue
            series_name = [c for c in parsed.columns if c != "Date"][0]
            out[series_name] = parsed
    return out


def load_snapshot(file) -> Dict[str, pd.DataFrame]:
    blob = file.getvalue()
    df = read_csv_like(blob, file.name)
    if df is None:
        return {}

    renamed = {c: normalize_col_name(c) for c in df.columns}
    df = df.rename(columns=renamed).copy()
    if "Date" not in df.columns:
        return {}
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    found: Dict[str, pd.DataFrame] = {}
    for s in EXPECTED_SERIES:
        if s in df.columns:
            tmp = df[["Date", s]].copy()
            tmp[s] = safe_numeric(tmp[s])
            tmp = tmp.dropna(subset=[s]).drop_duplicates(subset=["Date"], keep="last")
            if not tmp.empty:
                found[s] = tmp

    # If the snapshot is single-series, still allow it.
    if not found:
        parsed = extract_single_series_from_df(df, fallback_name=file.name)
        if parsed is not None:
            s = [c for c in parsed.columns if c != "Date"][0]
            found[s] = parsed
    return found


def merge_series(series_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for _, sdf in series_dict.items():
        if merged is None:
            merged = sdf.copy()
        else:
            merged = merged.merge(sdf, on="Date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["Date"] + EXPECTED_SERIES)
    merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return merged


def append_snapshot(base: pd.DataFrame, snapshot_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    snap = merge_series(snapshot_dict)
    if snap.empty:
        return base
    if base.empty:
        out = snap
    else:
        out = pd.concat([base, snap], ignore_index=True)
    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return out


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_strength_index(series: pd.Series, long_span: int = 25, short_span: int = 13, signal_span: int = 7) -> Tuple[pd.Series, pd.Series]:
    delta = series.diff()
    abs_delta = delta.abs()
    double_smoothed = ema(ema(delta, long_span), short_span)
    double_abs = ema(ema(abs_delta, long_span), short_span)
    tsi = 100 * double_smoothed / double_abs.replace(0, np.nan)
    signal = ema(tsi, signal_span)
    return tsi, signal


def bollinger_percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom


def adx_from_close(close: pd.Series, window: int = 14) -> pd.Series:
    # Close-only approximation for breadth series. Good enough for persistence scoring.
    high = close * (1 + 0.002)
    low = close * (1 - 0.002)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx


def minmax_to_0_100(series: pd.Series, low: float, high: float) -> pd.Series:
    scaled = (series - low) / (high - low)
    return 100 * scaled.clip(0, 1)


def tanh_to_0_100(series: pd.Series, scale: float = 1.5) -> pd.Series:
    x = np.tanh(series / scale)
    return 50 + 50 * x


def compute_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    tsi, tsi_signal = true_strength_index(series)
    roc20 = series.pct_change(20) * 100
    z63 = rolling_zscore(series, 63)
    roc_z63 = rolling_zscore(roc20, 63)
    pct_b = bollinger_percent_b(series, 20, 2.0)
    adx14 = adx_from_close(series, 14)
    slope10 = series.diff(10)
    slope_z = rolling_zscore(slope10, 63)

    pos_score = 0.6 * tanh_to_0_100(z63, 1.75) + 0.4 * (pct_b * 100).clip(0, 100)
    trend_score = 0.5 * ((tsi + 100) / 2).clip(0, 100) + 0.25 * minmax_to_0_100(adx14, 10, 40) + 0.25 * tanh_to_0_100(slope_z, 1.5)
    accel_score = 0.65 * tanh_to_0_100(roc_z63, 1.5) + 0.35 * tanh_to_0_100(tsi - tsi_signal, 8.0)
    master = 0.35 * pos_score + 0.40 * trend_score + 0.25 * accel_score

    out = pd.DataFrame({
        f"{prefix}_raw": series,
        f"{prefix}_z63": z63,
        f"{prefix}_roc20": roc20,
        f"{prefix}_roc_z63": roc_z63,
        f"{prefix}_pctb": pct_b,
        f"{prefix}_tsi": tsi,
        f"{prefix}_tsi_signal": tsi_signal,
        f"{prefix}_adx14": adx14,
        f"{prefix}_slope10": slope10,
        f"{prefix}_pos_score": pos_score,
        f"{prefix}_trend_score": trend_score,
        f"{prefix}_accel_score": accel_score,
        f"{prefix}_master_score": master,
    })
    return out


def maybe_cumulate(df: pd.DataFrame, mode_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, mode in mode_map.items():
        if col in out.columns and mode == "Daily delta → cumulative":
            out[col] = out[col].fillna(0).cumsum()
    return out


def build_model_frame(df: pd.DataFrame, include_rsp: bool, rsp_weight: float) -> pd.DataFrame:
    model = df[["Date"]].copy()

    weights = DEFAULT_WEIGHTS.copy()
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    for col in OSCILLATOR_SERIES:
        if col in df.columns:
            feats = compute_features(df[col], col)
            model = pd.concat([model, feats], axis=1)

    breadth_master = None
    for col in OSCILLATOR_SERIES:
        score_col = f"{col}_master_score"
        if score_col in model.columns:
            contrib = model[score_col] * weights[col]
            breadth_master = contrib if breadth_master is None else breadth_master.add(contrib, fill_value=0)
    model["Breadth_Ultimate"] = breadth_master

    if include_rsp and "RSP" in df.columns:
        rsp_feats = compute_features(df["RSP"], "RSP")
        model = pd.concat([model, rsp_feats], axis=1)
        model["Ultimate_With_RSP"] = (
            (1 - rsp_weight) * model["Breadth_Ultimate"] + rsp_weight * model["RSP_master_score"]
        )
    else:
        model["Ultimate_With_RSP"] = model["Breadth_Ultimate"]

    model["Ultimate_5d_Change"] = model["Ultimate_With_RSP"].diff(5)
    model["Breadth_5d_Change"] = model["Breadth_Ultimate"].diff(5)

    model["State"] = model["Ultimate_With_RSP"].apply(label_state)
    return model


def label_state(score: float) -> str:
    if pd.isna(score):
        return "Insufficient data"
    if score >= 75:
        return "Expansion"
    if score >= 60:
        return "Constructive"
    if score >= 45:
        return "Repair / Neutral"
    if score >= 30:
        return "Stress"
    return "Washout"


def similarity_backtest(model: pd.DataFrame, price_series: pd.Series, target_col: str, k: int = 20, lookahead: Tuple[int, int, int] = (5, 10, 20), min_gap: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        target_col,
        "Ultimate_5d_Change",
        "Breadth_Ultimate",
        "Breadth_5d_Change",
        "NYAD_master_score",
        "NYSI_master_score",
        "NYHL_master_score",
        "BPSPX_master_score",
        "SPXA50R_master_score",
        "BPSPX_roc_z63",
        "SPXA50R_roc_z63",
    ]
    feature_cols = [c for c in feature_cols if c in model.columns]

    hist = model.copy()
    for h in lookahead:
        hist[f"fwd_{h}d"] = price_series.shift(-h) / price_series - 1
    hist = hist.dropna(subset=feature_cols + [f"fwd_{lookahead[-1]}d"])
    if len(hist) < 100:
        return pd.DataFrame(), hist

    current = hist.iloc[-1]
    candidate_pool = hist.iloc[:-min_gap].copy()
    if candidate_pool.empty:
        return pd.DataFrame(), hist

    X = candidate_pool[feature_cols].copy()
    cur = current[feature_cols]

    mu = X.mean()
    sigma = X.std(ddof=0).replace(0, np.nan)
    Xs = (X - mu) / sigma
    cs = (cur - mu) / sigma
    dist = np.sqrt(((Xs - cs) ** 2).sum(axis=1))

    candidate_pool = candidate_pool.assign(distance=dist)
    candidate_pool = candidate_pool.sort_values("distance").head(k)
    candidate_pool["similarity"] = 1 / (1 + candidate_pool["distance"])

    cols = ["Date", "similarity", "distance"] + [f"fwd_{h}d" for h in lookahead]
    return candidate_pool[cols].reset_index(drop=True), hist


def weighted_forward_stats(analogs: pd.DataFrame, horizons: Tuple[int, int, int] = (5, 10, 20)) -> Dict[str, float]:
    if analogs.empty:
        return {}
    w = analogs["similarity"].clip(lower=1e-9)
    out: Dict[str, float] = {}
    for h in horizons:
        vals = analogs[f"fwd_{h}d"]
        out[f"mean_{h}d"] = np.average(vals, weights=w)
        out[f"median_{h}d"] = float(vals.median())
        out[f"winrate_{h}d"] = np.average((vals > 0).astype(float), weights=w)
        out[f"p10_{h}d"] = float(vals.quantile(0.10))
        out[f"p90_{h}d"] = float(vals.quantile(0.90))
    return out


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def nearest_available_date(dates: pd.Series, selected_date) -> Optional[pd.Timestamp]:
    if dates.empty:
        return None
    d = pd.Timestamp(selected_date)
    valid = dates.dropna().sort_values().unique()
    if len(valid) == 0:
        return None
    idx = np.searchsorted(valid, d.to_datetime64(), side="right") - 1
    if idx < 0:
        return pd.Timestamp(valid[0])
    return pd.Timestamp(valid[idx])


def plot_main(model: pd.DataFrame, source_df: pd.DataFrame, use_rsp: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model["Date"], y=model["Breadth_Ultimate"], name="Breadth Ultimate", mode="lines"))
    if use_rsp and "Ultimate_With_RSP" in model.columns:
        fig.add_trace(go.Scatter(x=model["Date"], y=model["Ultimate_With_RSP"], name="Ultimate + RSP", mode="lines"))
    if use_rsp and "RSP" in source_df.columns:
        rsp_z = rolling_zscore(source_df["RSP"], 63)
        rsp_norm = 50 + 18 * rsp_z.clip(-2.75, 2.75)
        fig.add_trace(go.Scatter(x=source_df["Date"], y=rsp_norm, name="RSP overlay (normalized)", mode="lines", yaxis="y"))
    fig.update_layout(
        title="Ultimate Breadth Oscillator",
        xaxis_title="Date",
        yaxis_title="Score (0-100)",
        hovermode="x unified",
        height=520,
    )
    for level, label in [(75, "Expansion"), (60, "Constructive"), (45, "Neutral/Repair"), (30, "Stress")]:
        fig.add_hline(y=level, line_dash="dot", opacity=0.35, annotation_text=label, annotation_position="right")
    return fig


def plot_components(model: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in OSCILLATOR_SERIES:
        mcol = f"{col}_master_score"
        if mcol in model.columns:
            fig.add_trace(go.Scatter(x=model["Date"], y=model[mcol], name=col, mode="lines"))
    fig.update_layout(title="Per-Series Master Scores", xaxis_title="Date", yaxis_title="Score (0-100)", hovermode="x unified", height=420)
    return fig


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in EXPECTED_SERIES:
        if col in df.columns:
            rows.append({
                "Series": col,
                "Start": df.loc[df[col].notna(), "Date"].min(),
                "End": df.loc[df[col].notna(), "Date"].max(),
                "Rows": int(df[col].notna().sum()),
                "Latest": df[col].dropna().iloc[-1] if df[col].notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


# -----------------------------
# UI
# -----------------------------
st.title("Breadth Ultimate Oscillator for NYAD / NYSI / NYHL")
st.caption("Upload a historical ZIP, then optionally append a daily snapshot. Core breadth lines can be cumulative or built from daily deltas. BPSPX and SPXA50R are treated as raw, non-cumulative breadth overlays.")

with st.sidebar:
    st.header("Inputs")
    hist_zip = st.file_uploader("Historical ZIP", type=["zip"])
    daily_file = st.file_uploader("Daily snapshot", type=["csv", "xlsx", "xls", "txt"])

    st.header("Series mode")
    st.write("Tell the app whether each uploaded core breadth series is already cumulative or should be cumulatively built from daily values.")
    mode_map = {
        s: st.selectbox(
            f"{s} mode",
            ["Already cumulative", "Daily delta → cumulative"],
            index=0,
            key=f"mode_{s}",
        )
        for s in CUMULATIVE_CORE_SERIES + ["RSP"]
    }

    st.header("Model")
    use_bpspx = st.checkbox("Add BPSPX to oscillator + analog", value=True)
    use_spxa50r = st.checkbox("Add SPXA50R to oscillator + analog", value=True)
    include_rsp = st.checkbox("Add RSP to final oscillator", value=True)
    rsp_weight = st.slider("RSP weight in final oscillator", min_value=0.05, max_value=0.40, value=0.20, step=0.05)
    analog_k = st.slider("Number of analogs", min_value=10, max_value=50, value=20, step=5)

    st.header("Historical lookup")
    enable_history_lookup = st.checkbox("Enable calendar lookup", value=True)

    show_raw = st.checkbox("Show raw merged data table", value=False)

if not hist_zip:
    st.info("Upload your historical ZIP to begin. The ZIP can contain separate files for NYAD, NYSI, NYHL, and optionally RSP.")
    st.stop()

try:
    hist_series = load_series_from_zip(hist_zip.getvalue())
except Exception as e:
    st.error(f"Could not read ZIP: {e}")
    st.stop()

if not hist_series:
    st.error("No usable series were found in the ZIP. Include files with a Date column and one value column for NYAD, NYSI, NYHL, or RSP.")
    st.stop()

merged = merge_series(hist_series)
if daily_file is not None:
    snapshot_series = load_snapshot(daily_file)
    merged = append_snapshot(merged, snapshot_series)

merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
merged = maybe_cumulate(merged, mode_map)
merged = merged.reset_index(drop=True)

present = [c for c in CUMULATIVE_CORE_SERIES if c in merged.columns]
missing_breadth = [c for c in CUMULATIVE_CORE_SERIES if c not in merged.columns]
if missing_breadth:
    st.warning(f"Missing breadth series: {', '.join(missing_breadth)}. The model works best when NYAD, NYSI, and NYHL are all present.")

if not use_bpspx and "BPSPX" in merged.columns:
    merged = merged.drop(columns=["BPSPX"])
if not use_spxa50r and "SPXA50R" in merged.columns:
    merged = merged.drop(columns=["SPXA50R"])

model = build_model_frame(merged, include_rsp=include_rsp, rsp_weight=rsp_weight)

left, right = st.columns([1.15, 0.85])
with left:
    st.plotly_chart(plot_main(model, merged, include_rsp), use_container_width=True)
with right:
    current = model.iloc[-1]
    st.subheader("Current score")
    st.metric("Breadth Ultimate", f"{current['Breadth_Ultimate']:.1f}", f"{model['Breadth_Ultimate'].diff().iloc[-1]:.1f}")
    if include_rsp and "Ultimate_With_RSP" in model.columns:
        st.metric("Ultimate + RSP", f"{current['Ultimate_With_RSP']:.1f}", f"{model['Ultimate_With_RSP'].diff().iloc[-1]:.1f}")
    st.metric("State", current["State"])
    for s in OSCILLATOR_SERIES:
        col = f"{s}_master_score"
        if col in model.columns:
            st.metric(f"{s} score", f"{current[col]:.1f}")
    if include_rsp and "RSP_master_score" in model.columns:
        st.metric("RSP score", f"{current['RSP_master_score']:.1f}")

st.plotly_chart(plot_components(model), use_container_width=True)

if enable_history_lookup:
    st.subheader("Historical calendar lookup")
    valid_dates = model["Date"].dropna().sort_values()
    if not valid_dates.empty:
        default_date = valid_dates.iloc[-1].date()
        selected_date = st.date_input(
            "Pick a historical date",
            value=default_date,
            min_value=valid_dates.iloc[0].date(),
            max_value=valid_dates.iloc[-1].date(),
        )
        anchor_date = nearest_available_date(model["Date"], selected_date)
        hist_row = model.loc[model["Date"] == anchor_date].iloc[-1]
        hist_source = merged.loc[merged["Date"] == anchor_date].iloc[-1] if (merged["Date"] == anchor_date).any() else None
        selected_target = "Ultimate_With_RSP" if include_rsp and "Ultimate_With_RSP" in model.columns else "Breadth_Ultimate"

        st.caption(f"Using nearest available row on or before the selected date: **{anchor_date.date()}**")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Historical oscillator", f"{hist_row[selected_target]:.1f}")
        with h2:
            st.metric("Historical breadth score", f"{hist_row['Breadth_Ultimate']:.1f}")
        with h3:
            st.metric("Historical state", hist_row["State"])
        with h4:
            if hist_source is not None and "RSP" in hist_source.index and pd.notna(hist_source["RSP"]):
                st.metric("RSP price", f"{hist_source['RSP']:.2f}")
            else:
                st.metric("RSP price", "n/a")

        detail_rows = []
        for s in OSCILLATOR_SERIES:
            mcol = f"{s}_master_score"
            if mcol in model.columns:
                raw_val = hist_source[s] if hist_source is not None and s in hist_source.index else np.nan
                detail_rows.append({
                    "Series": s,
                    "Raw": raw_val,
                    "Score": hist_row[mcol],
                })
        if include_rsp and "RSP_master_score" in model.columns:
            raw_val = hist_source["RSP"] if hist_source is not None and "RSP" in hist_source.index else np.nan
            detail_rows.append({"Series": "RSP", "Raw": raw_val, "Score": hist_row["RSP_master_score"]})
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

with st.expander("Data coverage"):
    st.dataframe(describe_dataset(merged), use_container_width=True)

# Analog engine
st.subheader("Analog prediction score")
if "RSP" not in merged.columns:
    st.info("Analog forward returns require RSP in the uploaded data.")
else:
    target_col = "Ultimate_With_RSP" if include_rsp and "Ultimate_With_RSP" in model.columns else "Breadth_Ultimate"
    analogs, hist = similarity_backtest(model, merged["RSP"], target_col=target_col, k=analog_k)
    stats = weighted_forward_stats(analogs)

    c1, c2, c3 = st.columns(3)
    if stats:
        with c1:
            st.metric("5d expected", format_pct(stats.get("mean_5d")), f"Win rate {stats.get('winrate_5d', np.nan) * 100:.0f}%")
            st.metric("5d median", format_pct(stats.get("median_5d")))
        with c2:
            st.metric("10d expected", format_pct(stats.get("mean_10d")), f"Win rate {stats.get('winrate_10d', np.nan) * 100:.0f}%")
            st.metric("10d median", format_pct(stats.get("median_10d")))
        with c3:
            st.metric("20d expected", format_pct(stats.get("mean_20d")), f"Win rate {stats.get('winrate_20d', np.nan) * 100:.0f}%")
            st.metric("20d median", format_pct(stats.get("median_20d")))

        summary = {
            "current_state": current["State"],
            "current_score": float(current[target_col]),
            "expected_5d": stats.get("mean_5d"),
            "expected_10d": stats.get("mean_10d"),
            "expected_20d": stats.get("mean_20d"),
            "winrate_5d": stats.get("winrate_5d"),
            "winrate_10d": stats.get("winrate_10d"),
            "winrate_20d": stats.get("winrate_20d"),
            "analogs_used": len(analogs),
        }
        st.code(json.dumps(summary, indent=2, default=str), language="json")
    else:
        st.warning("Not enough overlapping history to build analogs yet. Add more data, especially RSP.")

    if not analogs.empty:
        analogs_display = analogs.copy()
        for h in [5, 10, 20]:
            analogs_display[f"fwd_{h}d"] = analogs_display[f"fwd_{h}d"].map(lambda x: f"{x * 100:.2f}%")
        analogs_display["similarity"] = analogs_display["similarity"].map(lambda x: f"{x:.3f}")
        analogs_display["distance"] = analogs_display["distance"].map(lambda x: f"{x:.3f}")
        st.dataframe(analogs_display, use_container_width=True)

        analog_fig = go.Figure()
        for h in [5, 10, 20]:
            analog_fig.add_trace(go.Histogram(x=analogs[f"fwd_{h}d"] * 100, name=f"{h}d", opacity=0.6))
        analog_fig.update_layout(barmode="overlay", title="Analog forward return distribution", xaxis_title="Forward return %", yaxis_title="Count", height=420)
        st.plotly_chart(analog_fig, use_container_width=True)

if show_raw:
    st.subheader("Merged source data")
    st.dataframe(merged.tail(250), use_container_width=True)

with st.expander("Expected file formats"):
    st.markdown(
        """
        **Historical ZIP**
        - Separate CSV/XLSX/TXT files are easiest.
        - Each file should have a `Date` column.
        - Each file can have either:
          - a single value column, or
          - a named series column like `NYAD`, `NYSI`, `NYHL`, `BPSPX`, `SPXA50R`, `RSP`.

        **Daily snapshot**
        - Can be one file containing `Date, NYAD, NYSI, NYHL, BPSPX, SPXA50R, RSP`.
        - Or a single-series file with `Date` and one value column.

        **Cumulative behavior**
        - For **NYAD / NYSI / NYHL**, if your uploaded numbers are already cumulative, leave the sidebar setting at **Already cumulative**.
        - If you upload daily net values and want the app to build the running line, switch that series to **Daily delta → cumulative**.
        - **BPSPX** and **SPXA50R** are treated as raw, non-cumulative inputs and are never cumulatively summed.
        """
    )
