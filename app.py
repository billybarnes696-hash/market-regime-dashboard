
import io
import json
import math
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Breadth Ultimate Oscillator", layout="wide")

CORE_CUMULATIVE_SERIES = ["NYAD", "NYSI", "NYHL"]
OPTIONAL_RAW_SERIES = ["BPSPX", "SPXA50R"]
PRICE_SERIES = ["RSP"]
ALL_SERIES = CORE_CUMULATIVE_SERIES + OPTIONAL_RAW_SERIES + PRICE_SERIES
DEFAULT_WEIGHTS = {"NYAD": 0.28, "NYSI": 0.28, "NYHL": 0.24, "BPSPX": 0.10, "SPXA50R": 0.10}


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_col_name(col: str) -> str:
    c = str(col).strip().upper().replace("$", "")
    c = c.replace(" ", "_")
    c = c.replace("-", "_")
    aliases = {
        "DATE": "Date",
        "DATETIME": "Date",
        "TIME": "Date",
        "NYAD": "NYAD",
        "ADLINE": "NYAD",
        "ADVANCE_DECLINE": "NYAD",
        "NYSI": "NYSI",
        "SUMMATION": "NYSI",
        "NYHL": "NYHL",
        "NHNL": "NYHL",
        "NEW_HIGHS_NEW_LOWS": "NYHL",
        "BPSPX": "BPSPX",
        "SPXA50R": "SPXA50R",
        "RSP": "RSP",
        "CLOSE": "Value",
        "LAST": "Value",
        "VALUE": "Value",
    }
    return aliases.get(c, c)


def infer_series_name_from_text(text: str) -> Optional[str]:
    upper = text.upper().replace("$", "")
    for s in ALL_SERIES:
        if s in upper:
            return s
    return None


def extract_stockcharts_history(blob: bytes, filename: str) -> Optional[pd.DataFrame]:
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        return None
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    if "DATE" not in lines[1].upper() or "CLOSE" not in lines[1].upper():
        return None

    series_name = infer_series_name_from_text(lines[0] + " " + filename)
    if not series_name:
        return None

    data_text = "\n".join(lines[1:])
    try:
        df = pd.read_csv(io.StringIO(data_text), skipinitialspace=True)
    except Exception:
        return None

    rename = {c: normalize_col_name(c) for c in df.columns}
    df = df.rename(columns=rename).copy()
    if "Date" not in df.columns:
        return None
    value_col = "Value" if "Value" in df.columns else None
    if value_col is None:
        for candidate in [series_name, "CLOSE", "Close", "close"]:
            if candidate in df.columns:
                value_col = candidate
                break
    if value_col is None:
        return None

    out = df[["Date", value_col]].copy()
    out.columns = ["Date", series_name]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[series_name] = safe_numeric(out[series_name])
    out = out.dropna().drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return out


def extract_single_series_from_df(df: pd.DataFrame, fallback_name: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    renamed = {c: normalize_col_name(c) for c in df.columns}
    df = df.rename(columns=renamed).copy()
    if "Date" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    series_name = None
    for s in ALL_SERIES:
        if s in df.columns or s in fallback_name.upper().replace("$", ""):
            series_name = s
            break

    value_col = None
    if series_name and series_name in df.columns:
        value_col = series_name
    elif "Value" in df.columns:
        value_col = "Value"
    else:
        candidates = [c for c in df.columns if c != "Date"]
        for c in candidates:
            temp = safe_numeric(df[c])
            if temp.notna().sum() > 0:
                df[c] = temp
        numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_candidates) == 1:
            value_col = numeric_candidates[0]
        elif numeric_candidates:
            value_col = numeric_candidates[0]

    if value_col is None or series_name is None:
        return None

    out = df[["Date", value_col]].copy()
    out.columns = ["Date", series_name]
    out[series_name] = safe_numeric(out[series_name])
    out = out.dropna(subset=[series_name]).drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return out


def read_csv_like(blob: bytes, filename: str) -> Optional[pd.DataFrame]:
    lower = filename.lower()
    if lower.endswith(".csv"):
        for kwargs in (
            {"skipinitialspace": True},
            {"sep": None, "engine": "python", "skipinitialspace": True},
        ):
            try:
                return pd.read_csv(io.BytesIO(blob), **kwargs)
            except Exception:
                pass
        return None
    if lower.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(blob))
        except Exception:
            return None
    if lower.endswith(".txt"):
        try:
            return pd.read_csv(io.BytesIO(blob), sep=None, engine="python", skipinitialspace=True)
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
            parsed = extract_stockcharts_history(blob, name)
            if parsed is None:
                raw_df = read_csv_like(blob, name)
                parsed = extract_single_series_from_df(raw_df, name)
            if parsed is None:
                continue
            s = [c for c in parsed.columns if c != "Date"][0]
            out[s] = parsed
    return out


def load_snapshot(file) -> Dict[str, pd.DataFrame]:
    blob = file.getvalue()
    # StockCharts symbol table snapshot
    try:
        raw = pd.read_csv(io.BytesIO(blob))
        if {"Symbol", "Close"}.issubset(raw.columns):
            table = raw.copy()
            table["Symbol"] = table["Symbol"].astype(str).str.upper().str.replace("$", "", regex=False)
            today = pd.Timestamp(st.session_state.get("snapshot_date", pd.Timestamp.today().normalize()))
            found = {}
            for s in ALL_SERIES:
                sub = table.loc[table["Symbol"] == s, "Close"]
                if not sub.empty:
                    found[s] = pd.DataFrame({"Date": [today], s: [pd.to_numeric(sub.iloc[0], errors="coerce")]})
            return found
    except Exception:
        pass

    df = read_csv_like(blob, file.name)
    if df is None:
        return {}
    renamed = {c: normalize_col_name(c) for c in df.columns}
    df = df.rename(columns=renamed).copy()
    found: Dict[str, pd.DataFrame] = {}

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        for s in ALL_SERIES:
            if s in df.columns:
                tmp = df[["Date", s]].copy()
                tmp[s] = safe_numeric(tmp[s])
                tmp = tmp.dropna(subset=[s]).drop_duplicates(subset=["Date"], keep="last")
                if not tmp.empty:
                    found[s] = tmp
        if found:
            return found

    parsed = extract_single_series_from_df(df, file.name)
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
        return pd.DataFrame(columns=["Date"] + ALL_SERIES)
    return merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


def append_snapshot(base: pd.DataFrame, snapshot_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    snap = merge_series(snapshot_dict)
    if snap.empty:
        return base
    out = pd.concat([base, snap], ignore_index=True) if not base.empty else snap
    return out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


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
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def minmax_to_0_100(series: pd.Series, low: float, high: float) -> pd.Series:
    scaled = (series - low) / (high - low)
    return 100 * scaled.clip(0, 1)


def tanh_to_0_100(series: pd.Series, scale: float = 1.5) -> pd.Series:
    return 50 + 50 * np.tanh(series / scale)


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

    return pd.DataFrame({
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


def maybe_cumulate(df: pd.DataFrame, mode_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, mode in mode_map.items():
        if col in out.columns and mode == "Daily delta → cumulative":
            out[col] = out[col].fillna(0).cumsum()
    return out


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


def build_model_frame(df: pd.DataFrame, include_rsp: bool, rsp_weight: float, include_optional: List[str]) -> pd.DataFrame:
    model = df[["Date"]].copy()
    active_breadth = [s for s in CORE_CUMULATIVE_SERIES + include_optional if s in df.columns]

    weight_base = {k: DEFAULT_WEIGHTS.get(k, 0.0) for k in active_breadth}
    total = sum(weight_base.values()) or 1.0
    weights = {k: v / total for k, v in weight_base.items()}

    for col in active_breadth:
        feats = compute_features(df[col], col)
        model = pd.concat([model, feats], axis=1)

    breadth_master = None
    for col in active_breadth:
        score_col = f"{col}_master_score"
        if score_col in model.columns:
            contrib = model[score_col] * weights[col]
            breadth_master = contrib if breadth_master is None else breadth_master.add(contrib, fill_value=0)
    model["Breadth_Ultimate"] = breadth_master

    if include_rsp and "RSP" in df.columns:
        rsp_feats = compute_features(df["RSP"], "RSP")
        model = pd.concat([model, rsp_feats], axis=1)
        model["Ultimate_With_RSP"] = (1 - rsp_weight) * model["Breadth_Ultimate"] + rsp_weight * model["RSP_master_score"]
    else:
        model["Ultimate_With_RSP"] = model["Breadth_Ultimate"]

    model["Ultimate_5d_Change"] = model["Ultimate_With_RSP"].diff(5)
    model["Breadth_5d_Change"] = model["Breadth_Ultimate"].diff(5)
    model["State"] = model["Ultimate_With_RSP"].apply(label_state)
    return model


def similarity_backtest(model: pd.DataFrame, price_series: pd.Series, target_col: str, analog_inputs: List[str], k: int = 20, lookahead: Tuple[int, int, int] = (5, 10, 20), min_gap: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [target_col, "Ultimate_5d_Change", "Breadth_Ultimate", "Breadth_5d_Change"]
    for s in analog_inputs:
        score_col = f"{s}_master_score"
        if score_col in model.columns:
            feature_cols.append(score_col)
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
    out = {}
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


def plot_main(model: pd.DataFrame, source_df: pd.DataFrame, include_rsp: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model["Date"], y=model["Breadth_Ultimate"], name="Breadth Ultimate", mode="lines"))
    if include_rsp and "Ultimate_With_RSP" in model.columns:
        fig.add_trace(go.Scatter(x=model["Date"], y=model["Ultimate_With_RSP"], name="Ultimate + RSP", mode="lines"))
    if "RSP" in source_df.columns:
        rsp_z = rolling_zscore(source_df["RSP"], 63)
        rsp_norm = 50 + 18 * rsp_z.clip(-2.75, 2.75)
        fig.add_trace(go.Scatter(x=source_df["Date"], y=rsp_norm, name="RSP overlay (normalized)", mode="lines"))
    fig.update_layout(title="Ultimate Breadth Oscillator", xaxis_title="Date", yaxis_title="Score (0-100)", hovermode="x unified", height=520)
    for level, label in [(75, "Expansion"), (60, "Constructive"), (45, "Neutral/Repair"), (30, "Stress")]:
        fig.add_hline(y=level, line_dash="dot", opacity=0.35, annotation_text=label, annotation_position="right")
    return fig


def plot_components(model: pd.DataFrame, active_inputs: List[str]) -> go.Figure:
    fig = go.Figure()
    for col in active_inputs:
        mcol = f"{col}_master_score"
        if mcol in model.columns:
            fig.add_trace(go.Scatter(x=model["Date"], y=model[mcol], name=col, mode="lines"))
    fig.update_layout(title="Per-Series Master Scores", xaxis_title="Date", yaxis_title="Score (0-100)", hovermode="x unified", height=420)
    return fig


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ALL_SERIES:
        if col in df.columns:
            non_na = df.loc[df[col].notna(), ["Date", col]]
            if non_na.empty:
                continue
            rows.append({
                "Series": col,
                "Start": non_na["Date"].min(),
                "End": non_na["Date"].max(),
                "Rows": int(non_na.shape[0]),
                "Latest": non_na[col].iloc[-1],
            })
    return pd.DataFrame(rows)


def get_lookup_row(df: pd.DataFrame, lookup_date: pd.Timestamp) -> Optional[pd.Series]:
    hist = df[df["Date"] <= lookup_date]
    if hist.empty:
        return None
    return hist.iloc[-1]


st.title("Breadth Ultimate Oscillator")
st.caption("Upload a historical ZIP, then optionally append a daily snapshot. Supports cumulative breadth core plus optional raw BPSPX/SPXA50R overlays.")

with st.sidebar:
    st.header("Inputs")
    hist_zip = st.file_uploader("Historical ZIP", type=["zip"])
    daily_file = st.file_uploader("Daily snapshot", type=["csv", "xlsx", "xls", "txt"])
    snapshot_date = st.date_input("Daily snapshot date", value=pd.Timestamp.today().date())
    st.session_state["snapshot_date"] = pd.Timestamp(snapshot_date)

    st.header("Series mode")
    mode_map = {}
    for s in ALL_SERIES:
        default_idx = 0 if s in CORE_CUMULATIVE_SERIES or s == "RSP" else 0
        mode_map[s] = st.selectbox(
            f"{s} mode",
            ["Already cumulative", "Daily delta → cumulative"] if s in CORE_CUMULATIVE_SERIES else ["Raw / non-cumulative", "Daily delta → cumulative"],
            index=0,
            key=f"mode_{s}"
        )

    st.header("Optional breadth overlays")
    include_bpspx = st.checkbox("Include BPSPX in oscillator", value=True)
    include_spxa50r = st.checkbox("Include SPXA50R in oscillator", value=True)
    use_bpspx_in_analog = st.checkbox("Use BPSPX in analog search", value=True)
    use_spxa50r_in_analog = st.checkbox("Use SPXA50R in analog search", value=True)

    st.header("Model")
    include_rsp = st.checkbox("Add RSP to final oscillator", value=True)
    rsp_weight = st.slider("RSP weight in final oscillator", min_value=0.05, max_value=0.40, value=0.20, step=0.05)
    analog_k = st.slider("Number of analogs", min_value=10, max_value=50, value=20, step=5)
    show_raw = st.checkbox("Show raw merged data table", value=False)

if not hist_zip:
    st.info("Upload your historical ZIP to begin.")
    st.stop()

try:
    hist_series = load_series_from_zip(hist_zip.getvalue())
except Exception as e:
    st.error(f"Could not read ZIP: {e}")
    st.stop()

if not hist_series:
    st.error("No usable series were found in the ZIP.")
    st.stop()

merged = merge_series(hist_series)
if daily_file is not None:
    merged = append_snapshot(merged, load_snapshot(daily_file))
merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
merged = maybe_cumulate(merged, mode_map)

include_optional = []
if include_bpspx and "BPSPX" in merged.columns:
    include_optional.append("BPSPX")
if include_spxa50r and "SPXA50R" in merged.columns:
    include_optional.append("SPXA50R")

model = build_model_frame(merged, include_rsp=include_rsp, rsp_weight=rsp_weight, include_optional=include_optional)
active_inputs = [s for s in CORE_CUMULATIVE_SERIES + include_optional if f"{s}_master_score" in model.columns]

left, right = st.columns([1.15, 0.85])
with left:
    st.plotly_chart(plot_main(model, merged, include_rsp), use_container_width=True)
with right:
    current = model.iloc[-1]
    st.subheader("Current score")
    st.metric("Breadth Ultimate", f"{current['Breadth_Ultimate']:.1f}", f"{model['Breadth_Ultimate'].diff().iloc[-1]:.1f}" if len(model) > 1 and pd.notna(model['Breadth_Ultimate'].diff().iloc[-1]) else None)
    if include_rsp and "Ultimate_With_RSP" in model.columns:
        delta = model["Ultimate_With_RSP"].diff().iloc[-1] if len(model) > 1 else np.nan
        st.metric("Ultimate + RSP", f"{current['Ultimate_With_RSP']:.1f}", f"{delta:.1f}" if pd.notna(delta) else None)
    st.metric("State", current["State"])
    for s in active_inputs:
        col = f"{s}_master_score"
        st.metric(f"{s} score", f"{current[col]:.1f}")
    if "RSP" in merged.columns:
        st.metric("RSP price", f"{merged['RSP'].dropna().iloc[-1]:.2f}")

st.plotly_chart(plot_components(model, active_inputs), use_container_width=True)

with st.expander("Data coverage"):
    st.dataframe(describe_dataset(merged), use_container_width=True)

st.subheader("Historical calendar lookup")
if not model.empty:
    min_date = model["Date"].min().date()
    max_date = model["Date"].max().date()
    lookup_date = st.date_input("Lookup date", value=max_date, min_value=min_date, max_value=max_date, key="lookup_date")
    row = get_lookup_row(pd.concat([model, merged.drop(columns=["Date"], errors="ignore")], axis=1), pd.Timestamp(lookup_date))
    if row is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lookup breadth score", f"{row['Breadth_Ultimate']:.1f}" if pd.notna(row["Breadth_Ultimate"]) else "n/a")
        c2.metric("Lookup ultimate + RSP", f"{row['Ultimate_With_RSP']:.1f}" if pd.notna(row["Ultimate_With_RSP"]) else "n/a")
        c3.metric("Lookup state", row["State"])
        c4.metric("Lookup RSP", f"{row['RSP']:.2f}" if "RSP" in row and pd.notna(row["RSP"]) else "n/a")

        comp_rows = []
        for s in active_inputs:
            sc = row.get(f"{s}_master_score", np.nan)
            if pd.notna(sc):
                comp_rows.append({"Input": s, "Score": round(float(sc), 2), "Raw": row.get(s, np.nan)})
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

st.subheader("Analog prediction score")
if "RSP" not in merged.columns:
    st.info("Analog forward returns require RSP in the uploaded data.")
else:
    analog_inputs = CORE_CUMULATIVE_SERIES.copy()
    if use_bpspx_in_analog and "BPSPX" in include_optional:
        analog_inputs.append("BPSPX")
    if use_spxa50r_in_analog and "SPXA50R" in include_optional:
        analog_inputs.append("SPXA50R")

    target_col = "Ultimate_With_RSP" if include_rsp and "Ultimate_With_RSP" in model.columns else "Breadth_Ultimate"
    analogs, hist = similarity_backtest(model, merged["RSP"], target_col=target_col, analog_inputs=analog_inputs, k=analog_k)
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
            "current_score": float(current[target_col]) if pd.notna(current[target_col]) else None,
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
        st.warning("Not enough overlapping history to build analogs yet.")

    if not analogs.empty:
        display = analogs.copy()
        for h in [5, 10, 20]:
            display[f"fwd_{h}d"] = display[f"fwd_{h}d"].map(lambda x: f"{x * 100:.2f}%")
        display["similarity"] = display["similarity"].map(lambda x: f"{x:.3f}")
        display["distance"] = display["distance"].map(lambda x: f"{x:.3f}")
        st.dataframe(display, use_container_width=True)

if show_raw:
    st.subheader("Merged source data")
    st.dataframe(merged.tail(250), use_container_width=True)

with st.expander("Expected file formats"):
    st.markdown("""
**Historical ZIP**
- Supports StockCharts historical exports like the ones you uploaded.
- Also supports normal CSV/XLSX/TXT files with `Date` plus one series column.

**Daily snapshot**
- Supports StockCharts symbol-table snapshot files with `Symbol` and `Close`.
- Or files with explicit columns like `Date, NYAD, NYSI, NYHL, BPSPX, SPXA50R, RSP`.

**Cumulative behavior**
- NYAD / NYSI / NYHL are usually cumulative in your use case.
- BPSPX / SPXA50R should stay raw / non-cumulative.
- RSP should stay raw price.
""")
