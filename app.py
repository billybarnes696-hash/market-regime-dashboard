import io
import json
import math
import zipfile
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Breadth Regime Dashboard", layout="wide")

CORE_CUMULATIVE_SERIES = ["NYAD", "NYSI", "NYHL"]
OPTIONAL_RAW_SERIES = ["BPSPX", "SPXA50R"]
PRICE_SERIES = ["RSP"]
ALL_SERIES = CORE_CUMULATIVE_SERIES + OPTIONAL_RAW_SERIES + PRICE_SERIES
DEFAULT_WEIGHTS = {"NYAD": 0.28, "NYSI": 0.28, "NYHL": 0.24, "BPSPX": 0.10, "SPXA50R": 0.10}
LOOKBACK_CHOICES = list(range(1, 21))

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def normalize_col_name(col: str) -> str:
    c = str(col).strip().upper().replace("$", "")
    c = c.replace("  ", " ").replace("-", " ")
    aliases = {
        "DATE": "Date",
        "DATETIME": "Date",
        "TIME": "Date",
        "CLOSE": "Value",
        "LAST": "Value",
        "VALUE": "Value",
        "NYAD": "NYAD",
        "NYSI": "NYSI",
        "NYHL": "NYHL",
        "BPSPX": "BPSPX",
        "SPXA50R": "SPXA50R",
        "RSP": "RSP",
        "NYMO": "NYMO",
    }
    return aliases.get(c, c)

def infer_series_name_from_text(text: str) -> Optional[str]:
    upper = text.upper().replace("$", "")
    for s in ALL_SERIES + ["NYMO"]:
        if s in upper:
            return s
    return None

def read_stockcharts_history(blob: bytes, filename: str) -> Optional[pd.DataFrame]:
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
    if series_name is None or series_name == "NYMO":
        return None

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[1:])), skipinitialspace=True)
    except Exception:
        return None

    df = df.rename(columns={c: normalize_col_name(c) for c in df.columns}).copy()
    if "Date" not in df.columns:
        return None

    value_col = None
    for candidate in [series_name, "Value", "Close", "CLOSE", "close"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        return None

    out = df[["Date", value_col]].copy()
    out.columns = ["Date", series_name]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[series_name] = safe_numeric(out[series_name])
    out = out.dropna(subset=["Date", series_name]).drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return out

def read_csv_like(blob: bytes, filename: str) -> Optional[pd.DataFrame]:
    lower = filename.lower()
    if lower.endswith(".csv"):
        tries = [
            dict(skipinitialspace=True),
            dict(sep=None, engine="python", skipinitialspace=True),
        ]
        for kwargs in tries:
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

def extract_single_series(df: Optional[pd.DataFrame], fallback_name: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df.rename(columns={c: normalize_col_name(c) for c in df.columns}).copy()
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    series_name = None
    fallback_upper = fallback_name.upper().replace("$", "")
    for s in ALL_SERIES:
        if s in df.columns or s in fallback_upper:
            series_name = s
            break
    if series_name is None:
        return None

    value_col = None
    for candidate in [series_name, "Value"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        others = [c for c in df.columns if c != "Date"]
        numeric_others = []
        for c in others:
            temp = safe_numeric(df[c])
            if temp.notna().sum() > 0:
                df[c] = temp
                numeric_others.append(c)
        if numeric_others:
            value_col = numeric_others[0]
    if value_col is None:
        return None

    out = df[["Date", value_col]].copy()
    out.columns = ["Date", series_name]
    out[series_name] = safe_numeric(out[series_name])
    out = out.dropna(subset=[series_name]).drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return out

def load_series_from_zip(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            blob = zf.read(name)
            parsed = read_stockcharts_history(blob, name)
            if parsed is None:
                parsed = extract_single_series(read_csv_like(blob, name), name)
            if parsed is None:
                continue
            s = [c for c in parsed.columns if c != "Date"][0]
            out[s] = parsed
    return out

def load_snapshot(file, snapshot_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    blob = file.getvalue()
    # StockCharts table snapshot
    try:
        raw = pd.read_csv(io.BytesIO(blob))
        if {"Symbol", "Close"}.issubset(raw.columns):
            raw = raw.copy()
            raw["Symbol"] = raw["Symbol"].astype(str).str.upper().replace("$", "", regex=False)
            found = {}
            for s in ALL_SERIES:
                vals = raw.loc[raw["Symbol"] == s, "Close"]
                if not vals.empty:
                    found[s] = pd.DataFrame({"Date": [snapshot_date], s: [pd.to_numeric(vals.iloc[0], errors="coerce")]})
            if found:
                return found
    except Exception:
        pass

    df = read_csv_like(blob, file.name)
    if df is None:
        return {}

    df = df.rename(columns={c: normalize_col_name(c) for c in df.columns}).copy()
    out: Dict[str, pd.DataFrame] = {}

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for s in ALL_SERIES:
            if s in df.columns:
                temp = df[["Date", s]].copy()
                temp[s] = safe_numeric(temp[s])
                temp = temp.dropna(subset=["Date", s]).drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
                if not temp.empty:
                    out[s] = temp
        if out:
            return out

    parsed = extract_single_series(df, file.name)
    if parsed is not None:
        s = [c for c in parsed.columns if c != "Date"][0]
        if parsed["Date"].isna().all():
            parsed["Date"] = snapshot_date
        out[s] = parsed
    return out

def merge_series(series_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for sdf in series_dict.values():
        merged = sdf.copy() if merged is None else merged.merge(sdf, on="Date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["Date"] + ALL_SERIES)
    return merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

def append_snapshot(base: pd.DataFrame, snapshot_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    snap = merge_series(snapshot_dict)
    if snap.empty:
        return base.copy()
    if base.empty:
        return snap
    out = pd.concat([base, snap], ignore_index=True)
    return out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

def maybe_cumulate(df: pd.DataFrame, mode_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for s, mode in mode_map.items():
        if s not in out.columns:
            continue
        out[s] = safe_numeric(out[s])
        if "cumulative" in mode.lower() and "already" not in mode.lower():
            out[s] = out[s].fillna(0).cumsum()
    return out

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std

def true_strength_index(series: pd.Series, long_span: int = 25, short_span: int = 13, signal_span: int = 7) -> Tuple[pd.Series, pd.Series]:
    delta = series.diff()
    abs_delta = delta.abs()
    sm = ema(ema(delta, long_span), short_span)
    sm_abs = ema(ema(abs_delta, long_span), short_span)
    tsi = 100 * sm / sm_abs.replace(0, np.nan)
    signal = ema(tsi, signal_span)
    return tsi, signal

def bollinger_percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)

def adx_from_close(close: pd.Series, window: int = 14) -> pd.Series:
    close = safe_numeric(close)
    high = close * 1.002
    low = close * 0.998
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    tr_components = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    tr = tr_components.max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()

def percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    def _rank(vals: pd.Series) -> float:
        vals = vals.dropna()
        if len(vals) <= 1:
            return np.nan
        return vals.rank(pct=True).iloc[-1]
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=False)

def normalize_to_score(series: pd.Series, center: float = 50.0, scale: float = 18.0, clip: float = 2.75) -> pd.Series:
    return center + scale * series.clip(-clip, clip)

def compute_series_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = safe_numeric(df[col]).astype(float)
    f = pd.DataFrame(index=df.index)
    f[f"{col}_raw"] = s
    f[f"{col}_z"] = rolling_zscore(s, 126)
    f[f"{col}_roc20"] = s.pct_change(20).replace([np.inf, -np.inf], np.nan)
    f[f"{col}_roc_z"] = rolling_zscore(f[f"{col}_roc20"], 126)
    tsi, tsi_signal = true_strength_index(s, 25, 13, 7)
    f[f"{col}_tsi"] = tsi
    f[f"{col}_tsi_signal"] = tsi_signal
    f[f"{col}_tsi_diff"] = tsi - tsi_signal
    f[f"{col}_pctb"] = bollinger_percent_b(s, 20, 2)
    f[f"{col}_adx"] = adx_from_close(s, 14)
    f[f"{col}_slope10"] = s.diff(10)
    f[f"{col}_slope_z"] = rolling_zscore(f[f"{col}_slope10"], 126)
    f[f"{col}_prank"] = percentile_rank(s, 252)
    f[f"{col}_delta5"] = s.diff(5)
    return f

def score_series_features(features: pd.DataFrame, col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    pos = normalize_to_score(features[f"{col}_z"]) * 0.55 + (features[f"{col}_pctb"].clip(0, 1) * 100) * 0.45
    trend = (
        (50 + (features[f"{col}_tsi"] / 2).clip(-45, 45)) * 0.50
        + normalize_to_score(features[f"{col}_slope_z"], center=50, scale=16, clip=3) * 0.30
        + (features[f"{col}_adx"].clip(0, 50) * 2) * 0.20
    )
    accel = (
        normalize_to_score(features[f"{col}_roc_z"], center=50, scale=16, clip=3) * 0.60
        + normalize_to_score(rolling_zscore(features[f"{col}_tsi_diff"], 126), center=50, scale=12, clip=3) * 0.40
    )
    master = pos * 0.35 + trend * 0.40 + accel * 0.25
    signal = ema(master, 8)
    out[f"{col}_position_score"] = pos
    out[f"{col}_trend_score"] = trend
    out[f"{col}_accel_score"] = accel
    out[f"{col}_master_score"] = master
    out[f"{col}_signal"] = signal
    out[f"{col}_cross"] = np.where(master >= signal, 1, -1)
    out[f"{col}_relrange"] = percentile_rank(master, 252) * 100
    return out

def weighted_average(df: pd.DataFrame, cols: List[str], weights: Dict[str, float]) -> pd.Series:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    w = np.array([weights.get(c.replace("_master_score", ""), 1.0) for c in available], dtype=float)
    w = w / w.sum()
    vals = df[available].astype(float)
    return (vals * w).sum(axis=1)

def classify_state(score: pd.Series, slope5: pd.Series, cross: pd.Series, relrange: pd.Series) -> pd.Series:
    labels = []
    for sc, sl, cr, rr in zip(score.fillna(np.nan), slope5.fillna(np.nan), cross.fillna(np.nan), relrange.fillna(np.nan)):
        if pd.isna(sc):
            labels.append("n/a")
            continue
        improving = (not pd.isna(sl) and sl > 0) or (not pd.isna(cr) and cr > 0)
        regressing = (not pd.isna(sl) and sl < 0) or (not pd.isna(cr) and cr < 0)
        if sc >= 72:
            labels.append("Expansion" if improving else "Exhaustion Risk")
        elif sc >= 60:
            labels.append("Constructive" if improving else "Constructive / Fading")
        elif sc >= 48:
            labels.append("Neutral / Improving" if improving else "Neutral / Regressing" if regressing else "Neutral")
        elif sc >= 38:
            labels.append("Repair / Progressing" if improving else "Repair / Stalling")
        else:
            labels.append("Washout / Reversal Watch" if improving else "Stress / Breakdown")
    return pd.Series(labels, index=score.index)

def assign_regimes(model: pd.DataFrame, target_col: str) -> pd.Series:
    rr = percentile_rank(model[target_col], 252)
    slope = model[target_col].diff(5)
    # Fixed: .fillna(method="ffill") removed in Pandas 2.2+
    tsi_fast, sig_fast = true_strength_index(model[target_col].ffill(), 7, 4, 5)
    out = []
    for a, b, c in zip(rr, slope, tsi_fast - sig_fast):
        if pd.isna(a):
            out.append(np.nan)
        elif a <= 0.15 and (pd.notna(b) and b <= 0):
            out.append("washout")
        elif a <= 0.45 and (pd.notna(b) and b > 0):
            out.append("repair")
        elif a <= 0.80 and (pd.notna(b) and b >= 0) and (pd.notna(c) and c >= 0):
            out.append("constructive")
        elif a > 0.80 and (pd.notna(c) and c < 0):
            out.append("exhaustion")
        else:
            out.append("range")
    return pd.Series(out, index=model.index)

def build_model_frame(source_df: pd.DataFrame, include_optional: List[str], include_rsp: bool, rsp_weight: float) -> pd.DataFrame:
    model = pd.DataFrame({"Date": pd.to_datetime(source_df["Date"])})
    input_series = [s for s in CORE_CUMULATIVE_SERIES + include_optional if s in source_df.columns]
    for s in input_series:
        f = compute_series_features(source_df, s)
        sc = score_series_features(f, s)
        model = pd.concat([model, f, sc], axis=1)

    core_cols = [f"{s}_master_score" for s in input_series if f"{s}_master_score" in model.columns]
    breadth = weighted_average(model, core_cols, DEFAULT_WEIGHTS)
    breadth_signal = ema(breadth, 8)
    model["Breadth_Ultimate"] = breadth
    model["Breadth_Signal"] = breadth_signal
    model["Breadth_Cross"] = np.where(breadth >= breadth_signal, 1, -1)
    model["Breadth_RelRange"] = percentile_rank(breadth, 252) * 100
    model["Breadth_Slope5"] = breadth.diff(5)
    model["State"] = classify_state(model["Breadth_Ultimate"], model["Breadth_Slope5"], model["Breadth_Cross"], model["Breadth_RelRange"])

    if include_rsp and "RSP" in source_df.columns:
        rsp_f = compute_series_features(source_df, "RSP")
        rsp_sc = score_series_features(rsp_f, "RSP")
        model = pd.concat([model, rsp_f, rsp_sc], axis=1)
        model["Ultimate_With_RSP"] = (1 - rsp_weight) * model["Breadth_Ultimate"] + rsp_weight * model["RSP_master_score"]
        model["Ultimate_Signal"] = ema(model["Ultimate_With_RSP"], 8)
        model["Ultimate_Cross"] = np.where(model["Ultimate_With_RSP"] >= model["Ultimate_Signal"], 1, -1)
        model["Ultimate_RelRange"] = percentile_rank(model["Ultimate_With_RSP"], 252) * 100
        model["Ultimate_Slope5"] = model["Ultimate_With_RSP"].diff(5)
        model["Ultimate_State"] = classify_state(model["Ultimate_With_RSP"], model["Ultimate_Slope5"], model["Ultimate_Cross"], model["Ultimate_RelRange"])
    else:
        model["Ultimate_With_RSP"] = model["Breadth_Ultimate"]
        model["Ultimate_Signal"] = model["Breadth_Signal"]
        model["Ultimate_Cross"] = model["Breadth_Cross"]
        model["Ultimate_RelRange"] = model["Breadth_RelRange"]
        model["Ultimate_Slope5"] = model["Breadth_Slope5"]
        model["Ultimate_State"] = model["State"]

    model["RegimeCluster"] = assign_regimes(model, "Ultimate_With_RSP")
    return model

def clip_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = pd.to_datetime(df["Date"]).max()
    start = end - pd.DateOffset(years=years)
    clipped = df[df["Date"] >= start].copy()
    return clipped if not clipped.empty else df.copy()

def plot_main(model: pd.DataFrame, source_df: pd.DataFrame, years: int, include_rsp: bool) -> go.Figure:
    model_view = clip_years(model, years)
    source_view = clip_years(source_df, years)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_view["Date"], y=model_view["Ultimate_With_RSP"], mode="lines", name="Master oscillator"))
    fig.add_trace(go.Scatter(x=model_view["Date"], y=model_view["Ultimate_Signal"], mode="lines", name="Signal line"))
    if include_rsp and "RSP" in source_view.columns:
        rsp_z = rolling_zscore(source_view["RSP"], 63)
        rsp_norm = normalize_to_score(rsp_z, center=50, scale=18, clip=3)
        fig.add_trace(go.Scatter(x=source_view["Date"], y=rsp_norm, mode="lines", name="RSP overlay (normalized)"))
    fig.update_layout(title=f"Master Oscillator ({years}Y view)", xaxis_title="Date", yaxis_title="Score / normalized", hovermode="x unified", height=520)
    for level, label in [(75, "Expansion"), (60, "Constructive"), (48, "Neutral"), (38, "Repair"), (25, "Stress")]:
        fig.add_hline(y=level, line_dash="dot", opacity=0.3, annotation_text=label, annotation_position="right")
    return fig

def plot_components(model: pd.DataFrame, active_inputs: List[str], years: int) -> go.Figure:
    model_view = clip_years(model, years)
    fig = go.Figure()
    for s in active_inputs:
        score_col = f"{s}_master_score"
        sig_col = f"{s}_signal"
        if score_col in model_view.columns:
            fig.add_trace(go.Scatter(x=model_view["Date"], y=model_view[score_col], mode="lines", name=f"{s} score"))
        if sig_col in model_view.columns:
            fig.add_trace(go.Scatter(x=model_view["Date"], y=model_view[sig_col], mode="lines", name=f"{s} signal", visible="legendonly"))
    fig.update_layout(title=f"Component Oscillators ({years}Y view)", xaxis_title="Date", yaxis_title="Score", hovermode="x unified", height=460)
    return fig

def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ALL_SERIES:
        if col not in df.columns:
            continue
        temp = df.loc[df[col].notna(), ["Date", col]]
        if temp.empty:
            continue
        rows.append({"Series": col, "Start": temp["Date"].min(), "End": temp["Date"].max(), "Rows": int(len(temp)), "Latest": float(temp[col].iloc[-1])})
    return pd.DataFrame(rows)

def get_lookup_idx(df: pd.DataFrame, lookup_date: pd.Timestamp) -> Optional[int]:
    mask = df["Date"] <= lookup_date
    if not mask.any():
        return None
    return int(np.where(mask)[0][-1])

def safe_price_at(series: pd.Series, idx: int, horizon: int = 0) -> Optional[float]:
    target = idx + horizon
    if target < 0 or target >= len(series):
        return None
    val = series.iloc[target]
    return None if pd.isna(val) else float(val)

def compute_forward_returns(rsp: pd.Series, horizons: List[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=rsp.index)
    for h in horizons:
        out[f"fwd_{h}d"] = rsp.shift(-h) / rsp - 1
        out[f"fwd_price_{h}d"] = rsp.shift(-h)
    return out

def similarity_backtest(
    model: pd.DataFrame,
    rsp: pd.Series,
    target_col: str,
    analog_inputs: List[str],
    k: int = 20,
    current_idx: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = pd.DataFrame({"Date": model["Date"]}).copy()
    work["target_score"] = safe_numeric(model[target_col])
    work["target_signal"] = safe_numeric(model["Ultimate_Signal"])
    work["relrange"] = safe_numeric(model["Ultimate_RelRange"]) / 100.0
    work["slope5"] = safe_numeric(model["Ultimate_Slope5"])
    work["cross"] = safe_numeric(model["Ultimate_Cross"])
    work["regime_code"] = model["RegimeCluster"].map({"washout": 0, "repair": 1, "range": 2, "constructive": 3, "exhaustion": 4})
    work["rsp"] = safe_numeric(rsp)
    for s in analog_inputs:
        for suffix in ["master_score", "signal", "relrange"]:
            col = f"{s}_{suffix}"
            if col in model.columns:
                work[col] = safe_numeric(model[col])

    fwd = compute_forward_returns(work["rsp"], [5, 10, 20])
    work = pd.concat([work, fwd], axis=1)

    candidate = work.copy().iloc[:-21].reset_index(drop=True)
    target = work.copy().reset_index(drop=True)
    if current_idx is None:
        current_idx = len(target) - 1

    feature_cols = [c for c in work.columns if c not in {"Date", "rsp", "fwd_5d", "fwd_10d", "fwd_20d", "fwd_price_5d", "fwd_price_10d", "fwd_price_20d"}]
    candidate = candidate.dropna(subset=["target_score"])
    target = target.dropna(subset=["target_score"])
    if candidate.empty or current_idx not in target.index:
        return pd.DataFrame(), work

    cs = target.loc[current_idx, feature_cols]
    if cs.isna().all():
        return pd.DataFrame(), work

    # relative-range-aware scaling
    Xs = candidate[feature_cols].apply(pd.to_numeric, errors="coerce")
    cs = pd.to_numeric(cs, errors="coerce")
    keep_cols = [c for c in feature_cols if pd.notna(cs[c]) and Xs[c].notna().sum() > 30]
    if not keep_cols:
        return pd.DataFrame(), work

    Xs = Xs[keep_cols].astype(float)
    cs = cs[keep_cols].astype(float)

    med = Xs.median()
    iqr = (Xs.quantile(0.75) - Xs.quantile(0.25)).replace(0, np.nan)
    std = Xs.std(ddof=0).replace(0, np.nan)
    scale = iqr.fillna(std).replace(0, 1.0).fillna(1.0)

    Xn = ((Xs - med) / scale).replace([np.inf, -np.inf], np.nan)
    csn = ((cs - med) / scale).replace([np.inf, -np.inf], np.nan)

    valid_rows = Xn.notna().all(axis=1)
    Xn = Xn.loc[valid_rows]
    candidates = candidate.loc[valid_rows].copy()
    if Xn.empty:
        return pd.DataFrame(), work

    arr = Xn.to_numpy(dtype=float)
    cur = csn.to_numpy(dtype=float)
    dist = np.sqrt(((arr - cur) ** 2).sum(axis=1))

    # reward same regime and same cross
    regime_penalty = np.where(candidates["regime_code"].to_numpy() == target.loc[current_idx, "regime_code"], 0.0, 0.75)
    cross_penalty = np.where(candidates["cross"].to_numpy() == target.loc[current_idx, "cross"], 0.0, 0.40)
    total_dist = dist + regime_penalty + cross_penalty

    candidates["distance"] = total_dist
    candidates["similarity"] = 1 / (1 + total_dist)
    candidates = candidates.sort_values(["distance", "Date"]).head(k).copy()
    return candidates, work

def weighted_forward_stats(analogs: pd.DataFrame) -> Dict[str, float]:
    if analogs.empty:
        return {}
    w = safe_numeric(analogs["similarity"]).fillna(0).clip(lower=0)
    if w.sum() <= 0:
        w = pd.Series(np.ones(len(analogs)), index=analogs.index)
    w = w / w.sum()
    out: Dict[str, float] = {}
    for h in [5, 10, 20]:
        vals = safe_numeric(analogs[f"fwd_{h}d"])
        mask = vals.notna()
        if mask.sum() == 0:
            continue
        ww = w[mask]
        ww = ww / ww.sum()
        vv = vals[mask]
        out[f"mean_{h}d"] = float((vv * ww).sum())
        out[f"median_{h}d"] = float(vv.median())
        out[f"winrate_{h}d"] = float((ww * (vv > 0).astype(float)).sum())
    return out

def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:.2f}%"

def format_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.{digits}f}"

def analog_distribution_chart(analogs: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    vals = safe_numeric(analogs.get(f"fwd_{horizon}d", pd.Series(dtype=float))).dropna() * 100
    if not vals.empty:
        fig.add_histogram(x=vals, nbinsx=20, name=f"{horizon}d returns")
    fig.update_layout(title=f"Analog return distribution: {horizon}d", xaxis_title="Forward return %", yaxis_title="Count", height=320)
    return fig

st.title("Breadth Regime Dashboard")
st.caption("Cumulative breadth core with optional BPSPX/SPXA50R overlays, TSI-style signal crosses, historical lookup, and analog projections.")

with st.sidebar:
    st.header("Inputs")
    hist_zip = st.file_uploader("Historical ZIP", type=["zip"])
    daily_file = st.file_uploader("Daily snapshot", type=["csv", "xlsx", "xls", "txt"])
    snapshot_date = pd.Timestamp(st.date_input("Daily snapshot date", value=pd.Timestamp.today().date()))
    st.header("Series mode")
    mode_map = {}
    for s in ALL_SERIES:
        opts = ["Already cumulative", "Daily delta → cumulative"] if s in CORE_CUMULATIVE_SERIES else ["Raw / non-cumulative", "Daily delta → cumulative"]
        mode_map[s] = st.selectbox(f"{s} mode", opts, index=0, key=f"mode_{s}")

    st.header("Optional breadth overlays")
    include_bpspx = st.checkbox("Include BPSPX in oscillator", value=True)
    include_spxa50r = st.checkbox("Include SPXA50R in oscillator", value=True)
    use_bpspx_in_analog = st.checkbox("Use BPSPX in analog search", value=True)
    use_spxa50r_in_analog = st.checkbox("Use SPXA50R in analog search", value=True)

    st.header("Display")
    years_master = st.selectbox("Master chart lookback", LOOKBACK_CHOICES, index=1)
    years_components = st.selectbox("Component chart lookback", LOOKBACK_CHOICES, index=1)

    st.header("Model")
    include_rsp = st.checkbox("Blend RSP into final oscillator", value=True)
    rsp_weight = st.slider("RSP blend weight", min_value=0.05, max_value=0.40, value=0.20, step=0.05)
    analog_k = st.slider("Number of analogs", min_value=10, max_value=60, value=20, step=5)
    dist_horizon = st.selectbox("Analog distribution horizon", [5, 10, 20], index=1)
    show_raw = st.checkbox("Show raw merged data", value=False)

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
    merged = append_snapshot(merged, load_snapshot(daily_file, snapshot_date))

merged = maybe_cumulate(merged, mode_map)
merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

include_optional = []
if include_bpspx and "BPSPX" in merged.columns:
    include_optional.append("BPSPX")
if include_spxa50r and "SPXA50R" in merged.columns:
    include_optional.append("SPXA50R")

model = build_model_frame(merged, include_optional, include_rsp, rsp_weight)
active_inputs = [s for s in CORE_CUMULATIVE_SERIES + include_optional if f"{s}_master_score" in model.columns]
current = model.iloc[-1]

left, right = st.columns([1.15, 0.85])
with left:
    st.plotly_chart(plot_main(model, merged, years_master, include_rsp), width="stretch")
with right:
    st.subheader("Current score")
    breadth_delta = model["Breadth_Ultimate"].diff().iloc[-1] if len(model) > 1 else np.nan
    ult_delta = model["Ultimate_With_RSP"].diff().iloc[-1] if len(model) > 1 else np.nan
    st.metric("Breadth ultimate", format_num(current["Breadth_Ultimate"], 1), None if pd.isna(breadth_delta) else format_num(breadth_delta, 1))
    st.metric("Master oscillator", format_num(current["Ultimate_With_RSP"], 1), None if pd.isna(ult_delta) else format_num(ult_delta, 1))
    st.metric("Master state", current["Ultimate_State"])
    st.metric("Regime cluster", current["RegimeCluster"])
    st.metric("Bull/Bear cross", "Bull" if current["Ultimate_Cross"] > 0 else "Bear")
    if "RSP" in merged.columns and merged["RSP"].notna().any():
        st.metric("RSP price", format_num(merged["RSP"].dropna().iloc[-1], 2))
    for s in active_inputs:
        if f"{s}_master_score" in current.index:
            label = f"{s} {'Bull' if current.get(f'{s}_cross', -1) > 0 else 'Bear'}"
            st.metric(label, format_num(current[f"{s}_master_score"], 1))
    st.plotly_chart(plot_components(model, active_inputs, years_components), width="stretch")

with st.expander("Data coverage"):
    st.dataframe(describe_dataset(merged), width="stretch")

st.subheader("Historical calendar lookup")
if not model.empty:
    min_date = model["Date"].min().date()
    max_date = model["Date"].max().date()
    lookup_date = st.date_input("Lookup date", value=max_date, min_value=min_date, max_value=max_date, key="lookup_date")
    lookup_idx = get_lookup_idx(model, pd.Timestamp(lookup_date))
    if lookup_idx is not None:
        row = model.iloc[lookup_idx]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Lookup master", format_num(row["Ultimate_With_RSP"], 1))
        r2.metric("Lookup state", row["Ultimate_State"])
        r3.metric("Lookup cross", "Bull" if row["Ultimate_Cross"] > 0 else "Bear")
        rsp_lookup = safe_price_at(merged["RSP"], lookup_idx, 0) if "RSP" in merged.columns else None
        r4.metric("Lookup RSP", format_num(rsp_lookup, 2))
        if "RSP" in merged.columns:
            p1, p2, p3 = st.columns(3)
            for col_obj, h in zip([p1, p2, p3], [5, 10, 20]):
                future_price = safe_price_at(merged["RSP"], lookup_idx, h)
                future_ret = None
                if rsp_lookup is not None and future_price is not None and rsp_lookup != 0:
                    future_ret = future_price / rsp_lookup - 1
                col_obj.metric(f"RSP +{h}d", format_num(future_price, 2), format_pct(future_ret))

        comp_rows = []
        for s in active_inputs:
            comp_rows.append({
                "Input": s,
                "Score": round(float(row.get(f"{s}_master_score", np.nan)), 2) if pd.notna(row.get(f"{s}_master_score", np.nan)) else np.nan,
                "Signal": round(float(row.get(f"{s}_signal", np.nan)), 2) if pd.notna(row.get(f"{s}_signal", np.nan)) else np.nan,
                "Cross": "Bull" if row.get(f"{s}_cross", -1) > 0 else "Bear",
                "Raw": row.get(f"{s}_raw", np.nan),
            })
        st.dataframe(pd.DataFrame(comp_rows), width="stretch")

st.subheader("Analog prediction score")
if "RSP" not in merged.columns or merged["RSP"].dropna().empty:
    st.info("Analog forward returns require RSP in the uploaded data.")
else:
    analog_inputs = CORE_CUMULATIVE_SERIES.copy()
    if use_bpspx_in_analog and "BPSPX" in include_optional:
        analog_inputs.append("BPSPX")
    if use_spxa50r_in_analog and "SPXA50R" in include_optional:
        analog_inputs.append("SPXA50R")
    
    analogs, hist = similarity_backtest(model, merged["RSP"], target_col="Ultimate_With_RSP", analog_inputs=analog_inputs, k=analog_k)
    stats = weighted_forward_stats(analogs)

    c1, c2, c3 = st.columns(3)
    current_rsp = merged["RSP"].dropna().iloc[-1]
    if stats:
        for col_obj, h in zip([c1, c2, c3], [5, 10, 20]):
            mean_ret = stats.get(f"mean_{h}d")
            median_ret = stats.get(f"median_{h}d")
            price_target = current_rsp * (1 + mean_ret) if mean_ret is not None and not pd.isna(mean_ret) else np.nan
            col_obj.metric(f"{h}d expected", format_pct(mean_ret), f"Median {format_pct(median_ret)}")
            col_obj.metric(f"{h}d implied RSP", format_num(price_target, 2), f"Win {stats.get(f'winrate_{h}d', np.nan) * 100:.0f}%")

        summary = {
            "current_state": str(current["Ultimate_State"]),
            "current_score": None if pd.isna(current["Ultimate_With_RSP"]) else float(current["Ultimate_With_RSP"]),
            "current_rsp": float(current_rsp),
            "regime_cluster": str(current["RegimeCluster"]),
            "expected_5d": stats.get("mean_5d"),
            "expected_10d": stats.get("mean_10d"),
            "expected_20d": stats.get("mean_20d"),
            "analogs_used": int(len(analogs)),
        }
        st.code(json.dumps(summary, indent=2, default=str), language="json")
        st.plotly_chart(analog_distribution_chart(analogs, dist_horizon), width="stretch")
    else:
        st.warning("Not enough overlapping history to build analogs yet.")

if not analogs.empty:
    display = analogs[["Date", "target_score", "relrange", "cross", "regime_code", "distance", "similarity", "fwd_5d", "fwd_10d", "fwd_20d", "fwd_price_5d", "fwd_price_10d", "fwd_price_20d"]].copy()
    display["cross"] = display["cross"].map({1: "Bull", -1: "Bear"})
    display["regime_code"] = display["regime_code"].map({0: "washout", 1: "repair", 2: "range", 3: "constructive", 4: "exhaustion"})
    display = display.rename(columns={"target_score": "score", "relrange": "relative_range"})
    for col in ["fwd_5d", "fwd_10d", "fwd_20d"]:
        display[col] = display[col].map(format_pct)
    for col in ["score", "relative_range", "distance", "similarity", "fwd_price_5d", "fwd_price_10d", "fwd_price_20d"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="ignore")
    st.dataframe(display, width="stretch")

if show_raw:
    st.subheader("Merged source data")
    st.dataframe(merged.tail(300), width="stretch")

with st.expander("How the state logic works"):
    st.markdown(
        """
Expansion: high score and still improving.
Exhaustion Risk: high score but rolling over.
Constructive: above neutral and improving.
Constructive / Fading: still positive but losing thrust.
Neutral / Improving vs Neutral / Regressing: same zone, different direction.
Repair / Progressing vs Repair / Stalling: early recovery vs bounce losing force.
Washout / Reversal Watch: deeply weak but turning up.
Stress / Breakdown: weak and still deteriorating.
The master oscillator is shown as a TSI-style score with a signal line, so you can read bull/bear crosses more cleanly than with a noisy raw score line.
"""
    )

with st.expander("Expected file formats"):
    st.markdown(
        """
Historical ZIP
Supports StockCharts historical exports.
Also supports normal CSV/XLSX/TXT files with `Date` plus one series column.
Daily snapshot
Supports StockCharts symbol-table snapshot files with `Symbol` and `Close`.
Or files with explicit columns like `Date, NYAD, NYSI, NYHL, BPSPX, SPXA50R, RSP`.
Cumulative behavior
NYAD / NYSI / NYHL are usually cumulative in your use case.
BPSPX / SPXA50R should stay raw / non-cumulative.
RSP should stay raw price.
"""
    )
