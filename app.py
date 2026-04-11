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
STATE_ORDER = [
    "Stress / Breakdown",
    "Washout / Reversal Watch",
    "Repair / Stalling",
    "Repair / Progressing",
    "Neutral / Regressing",
    "Neutral / Improving",
    "Constructive / Fading",
    "Constructive",
    "Expansion",
    "Exhaustion Risk",
]


# -----------------------------
# Helpers
# -----------------------------

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_col_name(col: str) -> str:
    c = str(col).strip().upper().replace("$", "")
    c = c.replace(" ", "_").replace("-", "_")
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
        "SYMBOL": "Symbol",
    }
    return aliases.get(c, c)


def infer_series_name_from_text(text: str) -> Optional[str]:
    upper = text.upper().replace("$", "")
    for s in ALL_SERIES + ["NYMO"]:
        if s in upper:
            return s
    return None


def format_num(x: Optional[float], decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.{decimals}f}"


def format_pct(x: Optional[float], decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{100 * float(x):.{decimals}f}%"


# -----------------------------
# File loaders
# -----------------------------

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

    try:
        raw = pd.read_csv(io.BytesIO(blob))
        raw = raw.rename(columns={c: normalize_col_name(c) for c in raw.columns}).copy()
        if {"Symbol", "Value"}.issubset(raw.columns):
            raw["Symbol"] = raw["Symbol"].astype(str).str.upper().str.replace("$", "", regex=False)
            found = {}
            for s in ALL_SERIES:
                vals = raw.loc[raw["Symbol"] == s, "Value"]
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


# -----------------------------
# Data prep
# -----------------------------

def merge_series(series_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for sdf in series_dict.values():
        merged = sdf.copy() if merged is None else merged.merge(sdf, on="Date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["Date"] + ALL_SERIES)
    merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return merged


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
        if "daily delta" in mode.lower():
            out[s] = out[s].fillna(0).cumsum()
    return out


def fill_for_indicators(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = safe_numeric(out[c]).ffill()
    return out


# -----------------------------
# Indicators
# -----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(10, window // 2)).mean()
    std = series.rolling(window, min_periods=max(10, window // 2)).std(ddof=0).replace(0, np.nan)
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
    ma = series.rolling(window, min_periods=max(10, window // 2)).mean()
    std = series.rolling(window, min_periods=max(10, window // 2)).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)


def roc(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change(window)


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

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return adx


def relative_range(series: pd.Series, window: int = 252) -> pd.Series:
    lo = series.rolling(window, min_periods=max(30, window // 4)).min()
    hi = series.rolling(window, min_periods=max(30, window // 4)).max()
    return (series - lo) / (hi - lo).replace(0, np.nan)


def clip_scale(series: pd.Series, lo: float, hi: float) -> pd.Series:
    x = series.clip(lower=lo, upper=hi)
    return (x - lo) / (hi - lo)


def build_input_features(series: pd.Series) -> pd.DataFrame:
    series = safe_numeric(series).ffill()
    z_lvl = rolling_zscore(series, 63)
    tsi, tsi_signal = true_strength_index(series, 25, 13, 7)
    roc_z = rolling_zscore(roc(series, 20), 63)
    pctb = bollinger_percent_b(series, 20, 2)
    adx = adx_from_close(series, 14)
    rr = relative_range(series, 252)
    slope = ema(series.diff(5), 5)
    slope_z = rolling_zscore(slope, 63)

    score = (
        0.24 * clip_scale(z_lvl, -2.5, 2.5)
        + 0.24 * clip_scale(tsi, -40, 40)
        + 0.18 * clip_scale(roc_z, -2.5, 2.5)
        + 0.16 * pctb.clip(0, 1)
        + 0.10 * clip_scale(adx, 10, 40)
        + 0.08 * rr.clip(0, 1)
    ) * 100.0
    score_signal = ema(score, 7)

    return pd.DataFrame(
        {
            "raw": series,
            "z_lvl": z_lvl,
            "tsi": tsi,
            "tsi_signal": tsi_signal,
            "roc_z": roc_z,
            "pctb": pctb,
            "adx": adx,
            "relrange": rr,
            "slope_z": slope_z,
            "master_score": score,
            "signal": score_signal,
            "cross": np.where(score >= score_signal, 1, -1),
            "score_delta_5": score.diff(5),
            "score_delta_20": score.diff(20),
        }
    )


# -----------------------------
# State logic
# -----------------------------

def classify_regime(score: float, d5: float, d20: float, cross: int) -> str:
    if pd.isna(score):
        return "Unknown"

    improving = (not pd.isna(d5) and d5 > 0) or (not pd.isna(d20) and d20 > 0)
    fading = (not pd.isna(d5) and d5 < 0) or (not pd.isna(d20) and d20 < 0)

    if score >= 82:
        return "Expansion" if improving and cross > 0 else "Exhaustion Risk"
    if score >= 65:
        return "Constructive" if improving and cross > 0 else "Constructive / Fading"
    if score >= 45:
        return "Neutral / Improving" if improving and cross > 0 else "Neutral / Regressing"
    if score >= 28:
        return "Repair / Progressing" if improving or cross > 0 else "Repair / Stalling"
    return "Washout / Reversal Watch" if improving or cross > 0 else "Stress / Breakdown"


def regime_code_from_state(state: str) -> int:
    mapping = {
        "Stress / Breakdown": 0,
        "Washout / Reversal Watch": 0,
        "Repair / Stalling": 1,
        "Repair / Progressing": 1,
        "Neutral / Regressing": 2,
        "Neutral / Improving": 2,
        "Constructive / Fading": 3,
        "Constructive": 3,
        "Expansion": 4,
        "Exhaustion Risk": 4,
    }
    return mapping.get(state, 2)


# -----------------------------
# Model frame
# -----------------------------

def build_model_frame(merged: pd.DataFrame, include_optional: List[str], include_rsp: bool, rsp_weight: float) -> pd.DataFrame:
    merged = merged.sort_values("Date").reset_index(drop=True).copy()
    use_cols = [s for s in CORE_CUMULATIVE_SERIES + include_optional + (["RSP"] if include_rsp else []) if s in merged.columns]
    merged = fill_for_indicators(merged, use_cols)

    model = pd.DataFrame({"Date": merged["Date"]})
    score_cols = []

    for s in CORE_CUMULATIVE_SERIES + include_optional:
        if s not in merged.columns:
            continue
        feats = build_input_features(merged[s])
        for c in feats.columns:
            model[f"{s}_{c}"] = feats[c]
        score_cols.append(f"{s}_master_score")

    breadth_numer = None
    breadth_denom = None
    for s in CORE_CUMULATIVE_SERIES + include_optional:
        sc = f"{s}_master_score"
        if sc not in model.columns:
            continue
        w = DEFAULT_WEIGHTS.get(s, 0.0)
        contrib = model[sc] * w
        breadth_numer = contrib if breadth_numer is None else breadth_numer.add(contrib, fill_value=np.nan)
        valid = model[sc].notna().astype(float) * w
        breadth_denom = valid if breadth_denom is None else breadth_denom.add(valid, fill_value=0.0)

    model["Breadth_Ultimate"] = breadth_numer / breadth_denom.replace(0, np.nan)
    model["Breadth_Signal"] = ema(model["Breadth_Ultimate"], 7)
    model["Breadth_Cross"] = np.where(model["Breadth_Ultimate"] >= model["Breadth_Signal"], 1, -1)
    model["Breadth_Delta_5"] = model["Breadth_Ultimate"].diff(5)
    model["Breadth_Delta_20"] = model["Breadth_Ultimate"].diff(20)
    model["Breadth_State"] = [
        classify_regime(s, d5, d20, c)
        for s, d5, d20, c in zip(model["Breadth_Ultimate"], model["Breadth_Delta_5"], model["Breadth_Delta_20"], model["Breadth_Cross"])
    ]

    if include_rsp and "RSP" in merged.columns:
        rsp_feats = build_input_features(merged["RSP"])
        for c in rsp_feats.columns:
            model[f"RSP_{c}"] = rsp_feats[c]
        model["Ultimate_With_RSP"] = (1 - rsp_weight) * model["Breadth_Ultimate"] + rsp_weight * model["RSP_master_score"]
    else:
        model["Ultimate_With_RSP"] = model["Breadth_Ultimate"]

    model["Ultimate_Signal"] = ema(model["Ultimate_With_RSP"], 7)
    model["Ultimate_Cross"] = np.where(model["Ultimate_With_RSP"] >= model["Ultimate_Signal"], 1, -1)
    model["Ultimate_Delta_5"] = model["Ultimate_With_RSP"].diff(5)
    model["Ultimate_Delta_20"] = model["Ultimate_With_RSP"].diff(20)
    model["Ultimate_State"] = [
        classify_regime(s, d5, d20, c)
        for s, d5, d20, c in zip(model["Ultimate_With_RSP"], model["Ultimate_Delta_5"], model["Ultimate_Delta_20"], model["Ultimate_Cross"])
    ]
    model["RegimeCluster"] = model["Ultimate_State"]
    model["RegimeCode"] = model["Ultimate_State"].map(regime_code_from_state)
    model["Ultimate_RelRange"] = relative_range(model["Ultimate_With_RSP"], 252)

    return model


# -----------------------------
# Analog engine
# -----------------------------

def safe_forward_return(price: pd.Series, horizon: int) -> pd.Series:
    return price.shift(-horizon) / price - 1.0


def safe_forward_price(price: pd.Series, horizon: int) -> pd.Series:
    return price.shift(-horizon)


def similarity_backtest(
    model: pd.DataFrame,
    rsp_series: pd.Series,
    target_col: str,
    analog_inputs: List[str],
    k: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if target_col not in model.columns:
        return pd.DataFrame(), {}

    work = model.copy()
    work["RSP"] = safe_numeric(rsp_series).ffill().to_numpy()
    work["fwd_5d"] = safe_forward_return(work["RSP"], 5)
    work["fwd_10d"] = safe_forward_return(work["RSP"], 10)
    work["fwd_20d"] = safe_forward_return(work["RSP"], 20)
    work["fwd_price_5d"] = safe_forward_price(work["RSP"], 5)
    work["fwd_price_10d"] = safe_forward_price(work["RSP"], 10)
    work["fwd_price_20d"] = safe_forward_price(work["RSP"], 20)

    feature_cols = [
        target_col,
        "Ultimate_Signal",
        "Ultimate_Delta_5",
        "Ultimate_Delta_20",
        "Ultimate_RelRange",
        "Ultimate_Cross",
        "RegimeCode",
    ]
    for s in analog_inputs:
        feature_cols.extend([
            f"{s}_master_score",
            f"{s}_signal",
            f"{s}_score_delta_5",
            f"{s}_score_delta_20",
            f"{s}_relrange",
            f"{s}_cross",
        ])

    feature_cols = [c for c in feature_cols if c in work.columns]
    if len(feature_cols) < 4:
        return pd.DataFrame(), {}

    work = work.apply(pd.to_numeric, errors="ignore")
    numeric_features = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    horizon_cols = ["fwd_5d", "fwd_10d", "fwd_20d", "fwd_price_5d", "fwd_price_10d", "fwd_price_20d"]
    valid = pd.concat([work[["Date"]], numeric_features, work[horizon_cols]], axis=1)
    valid = valid.dropna(subset=[target_col])

    if len(valid) < max(80, k + 20):
        return pd.DataFrame(), {}

    current = valid.iloc[-1]
    pool = valid.iloc[:-21].copy() if len(valid) > 21 else valid.iloc[:-1].copy()
    pool = pool.dropna(subset=["fwd_5d", "fwd_10d", "fwd_20d"])
    if pool.empty:
        return pd.DataFrame(), {}

    X = pool[feature_cols].apply(pd.to_numeric, errors="coerce")
    cur = pd.to_numeric(current[feature_cols], errors="coerce")

    keep_cols = [c for c in feature_cols if pd.notna(cur[c]) and X[c].notna().sum() >= max(30, int(0.5 * len(X)))]
    if len(keep_cols) < 4:
        return pd.DataFrame(), {}

    X = X[keep_cols].copy()
    cur = cur[keep_cols].copy()

    X = X.fillna(X.median())
    cur = cur.fillna(X.median())

    mu = X.mean()
    sigma = X.std(ddof=0).replace(0, np.nan)
    Xs = ((X - mu) / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cs = ((cur - mu) / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xs_np = np.asarray(Xs.to_numpy(dtype=np.float64), dtype=np.float64)
    cs_np = np.asarray(cs.to_numpy(dtype=np.float64), dtype=np.float64)
    dist = np.linalg.norm(Xs_np - cs_np, axis=1)

    pool = pool.loc[X.index].copy()
    pool["distance"] = dist
    pool["similarity"] = 1.0 / (1.0 + pool["distance"])
    if "Ultimate_Cross" in keep_cols:
        pool.loc[pool["Ultimate_Cross"] == current.get("Ultimate_Cross"), "similarity"] *= 1.15
    if "RegimeCode" in keep_cols:
        pool.loc[pool["RegimeCode"] == current.get("RegimeCode"), "similarity"] *= 1.20

    pool = pool.sort_values(["distance", "Date"]).head(k).copy()
    if pool.empty:
        return pd.DataFrame(), {}

    hist = {
        "current_score": float(current[target_col]) if pd.notna(current[target_col]) else np.nan,
        "current_relrange": float(current.get("Ultimate_RelRange", np.nan)),
        "current_cross": int(current.get("Ultimate_Cross", 0)) if pd.notna(current.get("Ultimate_Cross", np.nan)) else 0,
        "current_regime_code": int(current.get("RegimeCode", 2)) if pd.notna(current.get("RegimeCode", np.nan)) else 2,
    }
    return pool, hist


def weighted_forward_stats(analogs: pd.DataFrame) -> Dict[str, float]:
    if analogs.empty:
        return {}
    w = analogs["similarity"].clip(lower=1e-9)
    out: Dict[str, float] = {}
    for h in [5, 10, 20]:
        vals = pd.to_numeric(analogs[f"fwd_{h}d"], errors="coerce")
        mask = vals.notna() & w.notna()
        if not mask.any():
            continue
        vals_np = vals[mask].to_numpy(dtype=float)
        w_np = w[mask].to_numpy(dtype=float)
        out[f"mean_{h}d"] = float(np.average(vals_np, weights=w_np))
        out[f"median_{h}d"] = float(np.median(vals_np))
        out[f"winrate_{h}d"] = float((vals_np > 0).mean())
    return out


# -----------------------------
# Charts / display helpers
# -----------------------------

def filter_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df["Date"].max()
    start = end - pd.DateOffset(years=years)
    return df[df["Date"] >= start].copy()


def plot_main(model: pd.DataFrame, merged: pd.DataFrame, years: int, include_rsp: bool) -> go.Figure:
    plot_df = filter_years(model, years)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Breadth_Ultimate"], name="Breadth score", mode="lines"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Breadth_Signal"], name="Breadth signal", mode="lines"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Ultimate_With_RSP"], name="Master score", mode="lines"))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Ultimate_Signal"], name="Master signal", mode="lines"))
    if include_rsp and "RSP" in merged.columns:
        rsp_df = merged[["Date", "RSP"]].copy()
        rsp_df = filter_years(rsp_df, years)
        fig.add_trace(go.Scatter(x=rsp_df["Date"], y=rsp_df["RSP"], name="RSP", mode="lines", yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="RSP", overlaying="y", side="right", showgrid=False))
    fig.update_layout(height=460, title=f"Master oscillator ({years}Y)", legend=dict(orientation="h"))
    return fig


def plot_components(model: pd.DataFrame, active_inputs: List[str], years: int) -> go.Figure:
    plot_df = filter_years(model, years)
    fig = go.Figure()
    for s in active_inputs:
        sc = f"{s}_master_score"
        sig = f"{s}_signal"
        if sc in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df[sc], name=f"{s} score", mode="lines"))
        if sig in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df[sig], name=f"{s} signal", mode="lines"))
    fig.update_layout(height=480, title=f"Component oscillators ({years}Y)", legend=dict(orientation="h"))
    return fig


def analog_distribution_chart(analogs: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    if analogs.empty:
        fig.update_layout(title="No analogs available")
        return fig
    vals = pd.to_numeric(analogs[f"fwd_{horizon}d"], errors="coerce").dropna() * 100.0
    fig.add_trace(go.Histogram(x=vals, nbinsx=20, name=f"{horizon}d returns"))
    fig.update_layout(height=340, title=f"Analog forward {horizon}d RSP return distribution")
    return fig


def describe_dataset(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in [s for s in ALL_SERIES if s in merged.columns]:
        ser = safe_numeric(merged[c])
        rows.append({
            "Series": c,
            "Rows": int(ser.notna().sum()),
            "First Date": merged.loc[ser.notna(), "Date"].min(),
            "Last Date": merged.loc[ser.notna(), "Date"].max(),
            "Latest": ser.dropna().iloc[-1] if ser.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def get_lookup_idx(df: pd.DataFrame, lookup_date: pd.Timestamp) -> Optional[int]:
    filt = df[df["Date"] <= lookup_date]
    if filt.empty:
        return None
    return int(filt.index[-1])


def safe_price_at(price_series: pd.Series, idx: int, forward_days: int = 0) -> Optional[float]:
    target_idx = idx + forward_days
    if target_idx < 0 or target_idx >= len(price_series):
        return None
    val = pd.to_numeric(price_series.iloc[target_idx], errors="coerce")
    return None if pd.isna(val) else float(val)


# -----------------------------
# UI
# -----------------------------

st.title("Breadth Regime Dashboard")
st.caption("Upload a historical ZIP plus an optional daily snapshot. The app supports cumulative breadth inputs, optional raw BPSPX/SPXA50R overlays, RSP blending, calendar lookup, and analog forecasting.")

with st.sidebar:
    st.header("Inputs")
    hist_zip = st.file_uploader("Historical ZIP", type=["zip"])
    daily_file = st.file_uploader("Daily snapshot", type=["csv", "txt", "xlsx", "xls"])
    snapshot_date = pd.Timestamp(st.date_input("Snapshot date", value=pd.Timestamp.today().date()))

    st.header("Series modes")
    mode_map: Dict[str, str] = {}
    for s in CORE_CUMULATIVE_SERIES:
        opts = ["Already cumulative", "Daily delta → cumulative"]
        mode_map[s] = st.selectbox(f"{s} mode", opts, index=0, key=f"mode_{s}")
    for s in OPTIONAL_RAW_SERIES + PRICE_SERIES:
        mode_map[s] = "Raw / non-cumulative"

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
    analog_k = st.slider("Number of analogs", min_value=10, max_value=80, value=25, step=5)
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
if model.empty:
    st.error("The model frame is empty after processing. Check that your uploaded files contain usable numeric history.")
    st.stop()

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
    st.metric("Master state", str(current["Ultimate_State"]))
    st.metric("Regime cluster", str(current["RegimeCluster"]))
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
min_date = model["Date"].min().date()
max_date = model["Date"].max().date()
lookup_date = st.date_input("Lookup date", value=max_date, min_value=min_date, max_value=max_date, key="lookup_date")
lookup_idx = get_lookup_idx(model, pd.Timestamp(lookup_date))
if lookup_idx is not None:
    row = model.iloc[lookup_idx]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Lookup master", format_num(row["Ultimate_With_RSP"], 1))
    r2.metric("Lookup state", str(row["Ultimate_State"]))
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
    current_rsp = float(merged["RSP"].dropna().iloc[-1])
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
        display_cols = [
            "Date", "distance", "similarity", "fwd_5d", "fwd_10d", "fwd_20d",
            "fwd_price_5d", "fwd_price_10d", "fwd_price_20d"
        ]
        keep = [c for c in display_cols if c in analogs.columns]
        display = analogs[keep].copy()
        for col in ["fwd_5d", "fwd_10d", "fwd_20d"]:
            if col in display.columns:
                display[col] = display[col].map(format_pct)
        st.dataframe(display, width="stretch")

if show_raw:
    st.subheader("Merged source data")
    st.dataframe(merged.tail(300), width="stretch")

with st.expander("How the state logic works"):
    st.markdown(
        """
- **Expansion**: high score and still improving.
- **Exhaustion Risk**: high score but rolling over.
- **Constructive**: above neutral and improving.
- **Constructive / Fading**: still positive but losing thrust.
- **Neutral / Improving** vs **Neutral / Regressing**: same zone, different direction.
- **Repair / Progressing** vs **Repair / Stalling**: early recovery vs bounce losing force.
- **Washout / Reversal Watch**: deeply weak but turning up.
- **Stress / Breakdown**: weak and still deteriorating.

The master oscillator is shown as a **TSI-style score with a signal line**, so you can read bull/bear crosses more cleanly than with a noisy raw score line.
        """
    )

with st.expander("Expected file formats"):
    st.markdown(
        """
**Historical ZIP**
- Supports StockCharts historical exports.
- Also supports normal CSV/XLSX/TXT files with `Date` plus one series column.

**Daily snapshot**
- Supports StockCharts symbol-table snapshot files with `Symbol` and `Close`.
- Or files with explicit columns like `Date, NYAD, NYSI, NYHL, BPSPX, SPXA50R, RSP`.

**Cumulative behavior**
- NYAD / NYSI / NYHL are usually cumulative in your use case.
- BPSPX / SPXA50R should stay raw / non-cumulative.
- RSP should stay raw price.
        """
    )
