
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pytz

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store_hybrid"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NY_TZ = "America/New_York"

st.set_page_config(page_title="Stable Market Engine Hybrid", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Stable Market Engine Hybrid")
st.caption("Stable dashboard shell + TSI-first real/proxy intraday logic. Alpaca-only feed.")


def to_ny_naive_index(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    else:
        idx = pd.DatetimeIndex(idx)
    return idx


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out.columns = [str(c).title() for c in out.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out[keep].copy()
    out.index = to_ny_naive_index(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(2, window // 3)).mean()


def slope(series: pd.Series, bars: int = 3) -> pd.Series:
    return series - series.shift(bars)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window, min_periods=max(5, window // 2)).mean()
    md = (tp - ma).abs().rolling(window, min_periods=max(5, window // 2)).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13, signal_period: int = 7) -> Tuple[pd.Series, pd.Series]:
    delta = series.diff()
    abs_delta = delta.abs()
    double_smoothed = ema(ema(delta, long_period), short_period)
    double_abs = ema(ema(abs_delta, long_period), short_period)
    tsi_line = 100 * double_smoothed / double_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal_period)
    return tsi_line, signal_line


def bollinger_pct_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = series.rolling(window, min_periods=max(5, window // 2)).mean()
    std = series.rolling(window, min_periods=max(5, window // 2)).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=max(5, window // 2)).mean()


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0.0)
    pv = typical * vol
    pv_sum = pv.rolling(window, min_periods=max(5, window // 3)).sum()
    v_sum = vol.rolling(window, min_periods=max(5, window // 3)).sum()
    return pv_sum / v_sum.replace(0, np.nan)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0.0)
    groups = pd.DatetimeIndex(df.index).date
    cum_pv = (typical * vol).groupby(groups).cumsum()
    cum_v = vol.groupby(groups).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def rolling_minmax_pct(series: pd.Series, window: int = 252) -> pd.Series:
    lo = series.rolling(window, min_periods=max(20, window // 5)).min()
    hi = series.rolling(window, min_periods=max(20, window // 5)).max()
    return ((series - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)


def true_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    min_valid = max(30, window // 5)
    for i, v in enumerate(values):
        if i + 1 < min_valid or not np.isfinite(v):
            continue
        w = values[max(0, i - window + 1): i + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_valid:
            continue
        out[i] = float((w <= v).mean())
    return pd.Series(out, index=series.index)


def hybrid_norm(series: pd.Series, window: int) -> pd.Series:
    pct = true_percentile(series, window)
    mm = rolling_minmax_pct(series, window)
    return (0.65 * pct + 0.35 * mm).clip(0, 1)


def parse_symbol_csv(uploaded) -> List[str]:
    if uploaded is None:
        return []
    raw = uploaded.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return []
    cols = {str(c).strip().lower(): c for c in df.columns}
    sym_col = cols.get("symbol") or cols.get("ticker") or next(iter(df.columns), None)
    if sym_col is None:
        return []
    return [str(x).strip().upper() for x in df[sym_col].dropna().tolist() if str(x).strip()]


def clean_symbols(text: str, uploaded_symbols: List[str]) -> List[str]:
    symbols: List[str] = []
    if text.strip():
        symbols.extend([s.strip().upper() for s in text.replace("\n", ",").split(",") if s.strip()])
    symbols.extend(uploaded_symbols)
    out = []
    for s in symbols:
        if s and s not in out:
            out.append(s)
    return out


def cache_path(symbol: str, kind: str) -> Path:
    safe = "".join(c for c in symbol if c.isalnum() or c in "._-")
    return CACHE_DIR / f"{safe}_{kind}.parquet"


def is_fresh(path: Path, max_hours: int = 18) -> bool:
    if not path.exists():
        return False
    age = pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
    return age < pd.Timedelta(hours=max_hours)


def clear_symbol_cache(symbols: List[str]) -> None:
    for s in symbols:
        for kind in ["daily", "1hour", "2hour"]:
            p = cache_path(s, kind)
            if p.exists():
                p.unlink(missing_ok=True)


def alpaca_client(key: str, secret: str) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(key, secret)


def _bars_df_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        if "symbol" in df.index.names:
            try:
                sub = df.xs(symbol, level="symbol").copy()
            except Exception:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    else:
        sub = df.copy()
    if "timestamp" in sub.columns:
        sub = sub.set_index("timestamp")
    return normalize_ohlcv(sub)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpaca_daily_batch(symbols: List[str], years: int, key: str, secret: str, feed: str) -> Dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    data_map: Dict[str, pd.DataFrame] = {}
    missing = []
    for s in symbols:
        p = cache_path(s, "daily")
        if is_fresh(p, max_hours=24):
            try:
                data_map[s] = pd.read_parquet(p)
                continue
            except Exception:
                pass
        missing.append(s)
    if not missing:
        return data_map
    client = alpaca_client(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(years=max(years, 5))).tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=missing, timeframe=TimeFrame.Day, start=start, end=end, adjustment="all", feed=feed)
    raw = None
    for attempt in range(3):
        try:
            raw = client.get_stock_bars(req).df
            break
        except Exception:
            if attempt == 2:
                return data_map
            time.sleep(1.5 * (attempt + 1))
    for s in missing:
        sub = _bars_df_for_symbol(raw, s)
        if not sub.empty:
            sub.to_parquet(cache_path(s, "daily"))
            data_map[s] = sub
    return data_map


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_alpaca_1hour(symbol: str, months: int, key: str, secret: str, feed: str) -> pd.DataFrame:
    p = cache_path(symbol, "1hour")
    if is_fresh(p, max_hours=6):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    client = alpaca_client(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(months=max(months, 3))).tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=start, end=end, adjustment="all", feed=feed)
    raw = None
    for attempt in range(3):
        try:
            raw = client.get_stock_bars(req).df
            break
        except Exception:
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(1.5 * (attempt + 1))
    sub = _bars_df_for_symbol(raw, symbol)
    if sub.empty:
        return pd.DataFrame()
    sub = sub.between_time("09:30", "16:00")
    if not sub.empty:
        sub.to_parquet(p)
    return sub


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_alpaca_2hour(symbol: str, months: int, key: str, secret: str, feed: str) -> pd.DataFrame:
    p = cache_path(symbol, "2hour")
    if is_fresh(p, max_hours=6):
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    client = alpaca_client(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(months=max(months, 3))).tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(30, "Min"), start=start, end=end, adjustment="all", feed=feed)
    raw = None
    for attempt in range(3):
        try:
            raw = client.get_stock_bars(req).df
            break
        except Exception:
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(1.5 * (attempt + 1))
    sub = _bars_df_for_symbol(raw, symbol)
    if sub.empty:
        return pd.DataFrame()
    sub = sub.between_time("09:30", "16:00")
    pieces = []
    for _, day_df in sub.groupby(sub.index.date):
        if day_df.empty:
            continue
        anchor = pd.Timestamp(day_df.index[0].date()) + pd.Timedelta(hours=9, minutes=30)
        ranges = [
            (anchor, anchor + pd.Timedelta(hours=2)),
            (anchor + pd.Timedelta(hours=2), anchor + pd.Timedelta(hours=4)),
            (anchor + pd.Timedelta(hours=4), anchor + pd.Timedelta(hours=6)),
            (anchor + pd.Timedelta(hours=6), anchor + pd.Timedelta(hours=6, minutes=30)),
        ]
        bins = []
        for start_ts, end_ts in ranges:
            chunk = day_df[(day_df.index >= start_ts) & (day_df.index < end_ts)]
            if chunk.empty:
                continue
            row = pd.DataFrame(
                {
                    "Open": [chunk["Open"].iloc[0]],
                    "High": [chunk["High"].max()],
                    "Low": [chunk["Low"].min()],
                    "Close": [chunk["Close"].iloc[-1]],
                    "Volume": [chunk["Volume"].sum()],
                },
                index=[start_ts],
            )
            bins.append(row)
        if bins:
            pieces.append(pd.concat(bins))
    out = normalize_ohlcv(pd.concat(pieces).sort_index()) if pieces else pd.DataFrame()
    if not out.empty:
        out.to_parquet(p)
    return out


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return normalize_ohlcv(df.resample("W-FRI").agg(agg).dropna(how="any"))


def build_proxy_intraday_from_daily(daily: pd.DataFrame, mode: str) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    anchors = ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"] if mode == "1H" else ["11:30", "13:30", "15:30"]
    reps = len(anchors)
    rows = []
    for ts, row in daily.iterrows():
        for a in anchors:
            hh, mm = map(int, a.split(":"))
            new_ts = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day, hour=hh, minute=mm)
            rows.append({
                "timestamp": new_ts,
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": row["Volume"] / reps if pd.notna(row["Volume"]) else np.nan,
            })
    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return normalize_ohlcv(out)


def trim_to_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty or not range_key or range_key == "MAX":
        return df.copy()
    end_ts = pd.Timestamp(df.index.max())
    mapping = {
        "2W": pd.DateOffset(weeks=2),
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "2Y": pd.DateOffset(years=2),
        "5Y": pd.DateOffset(years=5),
        "10Y": pd.DateOffset(years=10),
    }
    offset = mapping.get(range_key)
    if offset is None:
        return df.copy()
    start_ts = end_ts - offset
    return df.loc[df.index >= start_ts].copy()


RANGE_OPTIONS = {
    "1h": ["2W", "1M", "3M", "6M", "1Y"],
    "2h": ["1M", "3M", "6M", "1Y", "2Y"],
    "daily": ["3M", "6M", "1Y", "2Y", "5Y", "10Y"],
    "weekly": ["1Y", "2Y", "5Y", "10Y", "MAX"],
}


def classify_state(row: pd.Series, structural: bool = False) -> Tuple[str, str]:
    tsi_val = float(row.get("tsi", np.nan))
    tsi_sig = float(row.get("tsi_signal", np.nan))
    tsi_gap = float(row.get("tsi_gap", np.nan))
    tsi_slope = float(row.get("tsi_slope_3", 0.0))
    exhaust = float(row.get("exhaustion_score", 50.0))
    price_chg = float(row.get("price_slope_3", 0.0))
    if not np.isfinite(tsi_val) or not np.isfinite(tsi_sig):
        return "NO DATA", "No data"

    if tsi_val < tsi_sig:
        if exhaust >= 72:
            if abs(price_chg) < 0.004:
                return "PUT", "TSI below signal, exhaustion high, but price has not broken yet"
            return "PUT", "TSI below signal with elevated exhaustion"
        if structural and tsi_gap < 0 and tsi_slope < 0:
            return "PUT", "Confirmed structural bearish cross"
        if tsi_gap < 0:
            return "PUT", "TSI below signal"
        return "NEUTRAL", "Transition"

    if tsi_val > tsi_sig:
        if exhaust <= 28 and tsi_slope > 0:
            return "CALL", "Bull turn from washed-out condition"
        if structural and tsi_gap > 0 and tsi_slope > 0:
            return "CALL", "Confirmed structural bullish cross"
        if tsi_gap > 0:
            return "CALL", "TSI above signal"
        return "NEUTRAL", "Transition"

    return "NEUTRAL", "Mixed"


def enrich_price_features(df: pd.DataFrame, timeframe_name: str, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = normalize_ohlcv(df.copy())
    out["ema_10"] = ema(out["Close"], 10)
    out["ema_20"] = ema(out["Close"], 20)
    out["sma_50"] = sma(out["Close"], 50)
    out["atr_14"] = atr(out, 14)
    out["atr_stretch"] = (out["Close"] - out["ema_20"]) / out["atr_14"].replace(0, np.nan)

    out["rsi_14"] = rsi(out["Close"], 14)
    out["cci_20"] = cci(out, 20)
    tsi_raw, tsi_sig_raw = tsi(out["Close"], 25, 13, 7)
    out["tsi"] = tsi_raw
    out["tsi_signal"] = tsi_sig_raw
    out["tsi_gap"] = out["tsi"] - out["tsi_signal"]
    out["tsi_slope_3"] = slope(out["tsi"], 3)
    out["pct_b"] = bollinger_pct_b(out["Close"], 20, 2)
    out["price_slope_3"] = out["Close"].pct_change(3)
    out["dist_ema20_pct"] = (out["Close"] / out["ema_20"]) - 1

    if timeframe_name in {"1H", "2H", "1H_PROXY", "2H_PROXY"}:
        out["vwap"] = session_vwap(out)
        win = 140
    else:
        out["vwap"] = rolling_vwap(out, 20)
        win = 252
    out["dist_vwap_pct"] = (out["Close"] / out["vwap"]) - 1

    out["rsi_pct"] = hybrid_norm(out["rsi_14"], win)
    out["cci_pct"] = hybrid_norm(out["cci_20"], win)
    out["tsi_pct"] = hybrid_norm(out["tsi"], win)
    out["pct_b_pct"] = hybrid_norm(out["pct_b"], win)
    out["dist_vwap_pct_pct"] = hybrid_norm(out["dist_vwap_pct"], win)
    out["dist_ema20_pct_pct"] = hybrid_norm(out["dist_ema20_pct"], win)

    raw_exhaust = (
        0.30 * out["tsi_pct"].fillna(0.5)
        + 0.22 * out["cci_pct"].fillna(0.5)
        + 0.15 * out["rsi_pct"].fillna(0.5)
        + 0.18 * out["pct_b_pct"].fillna(0.5)
        + 0.15 * ((out["dist_vwap_pct_pct"].fillna(0.5) + out["dist_ema20_pct_pct"].fillna(0.5)) / 2.0)
    )
    out["exhaustion_score"] = ema(raw_exhaust * 100, 4)
    out["regime_score"] = (0.55 * out["tsi_pct"].fillna(0.5) + 0.45 * out["dist_ema20_pct_pct"].fillna(0.5)) * 100
    out["regime_bucket"] = pd.cut(
        out["regime_score"],
        bins=[-np.inf, 35, 50, 65, 80, np.inf],
        labels=["Weak", "Fading", "Neutral", "Strong", "Strong & Extended"],
    ).astype("object").fillna("Neutral")

    out["uo"] = out["tsi"]
    out["uo_signal"] = out["tsi_signal"]
    out["uo_decision"] = out["tsi"]
    out["uo_signal_decision"] = out["tsi_signal"]
    out["uo_gap"] = out["tsi_gap"]
    out["uo_slope_3"] = out["tsi_slope_3"]
    out["uo_pctile"] = out["tsi_pct"].fillna(0.5)
    out["ob_internal_score"] = (0.5 * out["rsi_pct"].fillna(0.5) + 0.5 * out["tsi_pct"].fillna(0.5)).clip(0, 1)
    out["ob_price_score"] = (0.5 * out["pct_b_pct"].fillna(0.5) + 0.5 * out["dist_ema20_pct_pct"].fillna(0.5)).clip(0, 1)
    out["os_internal_score"] = 1 - out["ob_internal_score"]
    out["os_price_score"] = 1 - out["ob_price_score"]
    return out


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [1, 2, 5]:
        out[f"fwd_ret_{n}"] = out["Close"].shift(-n) / out["Close"] - 1
    return out


def summarize_analogs(frame: pd.DataFrame, row: pd.Series, horizons: List[int], structural: bool = False) -> Dict[str, float]:
    if frame.empty or row is None or row.empty:
        return {}
    enriched = add_forward_returns(frame.copy())
    calls = [classify_state(r, structural)[0] for _, r in enriched.iterrows()]
    enriched["call"] = calls
    current_state = row.get("call", "NEUTRAL")
    current_regime = row.get("regime_bucket", "Neutral")
    pool = enriched.iloc[:-max(horizons)].copy()
    pool = pool[(pool["call"] == current_state) & (pool["regime_bucket"] == current_regime)]
    if pool.empty:
        return {}
    out = {"n": float(len(pool))}
    for n in horizons:
        c = f"fwd_ret_{n}"
        vals = pool[c].dropna()
        if len(vals):
            out[f"ret_{n}_median"] = float(vals.median())
            out[f"ret_{n}_p_up"] = float((vals > 0).mean())
            out[f"ret_{n}_p_down"] = float((vals < 0).mean())
    return out


def compute_severity(frame: pd.DataFrame, row: pd.Series) -> Dict[str, float]:
    pct = frame["uo_pctile"].dropna() if not frame.empty else pd.Series(dtype=float)
    def count_recent(cond: pd.Series) -> int:
        c = 0
        for ok in reversed(cond.fillna(False).astype(bool).tolist()):
            if ok:
                c += 1
            else:
                break
        return c
    bars_above_90 = count_recent(pct > 0.90)
    bars_below_10 = count_recent(pct < 0.10)
    return {
        "bars_above_90": bars_above_90,
        "bars_below_10": bars_below_10,
        "late_cycle_flag": int(bars_above_90 >= 4 and float(row.get("uo_slope_3", 0.0)) <= 0),
    }


def recommendation(hourly_call: str, tactical_call: str, daily_call: str, analogs: Dict[str, float]) -> str:
    if tactical_call == "PUT" and daily_call == "CALL":
        return "PUT setup forming"
    if tactical_call == "CALL" and daily_call == "CALL":
        return "CALL / trend aligned"
    if daily_call == "PUT":
        return "RISK OFF / daily weak"
    if analogs and analogs.get("ret_2_p_up", 0.5) > 0.62:
        return "CALL / analogs supportive"
    return "NEUTRAL / mixed"


def combine(hourly_call: str, tactical_call: str, daily_call: str, weekly_call: str) -> str:
    score = {"CALL": 1, "PUT": -1, "NEUTRAL": 0, "NO DATA": 0}
    net = 0.20 * score.get(hourly_call, 0) + 0.30 * score.get(tactical_call, 0) + 0.30 * score.get(daily_call, 0) + 0.20 * score.get(weekly_call, 0)
    if net >= 0.55:
        return "CALL"
    if net <= -0.55:
        return "PUT"
    if daily_call == "CALL" and weekly_call == "CALL":
        return "CALL ON PULLBACK"
    return "NEUTRAL"


def frame_warning_message(row: pd.Series, timeframe_label: str) -> str:
    if row is None or row.empty:
        return f"{timeframe_label}: no data."
    call = row.get("call", "NEUTRAL")
    ex = float(row.get("exhaustion_score", 50.0))
    if call == "PUT" and ex >= 72:
        return f"{timeframe_label}: bearish cross with elevated exhaustion."
    if call == "PUT":
        return f"{timeframe_label}: below signal and weakening."
    if call == "CALL" and ex <= 28:
        return f"{timeframe_label}: bull turn from washed-out condition."
    if call == "CALL":
        return f"{timeframe_label}: above signal and constructive."
    return f"{timeframe_label}: mixed state."


def distance_to_cross(row: pd.Series, frame: pd.DataFrame) -> Dict[str, float]:
    if row is None or row.empty or frame.empty:
        return {"gap": np.nan, "abs_gap": np.nan, "range_pct": np.nan}
    gap = float(row.get("uo_decision", np.nan) - row.get("uo_signal_decision", np.nan))
    recent = (frame["uo_decision"] - frame["uo_signal_decision"]).dropna().tail(40)
    denom = max(float(recent.abs().max()), 1e-9) if not recent.empty else np.nan
    return {"gap": gap, "abs_gap": abs(gap) if pd.notna(gap) else np.nan, "range_pct": abs(gap) / denom * 100 if pd.notna(gap) and pd.notna(denom) else np.nan}


def make_engine_figure(item: Dict[str, object], hourly_range: str, tactical_range: str, daily_range: str, weekly_range: str) -> go.Figure:
    hourly_df = trim_to_range(item["hourly"], hourly_range)
    tactical_df = trim_to_range(item["tactical"], tactical_range)
    daily_df = trim_to_range(item["daily"], daily_range)
    weekly_df = trim_to_range(item["weekly"], weekly_range)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.04, subplot_titles=(item["hourly_label"], item["tactical_label"], "Daily", "Weekly"))
    for i, (label, df) in enumerate([(item["hourly_label"], hourly_df), (item["tactical_label"], tactical_df), ("Daily", daily_df), ("Weekly", weekly_df)], start=1):
        if df.empty:
            continue
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{label} Price"), row=i, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_20"], name=f"{label} EMA20", line=dict(dash="dot")), row=i, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name=f"{label} SMA50", line=dict(dash="dash")), row=i, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["uo"], name=f"{label} Osc"), row=i, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["uo_signal"], name=f"{label} Signal", line=dict(dash="dot")), row=i, col=1)
    fig.update_layout(height=1200, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_extreme_table(frame_items: List[Tuple[str, pd.Series]]) -> None:
    rows = []
    for label, row in frame_items:
        if row is None or row.empty:
            rows.append({"Frame": label, "State": "No data", "OB internal": np.nan, "OB price": np.nan, "OS internal": np.nan, "OS price": np.nan})
            continue
        rows.append({
            "Frame": label,
            "State": row.get("reason", "n/a"),
            "OB internal": round(float(row.get("ob_internal_score", np.nan)) * 100, 1),
            "OB price": round(float(row.get("ob_price_score", np.nan)) * 100, 1),
            "OS internal": round(float(row.get("os_internal_score", np.nan)) * 100, 1),
            "OS price": round(float(row.get("os_price_score", np.nan)) * 100, 1),
        })
    st.markdown("### Overbought / oversold diagnostics")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


with st.sidebar:
    st.header("Credentials")
    alpaca_key = st.text_input("Alpaca API key", type="password", value=os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret = st.text_input("Alpaca API secret", type="password", value=os.getenv("ALPACA_SECRET_KEY", ""))
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], index=0)

    st.header("Input")
    symbols_text = st.text_area("Paste tickers (comma or line separated)", value="QQQ")
    upload_watchlist = st.file_uploader("Upload results/watchlist CSV", type=["csv"])

    st.header("Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=1)
    history_years = st.selectbox("Historical years", [3, 5, 10], index=1)
    analysis_mode = st.radio("Analysis mode", ["Current", "Historical"], index=0)
    analysis_date = st.date_input("Calendar lookback", value=pd.Timestamp.today().date(), disabled=(analysis_mode == "Current"))
    intraday_source = st.radio("Intraday source", ["proxy", "real", "both"], index=0, format_func=lambda x: {"proxy": "Proxy from Daily", "real": "Real Alpaca", "both": "Both (overlay)"}[x])
    force_refresh = st.checkbox("Force refresh data (clear cache)", value=False)
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

if not run_analysis:
    st.stop()

if not alpaca_key or not alpaca_secret:
    st.error("Enter Alpaca API key and secret.")
    st.stop()

uploaded_symbols = parse_symbol_csv(upload_watchlist)
symbols = clean_symbols(symbols_text, uploaded_symbols)
if not symbols:
    st.error("Provide at least one symbol.")
    st.stop()

all_syms = list(dict.fromkeys(symbols + [benchmark]))
if force_refresh:
    clear_symbol_cache(all_syms)
    fetch_alpaca_daily_batch.clear()
    fetch_alpaca_2hour.clear()
    fetch_alpaca_1hour.clear()

st.info("Loading Alpaca daily data...")
daily_map = fetch_alpaca_daily_batch(all_syms, history_years, alpaca_key, alpaca_secret, feed)
if benchmark not in daily_map or daily_map[benchmark].empty:
    st.error(f"Could not load benchmark {benchmark} from Alpaca.")
    st.stop()
benchmark_daily = daily_map[benchmark]

rows: List[Dict[str, object]] = []
detail: Dict[str, Dict[str, object]] = {}
progress = st.progress(0.0)

for i, sym in enumerate(symbols):
    progress.progress((i + 1) / max(1, len(symbols)))
    daily_raw = daily_map.get(sym, pd.DataFrame())
    if daily_raw.empty:
        rows.append({"Symbol": sym, "Status": "No data"})
        continue

    tactical_real = fetch_alpaca_2hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
    hourly_real = fetch_alpaca_1hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
    hourly_proxy = build_proxy_intraday_from_daily(daily_raw, "1H")
    tactical_proxy = build_proxy_intraday_from_daily(daily_raw, "2H")

    if intraday_source == "real":
        hourly_raw = hourly_real if not hourly_real.empty else hourly_proxy
        tactical_raw = tactical_real if not tactical_real.empty else tactical_proxy
        hourly_label = "1-Hour (Real)" if not hourly_real.empty else "1-Hour (Proxy Fallback)"
        tactical_label = "2-Hour (Real)" if not tactical_real.empty else "2-Hour (Proxy Fallback)"
        hourly_mode = "1H" if not hourly_real.empty else "1H_PROXY"
        tactical_mode = "2H" if not tactical_real.empty else "2H_PROXY"
        hourly_overlay = None
        tactical_overlay = None
    elif intraday_source == "proxy":
        hourly_raw = hourly_proxy
        tactical_raw = tactical_proxy
        hourly_label = "1-Hour (Proxy)"
        tactical_label = "2-Hour (Proxy)"
        hourly_mode = "1H_PROXY"
        tactical_mode = "2H_PROXY"
        hourly_overlay = None
        tactical_overlay = None
    else:
        hourly_raw = hourly_real if not hourly_real.empty else hourly_proxy
        tactical_raw = tactical_real if not tactical_real.empty else tactical_proxy
        hourly_label = "1-Hour (Real)"
        tactical_label = "2-Hour (Real)"
        hourly_mode = "1H" if not hourly_real.empty else "1H_PROXY"
        tactical_mode = "2H" if not tactical_real.empty else "2H_PROXY"
        hourly_overlay = hourly_proxy
        tactical_overlay = tactical_proxy

    weekly_raw = resample_weekly(daily_raw)

    hourly_df = enrich_price_features(hourly_raw, hourly_mode, benchmark_daily)
    tactical_df = enrich_price_features(tactical_raw, tactical_mode, benchmark_daily)
    daily_df = enrich_price_features(daily_raw, "Daily", benchmark_daily)
    weekly_df = enrich_price_features(weekly_raw, "Weekly", benchmark_daily)
    hourly_overlay_df = enrich_price_features(hourly_overlay, "1H_PROXY", benchmark_daily) if hourly_overlay is not None and not hourly_overlay.empty else pd.DataFrame()
    tactical_overlay_df = enrich_price_features(tactical_overlay, "2H_PROXY", benchmark_daily) if tactical_overlay is not None and not tactical_overlay.empty else pd.DataFrame()

    if analysis_mode == "Historical":
        end_dt = pd.Timestamp(analysis_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        hourly_df = hourly_df.loc[hourly_df.index <= end_dt].copy()
        tactical_df = tactical_df.loc[tactical_df.index <= end_dt].copy()
        daily_df = daily_df.loc[daily_df.index <= end_dt].copy()
        weekly_df = weekly_df.loc[weekly_df.index <= end_dt].copy()
        if not hourly_overlay_df.empty:
            hourly_overlay_df = hourly_overlay_df.loc[hourly_overlay_df.index <= end_dt].copy()
        if not tactical_overlay_df.empty:
            tactical_overlay_df = tactical_overlay_df.loc[tactical_overlay_df.index <= end_dt].copy()

    if hourly_df.empty or tactical_df.empty or daily_df.empty or weekly_df.empty:
        rows.append({"Symbol": sym, "Status": "Frame missing"})
        continue

    hourly_row = hourly_df.iloc[-1].copy()
    tactical_row = tactical_df.iloc[-1].copy()
    daily_row = daily_df.iloc[-1].copy()
    weekly_row = weekly_df.iloc[-1].copy()

    hourly_call, hourly_reason = classify_state(hourly_row, structural=False)
    tactical_call, tactical_reason = classify_state(tactical_row, structural=False)
    daily_call, daily_reason = classify_state(daily_row, structural=True)
    weekly_call, weekly_reason = classify_state(weekly_row, structural=True)

    hourly_row["call"], hourly_row["reason"] = hourly_call, hourly_reason
    tactical_row["call"], tactical_row["reason"] = tactical_call, tactical_reason
    daily_row["call"], daily_row["reason"] = daily_call, daily_reason
    weekly_row["call"], weekly_row["reason"] = weekly_call, weekly_reason

    hourly_analogs = summarize_analogs(hourly_df, hourly_row, [1, 3, 6, 12], structural=False)
    tactical_analogs = summarize_analogs(tactical_df, tactical_row, [1, 2, 4, 8], structural=False)
    daily_analogs = summarize_analogs(daily_df, daily_row, [1, 2, 5, 10], structural=True)

    combined = combine(hourly_call, tactical_call, daily_call, weekly_call)
    sev = compute_severity(tactical_df, tactical_row)
    reco = recommendation(hourly_call, tactical_call, daily_call, tactical_analogs)

    rows.append({
        "Symbol": sym,
        "Status": "OK",
        "1H": hourly_call,
        "2H": tactical_call,
        "Daily": daily_call,
        "Weekly": weekly_call,
        "Combined": combined,
        "Overheat Score": round(float(tactical_row.get("exhaustion_score", np.nan)), 1),
        "2H %ile": round(float(tactical_row.get("uo_pctile", np.nan)) * 100, 1),
        "Daily %ile": round(float(daily_row.get("uo_pctile", np.nan)) * 100, 1),
        "RSI14": round(float(daily_row.get("rsi_14", np.nan)), 1),
        "CCI20": round(float(daily_row.get("cci_20", np.nan)), 1),
        "Analog N": int(daily_analogs.get("n", 0)) if daily_analogs else 0,
        "Analog 2d Med": round(float(daily_analogs.get("ret_2_median", np.nan)) * 100, 2) if daily_analogs else np.nan,
        "Analog 2d Up %": round(float(daily_analogs.get("ret_2_p_up", np.nan)) * 100, 1) if daily_analogs else np.nan,
    })

    detail[sym] = {
        "hourly": hourly_df,
        "hourly_label": hourly_label,
        "hourly_call": (hourly_call, hourly_reason),
        "hourly_overlay": hourly_overlay_df,
        "tactical": tactical_df,
        "tactical_overlay": tactical_overlay_df,
        "daily": daily_df,
        "weekly": weekly_df,
        "tactical_call": (tactical_call, tactical_reason),
        "daily_call": (daily_call, daily_reason),
        "weekly_call": (weekly_call, weekly_reason),
        "combined": combined,
        "recommendation": reco,
        "severity": sev,
        "hourly_analogs": hourly_analogs,
        "analog_summary": tactical_analogs,
        "daily_analogs": daily_analogs,
        "cross": distance_to_cross(tactical_row, tactical_df),
        "asof": tactical_df.index.max(),
        "tactical_label": tactical_label,
        "intraday_stale": False,
    }

progress.empty()

results_df = pd.DataFrame(rows)
st.subheader("Ranked Results")
if results_df.empty:
    st.warning("No results generated.")
    st.stop()
results_df = results_df.sort_values("Overheat Score", ascending=False)
st.dataframe(results_df, width="stretch", hide_index=True)
st.download_button("Download results CSV", results_df.to_csv(index=False).encode("utf-8"), "stable_market_engine_hybrid.csv", "text/csv")

valid_symbols = results_df.loc[results_df["Status"] == "OK", "Symbol"].tolist()
if not valid_symbols:
    st.stop()
if "selected_symbol" not in st.session_state or st.session_state.selected_symbol not in valid_symbols:
    st.session_state.selected_symbol = valid_symbols[0]
selected = st.selectbox("Select symbol", valid_symbols, index=valid_symbols.index(st.session_state.selected_symbol))
st.session_state.selected_symbol = selected
item = detail[selected]

st.subheader("Detailed Analysis")
hourly_call, hourly_reason = item["hourly_call"]
tactical_call, tactical_reason = item["tactical_call"]
daily_call, daily_reason = item["daily_call"]
weekly_call, weekly_reason = item["weekly_call"]
tactical_label = item["tactical_label"]
hourly_label = item["hourly_label"]

c1, c2, c3, c4 = st.columns(4)
for col, label, call, reason in [
    (c1, hourly_label, hourly_call, hourly_reason),
    (c2, tactical_label, tactical_call, tactical_reason),
    (c3, "Daily", daily_call, daily_reason),
    (c4, "Weekly", weekly_call, weekly_reason),
]:
    with col:
        emoji = "🟢" if call == "CALL" else "🔴" if call == "PUT" else "🟡"
        st.metric(label, f"{emoji} {call}")
        st.caption(reason)

st.markdown(f"**Combined:** {item['combined']}")
st.markdown(f"**Recommendation:** {item['recommendation']}")

st.markdown("### Frame warnings")
frame_items = [
    (hourly_label, item["hourly"].iloc[-1] if not item["hourly"].empty else pd.Series(dtype=float)),
    (tactical_label, item["tactical"].iloc[-1] if not item["tactical"].empty else pd.Series(dtype=float)),
    ("Daily", item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float)),
    ("Weekly", item["weekly"].iloc[-1] if not item["weekly"].empty else pd.Series(dtype=float)),
]
for label, row in frame_items:
    st.write(frame_warning_message(row, label))

cross = item["cross"]
st.markdown("### Distance to cross")
st.write(
    pd.DataFrame([{
        "Cross gap sign": np.sign(cross["gap"]) if pd.notna(cross["gap"]) else np.nan,
        "Cross gap": cross["gap"],
        "Cross distance %": cross["range_pct"],
    }]),
    width="stretch",
)

render_extreme_table(frame_items)

st.markdown("### Historical analogs")
a1, a2, a3 = st.columns(3)
with a1:
    st.write("**1H analogs**")
    st.json(item["hourly_analogs"] if item["hourly_analogs"] else {})
with a2:
    st.write("**2H analogs**")
    st.json(item["analog_summary"] if item["analog_summary"] else {})
with a3:
    st.write("**Daily analogs**")
    st.json(item["daily_analogs"] if item["daily_analogs"] else {})

st.markdown("### Mega view")
h_range = st.selectbox("Mega 1H Range", RANGE_OPTIONS["1h"], index=2)
t_range = st.selectbox("Mega 2H Range", RANGE_OPTIONS["2h"], index=2)
d_range = st.selectbox("Mega Daily Range", RANGE_OPTIONS["daily"], index=2)
w_range = st.selectbox("Mega Weekly Range", RANGE_OPTIONS["weekly"], index=2)
st.plotly_chart(make_engine_figure(item, h_range, t_range, d_range, w_range), use_container_width=True)

st.markdown("### Frame tabs")
tab1, tab2, tab3, tab4 = st.tabs([hourly_label, tactical_label, "Daily", "Weekly"])

def render_frame_tab(tab, label, df, overlay, analogs, structural=False):
    with tab:
        if df.empty:
            st.warning("No data available.")
            return
        left, right = st.columns([2.0, 1.0], vertical_alignment="top")
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df["ema_20"], name="EMA20", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA50", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=df.index, y=df["tsi"], name="TSI"))
            fig.add_trace(go.Scatter(x=df.index, y=df["tsi_signal"], name="TSI Signal", line=dict(dash="dot")))
            if overlay is not None and not overlay.empty:
                fig.add_trace(go.Scatter(x=overlay.index, y=overlay["tsi"], name="Proxy TSI", line=dict(dash="dot")))
                fig.add_trace(go.Scatter(x=overlay.index, y=overlay["tsi_signal"], name="Proxy Signal", line=dict(dash="dash")))
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            row = df.iloc[-1]
            call, reason = classify_state(row, structural=structural)
            st.write(f"**State:** {call}")
            st.write(f"**Reason:** {reason}")
            st.write(f"**TSI(25,13,7):** {row['tsi']:.1f}")
            st.write(f"**TSI Signal:** {row['tsi_signal']:.1f}")
            st.write(f"**TSI Gap:** {row['tsi_gap']:.2f}")
            st.write(f"**Exhaustion score:** {row['exhaustion_score']:.1f}")
            st.write(f"**RSI(14):** {row['rsi_14']:.1f}")
            st.write(f"**CCI(20):** {row['cci_20']:.1f}")
            st.write(f"**%B(20,2):** {row['pct_b']:.2f}")
            st.write(f"**Dist EMA20:** {row['dist_ema20_pct']*100:.2f}%")
            st.write(f"**Dist VWAP:** {row['dist_vwap_pct']*100:.2f}%")
            st.write(f"**Regime:** {row['regime_bucket']}")
            st.write("**Analogs:**")
            st.json(analogs if analogs else {})

render_frame_tab(tab1, hourly_label, item["hourly"], item["hourly_overlay"], item["hourly_analogs"], structural=False)
render_frame_tab(tab2, tactical_label, item["tactical"], item["tactical_overlay"], item["analog_summary"], structural=False)
render_frame_tab(tab3, "Daily", item["daily"], None, item["daily_analogs"], structural=True)
render_frame_tab(tab4, "Weekly", item["weekly"], None, {}, structural=True)
