from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store_alpaca"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NY_TZ = "America/New_York"

st.set_page_config(page_title="Stable Market Engine Clean", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Utility helpers
# -----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def slope(series: pd.Series, bars: int = 3) -> pd.Series:
    return series.diff(bars) / bars


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


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
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
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
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()


def true_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    min_valid = max(30, window // 5)
    for i, v in enumerate(values):
        if i + 1 < min_valid or not np.isfinite(v):
            continue
        w = values[max(0, i - window + 1) : i + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_valid:
            continue
        out[i] = float((w <= v).mean())
    return pd.Series(out, index=series.index)


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window, min_periods=max(30, window // 5)).mean()
    std = series.rolling(window, min_periods=max(30, window // 5)).std()
    return (series - mean) / std.replace(0, np.nan)


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0.0)
    pv = typical * vol
    pv_sum = pv.rolling(window, min_periods=max(5, window // 4)).sum()
    v_sum = vol.rolling(window, min_periods=max(5, window // 4)).sum()
    return pv_sum / v_sum.replace(0, np.nan)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0.0)
    if isinstance(df.index, pd.DatetimeIndex):
        groups = df.index.date
    else:
        groups = np.arange(len(df))
    cum_pv = (typical * vol).groupby(groups).cumsum()
    cum_v = vol.groupby(groups).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)



def hybrid_normalize(series: pd.Series, window: int) -> pd.Series:
    pct = true_percentile(series, window)
    z = rolling_zscore(series, window).clip(-3, 3)
    z01 = (z + 3) / 6
    lo = series.rolling(window, min_periods=max(20, window // 5)).min()
    hi = series.rolling(window, min_periods=max(20, window // 5)).max()
    mm = (series - lo) / (hi - lo).replace(0, np.nan)
    out = 0.55 * pct + 0.35 * z01 + 0.10 * mm
    return out.clip(0, 1)


def centered_pct(series: pd.Series) -> pd.Series:
    return (series.fillna(0.5) - 0.5) * 2


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out.columns = [str(c).title() for c in out.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    if len(keep) < 4:
        return pd.DataFrame()
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(NY_TZ).tz_localize(None)
    out = out.sort_index()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    return out.dropna(subset=["Open", "High", "Low", "Close"])


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


def clear_symbol_cache(symbols: List[str]) -> None:
    for s in symbols:
        for kind in ["daily", "1hour", "2hour", "hourly_proxy", "2hour_proxy"]:
            p = cache_path(s, kind)
            if p.exists():
                p.unlink(missing_ok=True)


def is_fresh(path: Path, max_hours: int = 18) -> bool:
    if not path.exists():
        return False
    age = pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
    return age < pd.Timedelta(hours=max_hours)


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
    fresh = []
    missing = []
    for s in symbols:
        p = cache_path(s, "daily")
        if is_fresh(p):
            try:
                data_map[s] = pd.read_parquet(p)
                fresh.append(s)
                continue
            except Exception:
                pass
        missing.append(s)
    if not missing:
        return data_map

    client = alpaca_client(key, secret)
    start = (pd.Timestamp.now(tz=NY_TZ) - pd.DateOffset(years=max(years, 5))).normalize().tz_localize(None)
    end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    req = StockBarsRequest(symbol_or_symbols=missing, timeframe=TimeFrame.Day, start=start, end=end, adjustment="raw", feed=feed)
    try:
        raw = client.get_stock_bars(req).df
    except Exception:
        return data_map
    for s in missing:
        sub = _bars_df_for_symbol(raw, s)
        if not sub.empty:
            sub.to_parquet(cache_path(s, "daily"))
            data_map[s] = sub
    return data_map


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
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(30, TimeFrameUnit.Minute), start=start, end=end, adjustment="raw", feed=feed)
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
    # Convert to NY local naive times, filter regular session, then resample 2H anchored at 9:30.
    idx = pd.DatetimeIndex(sub.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    sub.index = idx
    sub = sub.between_time("09:30", "16:00")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    pieces = []
    for _, day_df in sub.groupby(sub.index.date):
        if day_df.empty:
            continue
        anchor = pd.Timestamp(day_df.index[0].date()) + pd.Timedelta(hours=9, minutes=30)
        tmp = day_df.copy()
        shifted = tmp.copy()
        shifted.index = shifted.index - (anchor - pd.Timestamp(day_df.index[0].date())) - pd.Timedelta(hours=9, minutes=30)
        # simpler deterministic windows: [9:30-11:29], [11:30-13:29], [13:30-15:29], [15:30-16:00]
        bins = []
        ranges = [
            (anchor, anchor + pd.Timedelta(hours=2)),
            (anchor + pd.Timedelta(hours=2), anchor + pd.Timedelta(hours=4)),
            (anchor + pd.Timedelta(hours=4), anchor + pd.Timedelta(hours=6)),
            (anchor + pd.Timedelta(hours=6), anchor + pd.Timedelta(hours=6, minutes=30)),
        ]
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
    out = pd.concat(pieces).sort_index() if pieces else pd.DataFrame()
    out = normalize_ohlcv(out)
    if not out.empty:
        out.to_parquet(p)
    return out


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
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=start, end=end, adjustment="raw", feed=feed)
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
    idx = pd.DatetimeIndex(sub.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    sub.index = idx
    sub = sub.between_time("09:30", "16:00")
    sub = normalize_ohlcv(sub)
    if not sub.empty:
        sub.to_parquet(p)
    return sub


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("W-FRI").agg(agg).dropna(how="any")


def time_compressed_proxy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, str, str]:
    # Convert daily bars into tactical approximations using time-compression math.
    # 6.5 trading hours/day -> about 3.25 two-hour bars/day.
    out = df.copy()
    if target == "2hour_proxy":
        label = "2-Hour Proxy"
        timeframe_name = "2hour_proxy"
    else:
        label = "Hourly Proxy"
        timeframe_name = "hourly_proxy"
    return out, timeframe_name, label


def add_ultimate_oscillator(out: pd.DataFrame, timeframe_name: str) -> pd.DataFrame:
    spans = {
        # lower frames are intentionally slowed to reduce noise and align with the trend stack.
        "hourly_proxy": (8, 21, 7),
        "real_1hour": (8, 21, 7),
        "2hour_proxy": (10, 26, 8),
        "real_2hour": (10, 26, 8),
        "daily": (16, 42, 10),
        "weekly": (8, 21, 7),
    }
    pre_smooth_map = {
        "hourly_proxy": 3,
        "real_1hour": 4,
        "2hour_proxy": 3,
        "real_2hour": 5,
        "daily": 5,
        "weekly": 3,
    }
    post_smooth_map = {
        "hourly_proxy": 2,
        "real_1hour": 2,
        "2hour_proxy": 2,
        "real_2hour": 3,
        "daily": 1,
        "weekly": 1,
    }

    fast, slow, sig = spans[timeframe_name]

    # Raw, bounded component transforms are smoother than percentile churn on sparse 2-hour bars.
    tsi_n = np.tanh(out["tsi"].fillna(0.0) / 35.0)
    cci_n = np.tanh(out["cci_20"].fillna(0.0) / 180.0)
    bb_n = ((out["pct_b"].fillna(0.5) - 0.5) * 2.0).clip(-1.25, 1.25)
    vwap_n = np.tanh(out["dist_vwap_pct"].fillna(0.0) * 18.0)
    z_n = np.tanh(out["close_zscore"].fillna(0.0) / 2.5)
    adx_dir = np.sign(out["tsi_gap"].fillna(0.0) + out["uo_seed_dir"].fillna(0.0))
    adx_n = (((out["adx_14"].fillna(18.0) - 18.0) / 22.0).clip(-1.0, 1.0)) * adx_dir.replace(0, 1)

    # Slightly de-emphasize VWAP in higher timeframes; keep it meaningful intraday.
    if timeframe_name in {"real_2hour", "2hour_proxy", "hourly_proxy"}:
        weights = dict(tsi=0.31, cci=0.22, bb=0.14, vwap=0.15, adx=0.10, z=0.08)
    elif timeframe_name == "daily":
        weights = dict(tsi=0.32, cci=0.20, bb=0.16, vwap=0.08, adx=0.12, z=0.12)
    else:
        weights = dict(tsi=0.34, cci=0.18, bb=0.16, vwap=0.06, adx=0.14, z=0.12)

    out["uo_base"] = (
        weights["tsi"] * tsi_n
        + weights["cci"] * cci_n
        + weights["bb"] * bb_n
        + weights["vwap"] * vwap_n
        + weights["adx"] * adx_n
        + weights["z"] * z_n
    )

    # Mild pinning penalty on tactical frames only: if price rises while TSI is flat and CCI fades,
    # keep the oscillator from spuriously re-accelerating.
    if timeframe_name in {"real_2hour", "2hour_proxy", "hourly_proxy"}:
        pin_penalty = 0.12 * out["pinning_up_flag"].fillna(0.0) - 0.12 * out["pinning_down_flag"].fillna(0.0)
        out["uo_base"] = out["uo_base"] - pin_penalty

    pre_smooth = pre_smooth_map[timeframe_name]
    out["uo_base_sm"] = ema(out["uo_base"], pre_smooth) if pre_smooth > 1 else out["uo_base"]
    out["uo_raw"] = ema(out["uo_base_sm"], fast) - ema(out["uo_base_sm"], slow)
    post_smooth = post_smooth_map[timeframe_name]
    out["uo"] = ema(out["uo_raw"], post_smooth) if post_smooth > 1 else out["uo_raw"]
    out["uo_signal"] = ema(out["uo"], sig)
    out["uo_gap"] = out["uo"] - out["uo_signal"]
    out["uo_slope_1"] = out["uo"].diff(1)
    out["uo_slope_3"] = out["uo"].diff(3)
    lookback = 120 if timeframe_name in {"2hour_proxy", "hourly_proxy", "real_1hour", "real_2hour"} else 252
    out["uo_pctile"] = true_percentile(out["uo"], lookback)

    out["uo_above_signal"] = (out["uo"] > out["uo_signal"]).astype(int)
    out["uo_below_signal"] = (out["uo"] < out["uo_signal"]).astype(int)
    out["uo_above_signal_2"] = (out["uo_above_signal"].rolling(2, min_periods=2).sum() == 2).astype(int)
    out["uo_below_signal_2"] = (out["uo_below_signal"].rolling(2, min_periods=2).sum() == 2).astype(int)
    gap_scale = out["uo_gap"].abs().rolling(40, min_periods=10).max().replace(0, np.nan)
    out["uo_gap_strength"] = (out["uo_gap"].abs() / gap_scale).clip(0, 1)
    return out



def enrich_price_features(df: pd.DataFrame, timeframe_name: str, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["ema_10"] = ema(out["Close"], 10)
    out["ema_20"] = ema(out["Close"], 20)
    out["sma_50"] = sma(out["Close"], 50)
    out["atr_14"] = atr(out, 14)
    out["atr_stretch"] = (out["Close"] - out["ema_20"]) / out["atr_14"].replace(0, np.nan)

    out["rsi_14"] = rsi(out["Close"], 14)
    out["rsi_slope_3"] = slope(out["rsi_14"], 3)

    out["cci_20"] = cci(out, 20)
    out["cci_slope_3"] = slope(out["cci_20"], 3)

    out["tsi"], out["tsi_signal"] = tsi(out["Close"], 25, 13, 7)
    out["tsi_gap"] = out["tsi"] - out["tsi_signal"]
    out["tsi_slope_3"] = slope(out["tsi"], 3)

    out["pct_b"] = bollinger_pct_b(out["Close"], 20, 2)
    out["adx_14"] = adx(out, 14)
    out["price_slope_3"] = slope(out["Close"], 3)
    out["dist_ema20_pct"] = (out["Close"] / out["ema_20"]) - 1

    if timeframe_name == "hourly_proxy":
        out["vwap"] = session_vwap(out)
    elif timeframe_name == "real_2hour":
        # Session VWAP resets create step-changes on sparse 2-hour bars; rolling VWAP is smoother.
        out["vwap"] = rolling_vwap(out, 12)
    elif timeframe_name == "2hour_proxy":
        out["vwap"] = rolling_vwap(out, 10)
    else:
        out["vwap"] = rolling_vwap(out, 20)
    out["dist_vwap_pct"] = (out["Close"] / out["vwap"]) - 1

    z_win = 120 if timeframe_name in {"hourly_proxy", "2hour_proxy", "real_1hour", "real_2hour"} else 252
    out["close_zscore"] = rolling_zscore(out["Close"], z_win)
    out["close_zscore_slope_3"] = slope(out["close_zscore"], 3)

    out["close_in_range"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["High"] - out[["Close", "Open"]].max(axis=1)) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["candle_score"] = 50 + out["upper_wick_pct"].fillna(0) * 30 - (out["close_in_range"].fillna(0.5) - 0.5) * 20

    tsi_flat_thresh = out["tsi_slope_3"].abs().rolling(50, min_periods=10).median().fillna(0.02)
    out["price_change_while_tsi_flat"] = np.where(
        out["tsi_slope_3"].abs() <= tsi_flat_thresh,
        out["Close"].pct_change(3),
        0.0,
    )
    out["pinning_up_flag"] = (
        (out["price_change_while_tsi_flat"] > 0.012)
        & (out["cci_slope_3"] < 0)
        & (out["pct_b"] > 0.75)
    ).astype(float)
    out["pinning_down_flag"] = (
        (out["price_change_while_tsi_flat"] < -0.012)
        & (out["cci_slope_3"] > 0)
        & (out["pct_b"] < 0.25)
    ).astype(float)
    out["uo_seed_dir"] = np.tanh(out["price_slope_3"].fillna(0.0) * 20.0)

    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(out.index).ffill()
        out["rs_bench_slope_5"] = slope(out["Close"] / aligned, 5)
    else:
        out["rs_bench_slope_5"] = 0.0

    win = 120 if timeframe_name in {"hourly_proxy", "2hour_proxy", "real_1hour", "real_2hour"} else 252
    for col in [
        "rsi_14", "cci_20", "tsi", "pct_b", "atr_stretch", "adx_14",
        "dist_ema20_pct", "dist_vwap_pct", "close_zscore"
    ]:
        out[f"{col}_pctile"] = hybrid_normalize(out[col], win)

    out = add_ultimate_oscillator(out, timeframe_name)
    return out


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [1, 2, 5]:
        out[f"fwd_ret_{n}"] = out["Close"].shift(-n) / out["Close"] - 1
    return out


ANALOG_FEATURES = [
    "uo_pctile", "uo_gap", "uo_slope_1", "uo_slope_3",
    "tsi", "tsi_gap", "tsi_slope_3", "tsi_pctile",
    "cci_20", "cci_slope_3", "cci_20_pctile",
    "pct_b", "pct_b_pctile",
    "dist_vwap_pct", "dist_vwap_pctile",
    "close_zscore", "close_zscore_slope_3", "close_zscore_pctile",
    "adx_14", "adx_14_pctile",
    "price_change_while_tsi_flat", "pinning_up_flag", "pinning_down_flag",
    "rs_bench_slope_5", "candle_score",
]


def find_analogs(frame: pd.DataFrame, current_ts: pd.Timestamp, n: int = 25) -> pd.DataFrame:
    enriched = add_forward_returns(frame)
    use = [c for c in ANALOG_FEATURES if c in enriched.columns]
    if current_ts not in enriched.index or len(use) < 8:
        return pd.DataFrame()
    cur_pos = enriched.index.get_loc(current_ts)
    pool = enriched.iloc[:max(0, cur_pos - 10)].dropna(subset=use + ["fwd_ret_1", "fwd_ret_2", "fwd_ret_5"]).copy()
    if len(pool) < 50:
        return pd.DataFrame()
    current = enriched.loc[current_ts, use].astype(float)
    X = pool[use].astype(float)
    std = X.std().replace(0, np.nan)
    z = ((X - current) / std).fillna(0.0)
    pool["distance"] = np.sqrt((z.to_numpy() ** 2).sum(axis=1))
    pool["similarity"] = 1 / (1 + pool["distance"])
    return pool.nsmallest(n, "distance").copy()


def summarize_analogs(analogs: pd.DataFrame) -> Dict[str, float]:
    if analogs.empty:
        return {}
    w = analogs["similarity"].fillna(1.0)
    out = {"n": float(len(analogs))}
    for n in [1, 2, 5]:
        c = f"fwd_ret_{n}"
        vals = analogs[c].fillna(0)
        out[f"ret_{n}_median"] = float(vals.median())
        out[f"ret_{n}_p_up"] = float(np.average((vals > 0).astype(float), weights=w))
        out[f"ret_{n}_p_down"] = float(np.average((vals < 0).astype(float), weights=w))
    return out


def classify_timeframe_call(row: pd.Series, timeframe: str) -> Tuple[str, str]:
    if row is None or row.empty:
        return "NO DATA", "No data"

    uo_pct = float(row.get("uo_pctile", 0.5))
    uo_gap = float(row.get("uo_gap", 0.0))
    uo_slope3 = float(row.get("uo_slope_3", 0.0))
    rsi_val = float(row.get("rsi_14", 50.0))
    cci_val = float(row.get("cci_20", 0.0))
    pct_b = float(row.get("pct_b", 0.5))
    gap_strength = float(row.get("uo_gap_strength", 0.0))

    is_tactical = timeframe in {"2hour_proxy", "hourly_proxy", "real_1hour", "real_2hour"}
    is_structural = timeframe in {"daily", "weekly"}

    # Fast tactical layer can respond quickly, but do not invert the heat structure.
    if is_tactical and uo_pct > 0.90 and uo_gap < 0 and uo_slope3 < 0 and rsi_val > 68:
        return "PUT", "Tactical momentum rolling from elevated zone"
    if is_tactical and uo_pct > 0.78 and uo_gap > 0 and uo_slope3 >= 0:
        return "CALL", "Composite rising above signal"
    if is_tactical and uo_pct < 0.20 and uo_gap > 0 and uo_slope3 > 0 and rsi_val < 38:
        return "CALL", "Turning up from washed-out zone"
    if is_tactical and uo_gap < 0 and uo_slope3 < 0:
        return "PUT", "Composite below signal and falling"

    # Daily/weekly require confirmed gap persistence and stronger gap before a regime cross counts.
    if is_structural:
        above2 = int(row.get("uo_above_signal_2", 0)) == 1
        below2 = int(row.get("uo_below_signal_2", 0)) == 1
        if uo_pct > 0.85 and below2 and uo_slope3 < 0 and gap_strength >= 0.18 and rsi_val > 65:
            return "PUT", "Confirmed daily rollover from elevated zone"
        if uo_pct < 0.18 and above2 and uo_slope3 > 0 and gap_strength >= 0.18 and rsi_val < 40:
            return "CALL", "Confirmed turn up from washed-out zone"
        if above2 and uo_slope3 > 0 and gap_strength >= 0.15:
            return "CALL", "Confirmed composite above signal"
        if below2 and uo_slope3 < 0 and gap_strength >= 0.15:
            return "PUT", "Confirmed composite below signal"
        return "NEUTRAL", "No confirmed structural cross"

    if uo_gap > 0 and uo_slope3 > 0:
        return "CALL", "Composite rising above signal"
    if uo_gap < 0 and uo_slope3 < 0:
        return "PUT", "Composite below signal and falling"
    return "NEUTRAL", "Mixed state"


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


def recommendation(call: str, severity: Dict[str, float], analogs: Dict[str, float]) -> str:
    if call == "CALL":
        if severity.get("late_cycle_flag", 0):
            return "CALL, but extended"
        if analogs and analogs.get("ret_2_p_up", 0.5) > 0.62:
            return "CALL / analogs supportive"
        return "CALL"
    if call == "PUT":
        if analogs and analogs.get("ret_2_p_down", 0.5) > 0.62:
            return "PUT / analogs confirm"
        return "PUT setup forming"
    if severity.get("late_cycle_flag", 0):
        return "WAIT, aging trend"
    return "NEUTRAL / mixed"


def combine(proxy_call: str, daily_call: str, weekly_call: str) -> str:
    score = {"CALL": 1, "PUT": -1, "NEUTRAL": 0, "NO DATA": 0}
    net = 0.30 * score.get(proxy_call, 0) + 0.45 * score.get(daily_call, 0) + 0.25 * score.get(weekly_call, 0)
    if net >= 0.55:
        return "CALL"
    if net <= -0.55:
        return "PUT"
    if daily_call == "CALL" and weekly_call == "CALL":
        return "CALL ON PULLBACK"
    return "NEUTRAL"


def align_asof(index: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if getattr(index, "tz", None) is not None and ts.tzinfo is None:
        return ts.tz_localize(index.tz)
    if getattr(index, "tz", None) is None and ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def slice_asof(df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    end_ts = align_asof(df.index, pd.Timestamp(analysis_date) + pd.Timedelta(hours=23, minutes=59, seconds=59))
    return df.loc[df.index <= end_ts].copy()


def distance_to_cross(row: pd.Series, frame: pd.DataFrame) -> Dict[str, float]:
    if row is None or row.empty or frame.empty:
        return {"gap": np.nan, "abs_gap": np.nan, "range_pct": np.nan}
    gap = float(row.get("uo", np.nan) - row.get("uo_signal", np.nan))
    recent = (frame["uo"] - frame["uo_signal"]).dropna().tail(40)
    denom = max(float(recent.abs().max()), 1e-9) if not recent.empty else np.nan
    return {"gap": gap, "abs_gap": abs(gap) if pd.notna(gap) else np.nan, "range_pct": abs(gap) / denom * 100 if pd.notna(gap) and pd.notna(denom) else np.nan}


def frame_warning_message(row: pd.Series, timeframe_label: str, structural_bias: str = "") -> str:
    if row is None or row.empty:
        return f"{timeframe_label}: no data."
    uo_gap = float(row.get("uo_gap", 0.0))
    uo_pct = float(row.get("uo_pctile", 0.5))
    price_slope = float(row.get("price_slope_3", 0.0))
    tsi_slope = float(row.get("tsi_slope_3", 0.0))
    cci_slope = float(row.get("cci_slope_3", 0.0))
    pin_up = int(row.get("pinning_up_flag", 0)) == 1
    pin_dn = int(row.get("pinning_down_flag", 0)) == 1
    bullish = (uo_pct < 0.20 and uo_gap > 0) or pin_up
    bearish = (uo_pct > 0.80 and uo_gap < 0) or pin_dn
    if bullish and price_slope <= 0:
        return f"{timeframe_label}: oversold and improving, but price has not lifted yet — early bullish divergence only, confirmation pending."
    if bullish and price_slope > 0:
        return f"{timeframe_label}: washed-out rebound is starting to confirm with price improvement."
    if bearish and price_slope >= 0:
        return f"{timeframe_label}: exhausted/rolling, but price has not broken yet — be careful pressing shorts before confirmation."
    if bearish and price_slope < 0:
        return f"{timeframe_label}: exhaustion is starting to confirm with price weakness."
    if abs(tsi_slope) < 0.05 and cci_slope < 0 and price_slope >= 0:
        return f"{timeframe_label}: momentum is fading under the surface while price still holds — caution, but not a confirmed short yet."
    if abs(tsi_slope) < 0.05 and cci_slope > 0 and price_slope <= 0:
        return f"{timeframe_label}: internal momentum is improving while price still lags — watch for upside confirmation."
    if structural_bias:
        return f"{timeframe_label}: aligned with the broader {structural_bias.lower()} bias, with no special exhaustion warning right now."
    return f"{timeframe_label}: mixed state, no clear exhaustion or washout warning."


def traffic_state(call: str, row: pd.Series) -> Tuple[str, str]:
    if row is None or row.empty:
        return "⚪", "No data"
    pct = float(row.get("uo_pctile", 0.5))
    gap = float(row.get("uo_gap", 0.0))
    slope3 = float(row.get("uo_slope_3", 0.0))
    if call == "CALL":
        if pct < 0.20 and gap > 0 and slope3 > 0:
            return "🟢", "Bull turn"
        return "🟢", "Bullish"
    if call == "PUT":
        if pct > 0.80 and gap < 0 and slope3 < 0:
            return "🔴", "Bear roll"
        return "🔴", "Bearish"
    if pct > 0.80 or pct < 0.20:
        return "🟡", "Extreme"
    return "🟡", "Neutral"


def render_traffic_lights(frame_items: List[Tuple[str, str, pd.Series]]) -> None:
    st.markdown("### Traffic lights")
    cols = st.columns(len(frame_items))
    for col, (label, call, row) in zip(cols, frame_items):
        emoji, sub = traffic_state(call, row)
        col.markdown(f"<div style='text-align:center;font-size:40px;line-height:1.1'>{emoji}</div>", unsafe_allow_html=True)
        col.markdown(f"**{label}**")
        col.caption(f"{call} · {sub}")


def plot_dashboard(symbol: str, hourly_df: pd.DataFrame, tactical_df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, asof_date: pd.Timestamp, tactical_label: str) -> None:
    fig = make_subplots(
        rows=5, cols=1, vertical_spacing=0.04,
        subplot_titles=[f"{symbol} Daily Price (as of {pd.Timestamp(asof_date).date()})", "1-Hour Ultimate Oscillator", f"{tactical_label} Ultimate Oscillator", "Daily Ultimate Oscillator", "Weekly Ultimate Oscillator"],
        row_heights=[0.30, 0.18, 0.18, 0.18, 0.16],
    )
    if not daily_df.empty:
        d = daily_df.tail(260)
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Daily"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["ema_20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["sma_50"], name="SMA50", line=dict(color="blue")), row=1, col=1)
    for row_num, frame, nm in zip([2,3,4,5], [hourly_df.tail(260), tactical_df.tail(260), daily_df.tail(260), weekly_df.tail(160)], ["Hourly","Tactical","Daily","Weekly"]):
        if frame.empty:
            continue
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo"], name=f"{nm} UO", line=dict(color="red", width=2.3)), row=row_num, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo_signal"], name=f"{nm} Signal", line=dict(color="black", width=1.3)), row=row_num, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row_num, col=1)
    fig.update_layout(height=1380, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")


# -----------------------------
# Sidebar
# -----------------------------
st.title("📈 Stable Market Engine Clean")
st.caption("Alpaca-only | raw prices | smooth ultimate oscillator (TSI/CCI/%B/VWAP/ADX/ZScore) | proxy default | 1h/2h/daily/weekly stack")

with st.sidebar:
    st.header("Credentials")
    alpaca_key = st.text_input("Alpaca API key", type="password")
    alpaca_secret = st.text_input("Alpaca API secret", type="password")
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], index=0)

    st.header("Input")
    symbols_text = st.text_area("Paste tickers (comma or line separated)", value="QQQ, SMH, INTC, NVDA, AMD, XLF", height=110)
    upload_watchlist = st.file_uploader("Upload results/watchlist CSV", type=["csv"])

    st.header("Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=0)
    history_years = st.selectbox("Historical years", [3, 5, 10], index=1)
    analysis_mode = st.radio("Analysis mode", ["Current", "Historical"], index=0)
    analysis_date = st.date_input("Calendar lookback", value=pd.Timestamp.today().date(), disabled=(analysis_mode == "Current"))
    intraday_source = st.radio("Intraday source", ["proxy", "real"], index=0, format_func=lambda x: {"proxy": "Proxy (default)", "real": "Real Alpaca"}[x])
    force_refresh = st.checkbox("Force refresh data (clear cache)", value=False)
    run_analysis = st.button("Run Analysis", type="primary", width="stretch")

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

    if intraday_source == "real":
        tactical_raw = fetch_alpaca_2hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
        tactical_timeframe = "real_2hour"
        tactical_label = "2-Hour (Real)"
        if tactical_raw.empty:
            tactical_raw, tactical_timeframe, _ = time_compressed_proxy(daily_raw, "2hour_proxy")
            tactical_label = "2-Hour (Proxy Fallback)"

        hourly_raw = fetch_alpaca_1hour(sym, months=12, key=alpaca_key, secret=alpaca_secret, feed=feed)
        if hourly_raw.empty:
            hourly_raw, hourly_timeframe, _ = time_compressed_proxy(daily_raw, "hourly_proxy")
            hourly_label = "1-Hour (Proxy Fallback)"
        else:
            hourly_timeframe, hourly_label = "real_1hour", "1-Hour (Real)"
    else:
        tactical_raw, tactical_timeframe, _ = time_compressed_proxy(daily_raw, "2hour_proxy")
        tactical_label = "2-Hour (Proxy)"
        hourly_raw, hourly_timeframe, _ = time_compressed_proxy(daily_raw, "hourly_proxy")
        hourly_label = "1-Hour (Proxy)"
    weekly_raw = resample_weekly(daily_raw)
    hourly_df = enrich_price_features(hourly_raw, hourly_timeframe, benchmark_daily)
    tactical_df = enrich_price_features(tactical_raw, tactical_timeframe, benchmark_daily)
    daily_df = enrich_price_features(daily_raw, "daily", benchmark_daily)
    weekly_df = enrich_price_features(weekly_raw, "weekly", benchmark_daily)

    asof = pd.Timestamp.today().normalize() if analysis_mode == "Current" else pd.Timestamp(analysis_date)
    hourly_view = slice_asof(hourly_df, asof)
    tactical_view = slice_asof(tactical_df, asof)
    daily_view = slice_asof(daily_df, asof)
    weekly_view = slice_asof(weekly_df, asof)
    if daily_view.empty:
        rows.append({"Symbol": sym, "Status": "No data on date"})
        continue

    hourly_row = hourly_view.iloc[-1] if not hourly_view.empty else pd.Series(dtype=float)
    tactical_row = tactical_view.iloc[-1] if not tactical_view.empty else pd.Series(dtype=float)
    daily_row = daily_view.iloc[-1]
    weekly_row = weekly_view.iloc[-1] if not weekly_view.empty else pd.Series(dtype=float)

    hourly_call, hourly_reason = classify_timeframe_call(hourly_row, hourly_timeframe)
    tactical_call, tactical_reason = classify_timeframe_call(tactical_row, tactical_timeframe)
    daily_call, daily_reason = classify_timeframe_call(daily_row, "daily")
    weekly_call, weekly_reason = classify_timeframe_call(weekly_row, "weekly")
    sev = compute_severity(tactical_view, tactical_row)
    analogs = find_analogs(daily_df, daily_row.name)
    analog_summary = summarize_analogs(analogs)
    reco = recommendation(tactical_call if tactical_call != "NEUTRAL" else daily_call, sev, analog_summary)
    combined = combine(tactical_call, daily_call, weekly_call)
    cross = distance_to_cross(tactical_row, tactical_view)
    overheat_score = (
        (float(tactical_row.get("uo_pctile", 0.5)) * 0.30)
        + (float(daily_row.get("uo_pctile", 0.5)) * 0.35)
        + (min(max(float(daily_row.get("rsi_14", 50)) / 100, 0), 1) * 0.15)
        + (min(max((float(daily_row.get("cci_20", 0)) + 200) / 400, 0), 1) * 0.20)
    ) * 100

    rows.append({
        "Symbol": sym,
        "Status": "OK",
        "As Of": str(daily_row.name.date()),
        "Hourly": hourly_call,
        "Tactical": tactical_call,
        "Daily": daily_call,
        "Weekly": weekly_call,
        "Combined": combined,
        "Recommendation": reco,
        "Price": round(float(daily_row.get("Close", np.nan)), 2),
        "Overheat Score": round(overheat_score, 1),
        "Tactical %ile": round(float(tactical_row.get("uo_pctile", np.nan)) * 100, 1) if not tactical_row.empty else np.nan,
        "Daily %ile": round(float(daily_row.get("uo_pctile", np.nan)) * 100, 1),
        "RSI14": round(float(daily_row.get("rsi_14", np.nan)), 1),
        "CCI20": round(float(daily_row.get("cci_20", np.nan)), 1),
        "Analog N": int(analog_summary.get("n", 0)) if analog_summary else 0,
        "Analog 2d Med": round(float(analog_summary.get("ret_2_median", np.nan)) * 100, 2) if analog_summary else np.nan,
        "Analog 2d Up %": round(float(analog_summary.get("ret_2_p_up", np.nan)) * 100, 1) if analog_summary else np.nan,
    })
    detail[sym] = {
        "hourly": hourly_view,
        "hourly_label": hourly_label,
        "hourly_call": (hourly_call, hourly_reason),
        "tactical": tactical_view,
        "daily": daily_view,
        "weekly": weekly_view,
        "tactical_call": (tactical_call, tactical_reason),
        "daily_call": (daily_call, daily_reason),
        "weekly_call": (weekly_call, weekly_reason),
        "combined": combined,
        "recommendation": reco,
        "severity": sev,
        "analogs": analogs,
        "analog_summary": analog_summary,
        "cross": cross,
        "asof": asof,
        "tactical_label": tactical_label,
    }
progress.empty()

results_df = pd.DataFrame(rows)
st.subheader("Ranked Results")
if results_df.empty:
    st.warning("No results generated.")
    st.stop()
results_df = results_df.sort_values("Overheat Score", ascending=False)
st.dataframe(results_df, width="stretch", hide_index=True)
st.download_button("Download results CSV", results_df.to_csv(index=False).encode("utf-8"), "stable_market_engine_clean.csv", "text/csv")

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

render_traffic_lights([
    (hourly_label, hourly_call, item["hourly"].iloc[-1] if not item["hourly"].empty else pd.Series(dtype=float)),
    (tactical_label, tactical_call, item["tactical"].iloc[-1] if not item["tactical"].empty else pd.Series(dtype=float)),
    ("Daily", daily_call, item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float)),
    ("Weekly", weekly_call, item["weekly"].iloc[-1] if not item["weekly"].empty else pd.Series(dtype=float)),
])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(hourly_label, hourly_call)
c2.metric(tactical_label, tactical_call)
c3.metric("Daily", daily_call)
c4.metric("Weekly", weekly_call)
c5.metric("Combined", item["combined"])
st.markdown(f"**{hourly_label} reason:** {hourly_reason}")
st.markdown(f"**{tactical_label} reason:** {tactical_reason}")
st.markdown(f"**Daily reason:** {daily_reason}")
st.markdown(f"**Weekly reason:** {weekly_reason}")
st.markdown(f"**Recommendation:** {item['recommendation']}")

structural_bias = item["combined"] if item["combined"] != "NEUTRAL" else daily_call
st.markdown("### Frame warnings")
st.write(frame_warning_message(item["hourly"].iloc[-1] if not item["hourly"].empty else pd.Series(dtype=float), hourly_label, structural_bias))
st.write(frame_warning_message(item["tactical"].iloc[-1] if not item["tactical"].empty else pd.Series(dtype=float), tactical_label, structural_bias))
st.write(frame_warning_message(item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float), "Daily", structural_bias))
st.write(frame_warning_message(item["weekly"].iloc[-1] if not item["weekly"].empty else pd.Series(dtype=float), "Weekly", structural_bias))

cross = item["cross"]
cd1, cd2, cd3 = st.columns(3)
cd1.metric("Distance to cross", f"{cross.get('abs_gap', np.nan):.4f}" if pd.notna(cross.get("abs_gap", np.nan)) else "n/a")
gap_label = "Above signal" if cross.get("gap", np.nan) >= 0 else ("Below signal" if pd.notna(cross.get("gap", np.nan)) else "n/a")
cd2.metric("Cross gap sign", gap_label)
cd3.metric("Cross distance %", f"{cross.get('range_pct', np.nan):.1f}%" if pd.notna(cross.get("range_pct", np.nan)) else "n/a")

sev = item["severity"]
sv1, sv2, sv3 = st.columns(3)
sv1.metric("Bars > 90", str(sev.get("bars_above_90", 0)))
sv2.metric("Bars < 10", str(sev.get("bars_below_10", 0)))
sv3.metric("Late-cycle", "Yes" if sev.get("late_cycle_flag", 0) else "No")

# As-of table
last_h = item["hourly"].iloc[-1] if not item["hourly"].empty else pd.Series(dtype=float)
last_t = item["tactical"].iloc[-1] if not item["tactical"].empty else pd.Series(dtype=float)
last_d = item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float)
last_w = item["weekly"].iloc[-1] if not item["weekly"].empty else pd.Series(dtype=float)
asof_table = pd.DataFrame([
    {"Frame": hourly_label, "Date": str(last_h.name.date()) if not last_h.empty else "n/a", "Price": round(float(last_h.get("Close", np.nan)), 2) if not last_h.empty else np.nan, "UO": round(float(last_h.get("uo", np.nan)), 4) if not last_h.empty else np.nan, "Signal": round(float(last_h.get("uo_signal", np.nan)), 4) if not last_h.empty else np.nan, "Percentile": round(float(last_h.get("uo_pctile", np.nan))*100, 1) if not last_h.empty else np.nan, "Candle Score": round(float(last_h.get("candle_score", np.nan)), 1) if not last_h.empty else np.nan},
    {"Frame": tactical_label, "Date": str(last_t.name.date()) if not last_t.empty else "n/a", "Price": round(float(last_t.get("Close", np.nan)), 2) if not last_t.empty else np.nan, "UO": round(float(last_t.get("uo", np.nan)), 4) if not last_t.empty else np.nan, "Signal": round(float(last_t.get("uo_signal", np.nan)), 4) if not last_t.empty else np.nan, "Percentile": round(float(last_t.get("uo_pctile", np.nan))*100, 1) if not last_t.empty else np.nan, "Candle Score": round(float(last_t.get("candle_score", np.nan)), 1) if not last_t.empty else np.nan},
    {"Frame": "Daily", "Date": str(last_d.name.date()) if not last_d.empty else "n/a", "Price": round(float(last_d.get("Close", np.nan)), 2) if not last_d.empty else np.nan, "UO": round(float(last_d.get("uo", np.nan)), 4) if not last_d.empty else np.nan, "Signal": round(float(last_d.get("uo_signal", np.nan)), 4) if not last_d.empty else np.nan, "Percentile": round(float(last_d.get("uo_pctile", np.nan))*100, 1) if not last_d.empty else np.nan, "Candle Score": round(float(last_d.get("candle_score", np.nan)), 1) if not last_d.empty else np.nan},
    {"Frame": "Weekly", "Date": str(last_w.name.date()) if not last_w.empty else "n/a", "Price": round(float(last_w.get("Close", np.nan)), 2) if not last_w.empty else np.nan, "UO": round(float(last_w.get("uo", np.nan)), 4) if not last_w.empty else np.nan, "Signal": round(float(last_w.get("uo_signal", np.nan)), 4) if not last_w.empty else np.nan, "Percentile": round(float(last_w.get("uo_pctile", np.nan))*100, 1) if not last_w.empty else np.nan, "Candle Score": round(float(last_w.get("candle_score", np.nan)), 1) if not last_w.empty else np.nan},
])
st.markdown("### As-of values")
st.dataframe(asof_table, width="stretch", hide_index=True)

st.markdown("### Historical analogs")
analog_summary = item["analog_summary"]
if analog_summary:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Analog count", str(int(analog_summary.get("n", 0))))
    a2.metric("2d median", f"{analog_summary.get('ret_2_median', np.nan)*100:.2f}%" if pd.notna(analog_summary.get("ret_2_median", np.nan)) else "n/a")
    a3.metric("2d up %", f"{analog_summary.get('ret_2_p_up', np.nan)*100:.1f}%" if pd.notna(analog_summary.get("ret_2_p_up", np.nan)) else "n/a")
    a4.metric("5d median", f"{analog_summary.get('ret_5_median', np.nan)*100:.2f}%" if pd.notna(analog_summary.get("ret_5_median", np.nan)) else "n/a")
    analogs = item["analogs"]
    if analogs is not None and not analogs.empty:
        show_cols = [c for c in ["Close", "uo", "uo_signal", "distance", "similarity", "fwd_ret_1", "fwd_ret_2", "fwd_ret_5"] if c in analogs.columns]
        show = analogs[show_cols].head(12).copy()
        try:
            show.index = show.index.strftime("%Y-%m-%d")
        except Exception:
            pass
        st.dataframe(show, width="stretch")
else:
    st.caption("No analog set available for this as-of date.")

plot_dashboard(selected, item["hourly"], item["tactical"], item["daily"], item["weekly"], item["asof"], tactical_label)
