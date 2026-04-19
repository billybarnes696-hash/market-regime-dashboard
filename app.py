"""
Stable Market Engine v4
Optimized Data Fetching: DefeatBeta (full history) + Yahoo (only last 10 days)
"""

import io
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Stable Market Engine v4", layout="wide", initial_sidebar_state="expanded")

SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLI": "Industrials", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLV": "Health Care", "XLB": "Materials",
    "XLU": "Utilities", "XLC": "Communication Services", "SMH": "Semiconductors",
}

DAILY_ANALOG_FEATURES = [
    "uo_pctile", "uo_gap", "uo_slope_1", "uo_slope_3", "uo_cross_up", "uo_cross_down",
    "rsi_14_pctile", "rsi_slope_3", "cci_20_pctile", "cci_slope_3", "tsi_pctile", "tsi_gap",
    "pct_b_pctile", "atr_stretch_pctile", "dist_ema20_pctile", "rs_bench_slope_5", "price_slope_3",
    "adx_14_pctile",
]

HOURLY_ANALOG_FEATURES = [
    "uo_pctile", "uo_gap", "uo_slope_1", "uo_slope_3", "uo_cross_up", "uo_cross_down",
    "rsi_14_pctile", "rsi_slope_3", "cci_20_pctile", "cci_slope_3", "tsi_pctile", "tsi_gap",
    "pct_b_pctile", "dist_ema20_pctile", "dist_vwap_pctile", "price_slope_3", "adx_14_pctile",
]

for key in ["detail_data", "results_df"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key == "detail_data" else None


# ============================================================================
# INDICATOR FUNCTIONS (same as before)
# ============================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


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


def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    min_valid = max(20, window // 5)
    for i, v in enumerate(values):
        if i + 1 < min_valid or not np.isfinite(v):
            continue
        window_vals = values[max(0, i - window + 1): i + 1]
        window_vals = window_vals[np.isfinite(window_vals)]
        if len(window_vals) < max(10, window // 10):
            continue
        out[i] = float((window_vals <= v).mean())
    return pd.Series(out, index=series.index)


def slope(series: pd.Series, bars: int = 3) -> pd.Series:
    return series.diff(bars) / bars


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna(how="any")


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    if isinstance(df.index, pd.DatetimeIndex):
        groups = df.index.date
    else:
        groups = np.arange(len(df))
    cum_pv = pv.groupby(groups).cumsum()
    cum_v = df["Volume"].groupby(groups).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename_map = {}
    for c in out.columns:
        lc = str(c).strip().lower()
        if lc in {"date", "datetime", "timestamp", "time"}:
            rename_map[c] = "Date"
        elif lc in {"open", "o"}:
            rename_map[c] = "Open"
        elif lc in {"high", "h"}:
            rename_map[c] = "High"
        elif lc in {"low", "l"}:
            rename_map[c] = "Low"
        elif lc in {"close", "adj close", "adj_close", "c"}:
            rename_map[c] = "Close"
        elif lc in {"volume", "vol", "v"}:
            rename_map[c] = "Volume"
        elif lc in {"ticker", "symbol"}:
            rename_map[c] = "Ticker"
    out = out.rename(columns=rename_map)
    if "Date" not in out.columns:
        out = out.reset_index()
        if "index" in out.columns and "Date" not in out.columns:
            out = out.rename(columns={"index": "Date"})
    need = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in need):
        return pd.DataFrame()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).set_index("Date")
    return out[["Open", "High", "Low", "Close", "Volume"]]


def normalize_history_text(text_blob: str, fallback_ticker: str = "") -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not text_blob.strip():
        return out
    try:
        df = pd.read_csv(io.StringIO(text_blob), skipinitialspace=True)
    except Exception:
        return out
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    ticker_col = cols_lower.get("ticker") or cols_lower.get("symbol")
    if ticker_col is not None:
        for tkr, grp in df.groupby(ticker_col):
            norm = normalize_ohlcv_columns(grp.drop(columns=[ticker_col]))
            tkr = str(tkr).strip().upper()
            if tkr and not norm.empty:
                out[tkr] = norm
        return out
    norm = normalize_ohlcv_columns(df)
    ticker = fallback_ticker.upper()
    if ticker and not norm.empty:
        out[ticker] = norm
    return out


def load_history_uploads(files) -> Dict[str, pd.DataFrame]:
    history_map: Dict[str, pd.DataFrame] = {}
    if not files:
        return history_map
    for file in files:
        try:
            file.seek(0)
        except Exception:
            pass
        ticker = Path(file.name).stem.strip().upper()
        name = file.name.lower()
        parsed: Dict[str, pd.DataFrame] = {}
        if name.endswith(".csv"):
            try:
                text_blob = file.getvalue().decode("utf-8", errors="ignore")
                parsed = normalize_history_text(text_blob, fallback_ticker=ticker)
            except Exception:
                parsed = {}
        else:
            try:
                raw = pd.read_excel(file)
                cols_lower = {str(c).strip().lower(): c for c in raw.columns}
                ticker_col = cols_lower.get("ticker") or cols_lower.get("symbol")
                if ticker_col is not None:
                    for tkr, grp in raw.groupby(ticker_col):
                        norm = normalize_ohlcv_columns(grp.drop(columns=[ticker_col]))
                        tkr = str(tkr).strip().upper()
                        if tkr and not norm.empty:
                            parsed[tkr] = norm
                else:
                    norm = normalize_ohlcv_columns(raw)
                    if ticker and not norm.empty:
                        parsed[ticker] = norm
            except Exception:
                parsed = {}
        for tkr, df in parsed.items():
            if tkr in history_map:
                history_map[tkr] = pd.concat([history_map[tkr], df]).drop_duplicates().sort_index()
            else:
                history_map[tkr] = df
    return history_map


def merge_history_with_recent(base_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    """Merge base history with recent Yahoo data, keeping latest values"""
    if base_df is None or base_df.empty:
        return recent_df.copy() if recent_df is not None else pd.DataFrame()
    if recent_df is None or recent_df.empty:
        return base_df.copy()
    cutoff = pd.to_datetime(base_df.index.max()) - pd.Timedelta(days=3)
    recent_only = recent_df[recent_df.index >= cutoff]
    merged = pd.concat([base_df, recent_only])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


# ============================================================================
# OPTIMIZED DATA FETCHING - KEY FIX
# ============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo_prices(ticker: str, interval: str, period: Optional[str] = None) -> pd.DataFrame:
    """Fetch data from Yahoo Finance with rate limit handling"""
    ticker = str(ticker).strip().upper()
    if not ticker or not re.fullmatch(r"[A-Z][A-Z0-9.\-\^=]{0,9}", ticker):
        return pd.DataFrame()
    
    for attempt in range(3):
        try:
            df = yf.download(
                tickers=ticker,
                interval=interval,
                period=period or "1y",
                progress=False,
                auto_adjust=True,
                threads=False,
                group_by="column",
                prepost=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns=str.title)
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep].dropna(how="all")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
            return df
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "too many requests" in msg:
                time.sleep(1.2 * (attempt + 1))
                continue
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_defeat_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """Load daily history from defeatbeta_api package (full history)"""
    try:
        try:
            from defeatbeta_api import Ticker as DefeatTicker
        except Exception:
            from defeatbeta_api.data.ticker import Ticker as DefeatTicker
    except Exception:
        return pd.DataFrame()

    try:
        t = DefeatTicker(ticker)
        df = t.price()
        if df is None or len(df) == 0:
            return pd.DataFrame()
        rename_map = {
            "report_date": "Date", "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        }
        df = df.rename(columns=rename_map)
        needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()
        df = df[needed].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        return df.tail(int(years * 252 * 1.3))
    except Exception:
        return pd.DataFrame()


def fetch_daily_history_optimized(ticker: str, uploaded_history: Dict[str, pd.DataFrame], years: int = 5) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    """
    OPTIMIZED: DefeatBeta (full history) + Yahoo (only last 10 days to fill gap)
    This dramatically reduces Yahoo API calls.
    """
    tkr = ticker.upper()
    meta = {
        "upload_status": "not_matched",
        "defeat_status": "not_checked",
        "yahoo_status": "not_checked",
        "daily_rows": "0",
    }
    
    # PRIORITY 1: Uploaded history (user provides)
    if tkr in uploaded_history:
        base = uploaded_history.get(tkr, pd.DataFrame())
        if base is not None and not base.empty:
            meta["upload_status"] = f"matched:{len(base)}"
            # Only fetch recent days from Yahoo (last 10 days)
            recent = fetch_yahoo_prices(tkr, "1d", period="10d")
            meta["yahoo_status"] = f"recent_fetched:{len(recent)}" if not recent.empty else "recent_empty"
            merged = merge_history_with_recent(base, recent)
            meta["daily_rows"] = str(len(merged))
            return merged, "upload+yahoo_recent", meta
    
    # PRIORITY 2: DefeatBeta (full history, ends ~1 week ago)
    defeat_df = fetch_defeat_history(tkr, years=years)
    if defeat_df is not None and not defeat_df.empty:
        meta["defeat_status"] = f"ok:{len(defeat_df)}"
        # ONLY fetch last 10 days from Yahoo to fill the gap
        recent = fetch_yahoo_prices(tkr, "1d", period="10d")
        meta["yahoo_status"] = f"recent_fetched:{len(recent)}" if not recent.empty else "recent_empty"
        merged = merge_history_with_recent(defeat_df, recent)
        meta["daily_rows"] = str(len(merged))
        source = "defeatbeta+yahoo_recent" if not recent.empty else "defeatbeta_only"
        return merged, source, meta
    
    meta["defeat_status"] = "empty_or_failed"
    
    # PRIORITY 3: Yahoo only (fallback)
    yahoo_df = fetch_yahoo_prices(tkr, "1d", period=f"{max(years, 2)}y")
    if yahoo_df is not None and not yahoo_df.empty:
        meta["yahoo_status"] = f"full_fallback:{len(yahoo_df)}"
        meta["daily_rows"] = str(len(yahoo_df))
        return yahoo_df, "yahoo_fallback", meta
    
    meta["yahoo_status"] = "empty_or_rate_limited"
    return pd.DataFrame(), "none", meta


def fetch_multi_timeframe(ticker: str, uploaded_history: Dict[str, pd.DataFrame], years: int) -> Dict[str, object]:
    """Fetch hourly, daily, and weekly data with optimized daily fetching"""
    daily, history_source, fetch_meta = fetch_daily_history_optimized(ticker, uploaded_history, years)
    hourly = fetch_yahoo_prices(ticker, "1h", period="5d")  # Only last 5 days for hourly
    degraded = False
    
    if daily.empty:
        fetch_meta["hourly_status"] = "skipped_no_daily"
        return {
            "hourly": pd.DataFrame(),
            "daily": pd.DataFrame(),
            "weekly": pd.DataFrame(),
            "history_source": history_source,
            "degraded_hourly": True,
            "fetch_meta": fetch_meta,
        }
    
    if hourly is None or hourly.empty:
        degraded = True
        fetch_meta["hourly_status"] = "rate_limited_using_daily_fallback"
        hourly = resample_ohlcv(daily.tail(120), "B")
    else:
        fetch_meta["hourly_status"] = f"ok:{len(hourly)}"
    
    weekly = resample_ohlcv(daily, "W-FRI")
    fetch_meta["weekly_rows"] = str(len(weekly))
    fetch_meta["daily_source"] = history_source
    
    return {
        "hourly": hourly,
        "daily": daily,
        "weekly": weekly,
        "history_source": history_source,
        "degraded_hourly": degraded,
        "fetch_meta": fetch_meta,
    }


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def centered_pct(series: pd.Series) -> pd.Series:
    return (series.fillna(0.5) - 0.5) * 2


def add_ultimate_oscillator(out: pd.DataFrame, timeframe_name: str) -> pd.DataFrame:
    spans = {"hourly": (5, 13, 5), "daily": (8, 21, 7), "weekly": (5, 13, 5)}

    def pct_col(name: str) -> pd.Series:
        if name in out.columns:
            return centered_pct(out[name])
        return pd.Series(0.0, index=out.index, dtype=float)

    fast, slow, sig = spans[timeframe_name]
    stretch = (
        0.18 * pct_col("rsi_14_pctile")
        + 0.18 * pct_col("cci_20_pctile")
        + 0.14 * pct_col("pct_b_pctile")
        + 0.12 * pct_col("atr_stretch_pctile")
        + 0.10 * pct_col("dist_ema20_pctile")
    )
    momentum = 0.20 * pct_col("tsi_pctile") + 0.08 * np.tanh(out["price_slope_3"].fillna(0) * 25)
    rs_part = 0.10 * np.tanh(out["rs_bench_slope_5"].fillna(0) * 25)
    quality = 1 + 0.15 * pct_col("adx_14_pctile")
    if "dist_vwap_pctile" in out.columns:
        stretch = stretch + 0.10 * centered_pct(out["dist_vwap_pctile"].fillna(0.5))
    out["uo_base"] = (stretch + momentum + rs_part) * quality
    out["uo"] = ema(out["uo_base"], fast) - ema(out["uo_base"], slow)
    out["uo_signal"] = ema(out["uo"], sig)
    out["uo_hist"] = out["uo"] - out["uo_signal"]
    out["uo_gap"] = out["uo_hist"]
    out["uo_slope_1"] = out["uo"].diff(1)
    out["uo_slope_3"] = out["uo"].diff(3)
    out["uo_pctile"] = rolling_percentile(out["uo"], 252 if timeframe_name != "hourly" else 120)
    out["uo_cross_up"] = ((out["uo"].shift(1) <= out["uo_signal"].shift(1)) & (out["uo"] > out["uo_signal"])).astype(float)
    out["uo_cross_down"] = ((out["uo"].shift(1) >= out["uo_signal"].shift(1)) & (out["uo"] < out["uo_signal"])).astype(float)
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
    out["volume_ma_20"] = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_ma_20"].replace(0, np.nan)
    out["close_in_range"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["High"] - out[["Close", "Open"]].max(axis=1)) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["candle_score"] = 50 + out["upper_wick_pct"].fillna(0) * 30 - (out["close_in_range"].fillna(0.5) - 0.5) * 20
    if timeframe_name == "hourly":
        out["vwap"] = compute_vwap(out)
        out["dist_vwap_pct"] = (out["Close"] / out["vwap"]) - 1
    else:
        out["dist_vwap_pct"] = np.nan
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(out.index).ffill()
        out["rs_vs_benchmark"] = out["Close"] / aligned
        out["rs_bench_slope_5"] = slope(out["rs_vs_benchmark"], 5)
    else:
        out["rs_vs_benchmark"] = 1.0
        out["rs_bench_slope_5"] = 0.0
    win = 252 if timeframe_name != "hourly" else 120
    for col in ["rsi_14", "cci_20", "tsi", "pct_b", "atr_stretch", "adx_14", "dist_ema20_pct", "volume_ratio", "dist_vwap_pct"]:
        if col in out.columns:
            out[f"{col}_pctile"] = rolling_percentile(out[col], win)
    out = add_ultimate_oscillator(out, timeframe_name)
    return out


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [1, 2, 3, 5]:
        out[f"fwd_ret_{n}"] = out["Close"].shift(-n) / out["Close"] - 1
    return out


# ============================================================================
# SIGNAL CLASSIFICATION
# ============================================================================

def classify_timeframe_call(row: pd.Series, timeframe: str) -> Tuple[str, float, str]:
    if row is None or row.empty:
        return "NO DATA", 0.0, "No data"
    uo_pct = row.get("uo_pctile", 0.5)
    uo_gap = row.get("uo_gap", 0.0)
    uo_slope = row.get("uo_slope_3", 0.0)
    rsi_val = row.get("rsi_14", 50.0)
    cci_val = row.get("cci_20", 0.0)
    tsi_gap = row.get("tsi_gap", 0.0)
    pct_b = row.get("pct_b", 0.5)
    adx_val = row.get("adx_14", 15.0)
    dist_ema = row.get("dist_ema20_pct", 0.0)

    if uo_pct > 0.80 and (uo_slope < 0 or uo_gap < 0 or tsi_gap < 0) and rsi_val > 70:
        conf = min(95.0, 55 + (uo_pct - 0.80) * 150)
        return "PUT", conf, "Rolling from elevated zone"
    if timeframe == "hourly" and rsi_val > 75 and cci_val > 90 and pct_b > 0.90 and (uo_gap <= 0 or tsi_gap <= 0):
        return "PUT", 78.0, "Hourly overheated and rolling"
    if timeframe == "hourly" and rsi_val > 75 and cci_val > 90 and pct_b > 0.90 and adx_val > 25 and uo_gap > 0:
        return "AVOID CHASE", 72.0, "Pinned continuation risk"
    if uo_pct < 0.20 and (uo_slope > 0 or uo_gap > 0 or tsi_gap > 0) and rsi_val < 35:
        conf = min(95.0, 55 + (0.20 - uo_pct) * 150)
        return "CALL", conf, "Turning up from washed-out zone"
    if uo_gap > 0 and uo_slope > 0 and dist_ema > -0.02:
        return "CALL", 62.0 + max(0.0, min(15.0, (uo_pct - 0.5) * 20)), "Composite rising above signal"
    if uo_gap < 0 and uo_slope < 0 and dist_ema < 0.02:
        return "PUT", 62.0 + max(0.0, min(15.0, (0.5 - uo_pct) * 20)), "Composite below signal and falling"
    if abs(uo_gap) < 0.02:
        return "NEUTRAL", 45.0, "Near signal-line equilibrium"
    return "NEUTRAL", 50.0, "Mixed state"


def detect_flags(hourly_row: pd.Series, daily_row: pd.Series, weekly_row: pd.Series) -> Dict[str, bool]:
    return {
        "hourly_roll": bool(hourly_row.get("uo_gap", 0) < 0 and hourly_row.get("uo_slope_3", 0) < 0 and hourly_row.get("uo_pctile", 0.5) > 0.7),
        "hourly_overheated": bool(hourly_row.get("rsi_14", 50) > 75 and hourly_row.get("cci_20", 0) > 90 and hourly_row.get("pct_b", 0.5) > 0.9),
        "bear_kiss": bool(hourly_row.get("price_slope_3", 0) >= 0 and hourly_row.get("rsi_slope_3", 0) < 0 and hourly_row.get("cci_slope_3", 0) < 0 and hourly_row.get("tsi_slope_3", 0) <= 0),
        "daily_bearish_divergence": bool(daily_row.get("price_slope_3", 0) >= 0 and daily_row.get("rsi_slope_3", 0) < 0 and daily_row.get("cci_slope_3", 0) < 0),
        "post_bottom_thrust": bool(daily_row.get("Close", np.nan) > daily_row.get("ema_20", np.nan) and daily_row.get("rsi_14", 0) > 55 and weekly_row.get("uo_gap", 0) >= 0),
        "dead_cat_risk": bool(weekly_row.get("Close", np.nan) < weekly_row.get("ema_20", np.nan) and daily_row.get("price_slope_3", 0) > 0 and daily_row.get("Close", np.nan) < daily_row.get("ema_20", np.nan)),
    }


# ============================================================================
# ANALOGS & MONTE CARLO
# ============================================================================

def nearest_analogs(df: pd.DataFrame, feature_cols: List[str], top_n: int = 15) -> pd.DataFrame:
    use = [c for c in feature_cols if c in df.columns]
    hist = df.dropna(subset=use + ["fwd_ret_1", "fwd_ret_2", "fwd_ret_5"]).copy()
    if len(hist) < max(40, top_n + 5):
        return pd.DataFrame()
    current = hist.iloc[-1]
    hist = hist.iloc[:-1].copy()
    X = hist[use].astype(float)
    cur = current[use].astype(float)
    std = X.std().replace(0, np.nan)
    z = (X - cur) / std
    hist["distance"] = np.sqrt((z.fillna(0) ** 2).sum(axis=1))
    hist["similarity"] = 1 / (1 + hist["distance"])
    return hist.sort_values("distance").head(top_n)


def summarize_analogs(analogs: pd.DataFrame) -> Dict[str, float]:
    if analogs.empty:
        return {}
    w = analogs["similarity"].fillna(1.0)
    out: Dict[str, float] = {"n": float(len(analogs))}
    for c in ["fwd_ret_1", "fwd_ret_2", "fwd_ret_5"]:
        if c in analogs.columns:
            vals = analogs[c].fillna(0)
            out[f"{c}_median"] = float(vals.median())
            out[f"{c}_mean_w"] = float(np.average(vals, weights=w))
            out[f"{c}_p_down"] = float(np.average((vals < 0).astype(float), weights=w))
            out[f"{c}_p_up"] = float(np.average((vals > 0).astype(float), weights=w))
    return out


def monte_carlo_from_analogs(analogs: pd.DataFrame, horizon_days: int = 5, n_sims: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if analogs.empty:
        return pd.DataFrame(), {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "prob_negative": 0.5}
    return_cols = [c for c in [f"fwd_ret_{i}" for i in range(1, horizon_days + 1)] if c in analogs.columns]
    vals = analogs[return_cols].dropna().values
    if len(vals) == 0:
        return pd.DataFrame(), {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "prob_negative": 0.5}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_sims):
        row = vals[rng.integers(0, len(vals))]
        path = [1.0]
        for r in row:
            path.append(path[-1] * (1 + r))
        boot.append(path)
    sim_paths = pd.DataFrame(boot).T
    terminal = sim_paths.iloc[-1] - 1
    summary = {
        "mean": float(terminal.mean()),
        "median": float(terminal.median()),
        "p10": float(terminal.quantile(0.10)),
        "p90": float(terminal.quantile(0.90)),
        "prob_negative": float((terminal < 0).mean()),
    }
    return sim_paths, summary


def combine_calls(hourly_call: str, daily_call: str, weekly_call: str, flags: Dict[str, bool], daily_analog: Dict[str, float], hourly_analog: Dict[str, float]) -> Tuple[str, str]:
    score_map = {"CALL": 1.0, "PUT": -1.0, "NEUTRAL": 0.0, "AVOID CHASE": -0.35, "NO DATA": 0.0}
    score = 0.30 * score_map.get(hourly_call, 0.0) + 0.45 * score_map.get(daily_call, 0.0) + 0.25 * score_map.get(weekly_call, 0.0)
    score += 0.35 * (daily_analog.get("fwd_ret_2_mean_w", 0.0) * 10)
    score += 0.20 * (hourly_analog.get("fwd_ret_1_mean_w", 0.0) * 10)
    if flags.get("hourly_roll") or flags.get("bear_kiss"):
        score -= 0.55
    if flags.get("hourly_overheated") and daily_call == "CALL":
        score -= 0.45
    if flags.get("post_bottom_thrust") and weekly_call == "CALL":
        score += 0.20
    if flags.get("dead_cat_risk"):
        score -= 0.35

    if hourly_call in {"PUT", "AVOID CHASE"} and daily_call == "CALL" and weekly_call == "CALL":
        return "WAIT / HOURLY TOO HOT", "Bullish trend, but hourly timing is poor"
    if score >= 0.55:
        return "CALL", "Timeframes and analogs are supportive"
    if score <= -0.55:
        return "PUT", "Timeframes and analogs lean bearish"
    if daily_call == "CALL" and weekly_call == "CALL":
        return "CALL ON PULLBACK", "Higher-timeframe trend is constructive, but entry timing needs reset"
    return "NEUTRAL", "Mixed timeframe signals"


@st.cache_data(ttl=900, show_spinner=False)
def get_option_candidates(symbol: str, option_type: str, max_expirations: int = 2) -> pd.DataFrame:
    if not option_type or option_type not in {"CALL", "PUT"}:
        return pd.DataFrame()
    try:
        ticker = yf.Ticker(symbol)
        exps = list(ticker.options)
        if not exps:
            return pd.DataFrame()
        hist = ticker.history(period="5d")
        if hist is None or hist.empty:
            return pd.DataFrame()
        current_price = float(hist["Close"].iloc[-1])
        now = pd.Timestamp.now("UTC").tz_localize(None)
        all_options = []
        for exp_date in exps[:max_expirations]:
            chain = ticker.option_chain(exp_date)
            options = chain.calls.copy() if option_type == "CALL" else chain.puts.copy()
            if options is None or options.empty:
                continue
            exp_dt = pd.Timestamp(exp_date)
            dte = int((exp_dt - now.normalize()).days)
            if option_type == "CALL":
                options = options[options["strike"] >= current_price * 0.95]
            else:
                options = options[options["strike"] <= current_price * 1.05]
            if options.empty:
                continue
            options["spread"] = options["ask"] - options["bid"]
            options["mid"] = (options["ask"] + options["bid"]) / 2
            rel_spread = (options["spread"] / options["mid"].replace(0, np.nan)).clip(upper=1).fillna(1)
            options["days_to_expiry"] = dte
            options["liq_score"] = (
                options["volume"].fillna(0).clip(upper=5000) / 5000 * 0.35
                + options["openInterest"].fillna(0).clip(upper=10000) / 10000 * 0.35
                + (1 - rel_spread) * 0.30
            )
            all_options.append(options.assign(expiration=exp_date, option_type=option_type))
        if not all_options:
            return pd.DataFrame()
        result = pd.concat(all_options, ignore_index=True)
        cols = ["contractSymbol", "expiration", "days_to_expiry", "strike", "option_type", "bid", "ask", "mid", "spread", "volume", "openInterest", "impliedVolatility", "liq_score"]
        return result[cols].sort_values(["liq_score", "volume", "openInterest"], ascending=False).head(12)
    except Exception:
        return pd.DataFrame()


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_timeframe_dashboard(symbol: str, hourly_df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, as_of_label: Optional[str] = None) -> None:
    fig = make_subplots(
        rows=4, cols=1, vertical_spacing=0.06,
        subplot_titles=[f"{symbol} Daily Price" + (f" (as of {as_of_label})" if as_of_label else ""), 
                       "Hourly Ultimate Oscillator", "Daily Ultimate Oscillator", "Weekly Ultimate Oscillator"],
        row_heights=[0.42, 0.19, 0.19, 0.20],
    )
    if not daily_df.empty:
        d = daily_df.tail(220)
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Daily"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["ema_20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["sma_50"], name="SMA50", line=dict(color="blue")), row=1, col=1)
    for row_num, frame in zip([2, 3, 4], [hourly_df.tail(150), daily_df.tail(220), weekly_df.tail(150)]):
        if frame.empty:
            continue
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo"], name=f"UO {row_num}", line=dict(color="red", width=2)), row=row_num, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo_signal"], name=f"Signal {row_num}", line=dict(color="black", width=1)), row=row_num, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row_num, col=1)
    fig.update_layout(height=950, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)


def build_symbol_list(manual_symbols: str, watchlist_file, uploaded_history_map: Dict[str, pd.DataFrame]) -> List[str]:
    symbols: List[str] = []
    if manual_symbols.strip():
        symbols.extend([s.strip().upper() for s in manual_symbols.replace("\n", ",").split(",") if s.strip()])
    if watchlist_file is not None:
        try:
            if watchlist_file.name.lower().endswith(".csv"):
                df = pd.read_csv(watchlist_file)
            else:
                df = pd.read_excel(watchlist_file)
            preferred = [c for c in df.columns if str(c).strip().lower() in {"ticker", "symbol", "tickers", "symbols"}]
            col = preferred[0] if preferred else df.columns[0]
            symbols.extend(df[col].dropna().astype(str).str.upper().str.strip().tolist())
        except Exception:
            pass
    if not symbols and uploaded_history_map:
        symbols.extend(uploaded_history_map.keys())
    symbols = [s for s in symbols if re.fullmatch(r"[A-Z][A-Z0-9.\-\^=]{0,9}", s)]
    return list(dict.fromkeys(symbols))


# ============================================================================
# MAIN APP
# ============================================================================

st.title("📊 Stable Market Engine v4")
st.caption("Optimized: DefeatBeta (full history) + Yahoo (only last 10 days) | Hourly from Yahoo | Calendar backtesting")

with st.sidebar:
    st.header("Input")
    manual_symbols = st.text_area("Paste tickers (comma or line separated)", value="QQQ, SMH, SQQQ", height=100)
    watchlist_file = st.file_uploader("Upload watchlist CSV/XLSX", type=["csv", "xlsx"])
    uploaded_history_files = st.file_uploader("Upload OHLCV files (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=True)

    st.header("Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=0)
    history_years = st.selectbox("Historical years", [3, 5, 10], index=1)
    analog_count = st.slider("Analogs", 5, 30, 15)
    mc_sims = st.slider("Monte Carlo simulations", 500, 5000, 1000, 500)
    show_options = st.checkbox("Show option chains", value=False)
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

st.info("""
**Data Source Strategy:**
- **Daily data**: DefeatBeta (full history) + Yahoo (last 10 days to fill gap)
- **Hourly data**: Yahoo (last 5 days only)
- **Fallback**: If Yahoo rate-limited, uses daily bars resampled to hourly
""")

if not run_analysis:
    st.stop()

with st.spinner("Loading uploaded history..."):
    uploaded_history_map = load_history_uploads(uploaded_history_files) if uploaded_history_files else {}

symbols = build_symbol_list(manual_symbols, watchlist_file, uploaded_history_map)
if not symbols:
    st.error("Provide at least one symbol via paste/watchlist or upload OHLCV files named by ticker.")
    st.stop()

# Fetch benchmark with optimized method
benchmark_daily, _, benchmark_meta = fetch_daily_history_optimized(benchmark, uploaded_history_map, history_years)
if benchmark_daily.empty:
    benchmark_daily = fetch_yahoo_prices(benchmark, "1d", period=f"{max(history_years, 2)}y")

st.caption(f"Benchmark: {benchmark} | Source: {benchmark_meta.get('defeat_status', 'n/a')} / {benchmark_meta.get('yahoo_status', 'n/a')} | Rows: {benchmark_meta.get('daily_rows', '0')}")

results: List[Dict[str, object]] = []
detail_data: Dict[str, Dict[str, object]] = {}
progress = st.progress(0.0)

for idx, symbol in enumerate(symbols):
    progress.progress((idx + 1) / len(symbols))
    data = fetch_multi_timeframe(symbol, uploaded_history_map, history_years)
    daily_df = data["daily"]
    hourly_df = data["hourly"]
    weekly_df = data["weekly"]
    
    if daily_df.empty:
        meta = data.get("fetch_meta", {})
        results.append({
            "Symbol": symbol,
            "Status": "No data",
            "Daily Source": meta.get("defeat_status", "n/a"),
            "Yahoo Recent": meta.get("yahoo_status", "n/a"),
            "Hourly": meta.get("hourly_status", "n/a"),
        })
        continue

    daily_df = add_forward_returns(enrich_price_features(daily_df, "daily", benchmark_daily))
    weekly_df = enrich_price_features(weekly_df, "weekly", benchmark_daily)
    hourly_df = add_forward_returns(enrich_price_features(hourly_df, "hourly", benchmark_daily))

    hourly_row = hourly_df.iloc[-1] if not hourly_df.empty else pd.Series(dtype=float)
    daily_row = daily_df.iloc[-1]
    weekly_row = weekly_df.iloc[-1] if not weekly_df.empty else pd.Series(dtype=float)

    hourly_call, hourly_conf, hourly_reason = classify_timeframe_call(hourly_row, "hourly")
    daily_call, daily_conf, daily_reason = classify_timeframe_call(daily_row, "daily")
    weekly_call, weekly_conf, weekly_reason = classify_timeframe_call(weekly_row, "weekly")
    flags = detect_flags(hourly_row, daily_row, weekly_row)

    daily_analogs = nearest_analogs(daily_df, DAILY_ANALOG_FEATURES, analog_count)
    daily_analog_summary = summarize_analogs(daily_analogs)
    hourly_analogs = nearest_analogs(hourly_df, HOURLY_ANALOG_FEATURES, min(analog_count, 12)) if not data["degraded_hourly"] else pd.DataFrame()
    hourly_analog_summary = summarize_analogs(hourly_analogs)

    combined_call, combined_reason = combine_calls(hourly_call, daily_call, weekly_call, flags, daily_analog_summary, hourly_analog_summary)
    sim_paths, mc_summary = monte_carlo_from_analogs(daily_analogs, 5, mc_sims)

    meta = data.get("fetch_meta", {})
    results.append({
        "Symbol": symbol,
        "Status": "OK",
        "Daily Source": meta.get("defeat_status", "n/a"),
        "Yahoo Recent": meta.get("yahoo_status", "n/a"),
        "Hourly": "Yahoo" if not data["degraded_hourly"] else "Degraded",
        "Hourly Call": hourly_call,
        "Daily Call": daily_call,
        "Weekly Call": weekly_call,
        "Combined": combined_call,
        "2d Med": round(daily_analog_summary.get("fwd_ret_2_median", 0) * 100, 2),
        "MC 5d Mean": round(mc_summary.get("mean", 0) * 100, 2),
    })

    detail_data[symbol] = {
        "hourly": hourly_df,
        "daily": daily_df,
        "weekly": weekly_df,
        "flags": flags,
        "hourly_call": (hourly_call, hourly_conf, hourly_reason),
        "daily_call": (daily_call, daily_conf, daily_reason),
        "weekly_call": (weekly_call, weekly_conf, weekly_reason),
        "combined": (combined_call, combined_reason),
        "daily_analogs": daily_analogs,
        "hourly_analogs": hourly_analogs,
        "daily_analog_summary": daily_analog_summary,
        "hourly_analog_summary": hourly_analog_summary,
        "sim_paths": sim_paths,
        "mc_summary": mc_summary,
        "history_source": data["history_source"],
        "degraded_hourly": data["degraded_hourly"],
        "fetch_meta": data.get("fetch_meta", {}),
    }

progress.empty()
results_df = pd.DataFrame(results)
st.session_state["results_df"] = results_df
st.session_state["detail_data"] = detail_data

st.subheader("Ranked Results")
if not results_df.empty:
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.download_button("Download results CSV", results_df.to_csv(index=False).encode("utf-8"), "stable_market_v4_results.csv", "text/csv")
else:
    st.warning("No results")
    st.stop()

valid_symbols = [r["Symbol"] for r in results if r.get("Status") == "OK"]
if valid_symbols:
    st.subheader("Detailed Analysis")
    selected_symbol = st.selectbox("Select symbol", valid_symbols)
    data = detail_data[selected_symbol]
    
    hc, hconf, hreason = data["hourly_call"]
    dc, dconf, dreason = data["daily_call"]
    wc, wconf, wreason = data["weekly_call"]
    cc, creason = data["combined"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hourly", hc)
    c2.metric("Daily", dc)
    c3.metric("Weekly", wc)
    c4.metric("Combined", cc)
    st.markdown(f"**Hourly reason:** {hreason}")
    st.markdown(f"**Daily reason:** {dreason}")
    st.markdown(f"**Weekly reason:** {wreason}")
    st.markdown(f"**Combined read:** {creason}")
    
    if data["degraded_hourly"]:
        st.warning("⚠️ Hourly data is degraded (using daily bars). Timing signals are lower quality.")
    
    st.caption(f"Data sources: Daily={data['fetch_meta'].get('defeat_status', 'n/a')} | Yahoo recent={data['fetch_meta'].get('yahoo_status', 'n/a')} | Hourly={data['fetch_meta'].get('hourly_status', 'n/a')}")

    flags_df = pd.DataFrame([{"Flag": k, "Detected": v} for k, v in data["flags"].items()])
    st.dataframe(flags_df, use_container_width=True, hide_index=True)

    plot_timeframe_dashboard(selected_symbol, data["hourly"], data["daily"], data["weekly"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Daily analogs")
        st.write({k: round(v, 4) if isinstance(v, float) else v for k, v in data["daily_analog_summary"].items()})
        if not data["daily_analogs"].empty:
            show = data["daily_analogs"].head(10).copy()
            show.index = show.index.strftime("%Y-%m-%d")
            st.dataframe(show[["Close", "similarity", "fwd_ret_1", "fwd_ret_2"]], use_container_width=True)
    with col_b:
        st.markdown("### Hourly analogs")
        st.write({k: round(v, 4) if isinstance(v, float) else v for k, v in data["hourly_analog_summary"].items()})
        if not data["hourly_analogs"].empty:
            show = data["hourly_analogs"].head(10).copy()
            show.index = show.index.astype(str)
            st.dataframe(show[["Close", "similarity", "fwd_ret_1", "fwd_ret_2"]], use_container_width=True)

    st.markdown("### Monte Carlo (daily analog-conditioned)")
    sim_paths = data["sim_paths"]
    if not sim_paths.empty:
        fig = go.Figure()
        x = sim_paths.index
        fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.90, axis=1), name="90th", line=dict(color="lightgreen", width=1)))
        fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.10, axis=1), name="10th", line=dict(color="lightcoral", width=1), fill='tonexty'))
        fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.50, axis=1), name="Median", line=dict(color="blue", width=2)))
        fig.update_layout(height=350, xaxis_title="Days Forward", yaxis_title="Cumulative Return")
        st.plotly_chart(fig, use_container_width=True)
        st.write({k: round(v, 4) if isinstance(v, float) else v for k, v in data["mc_summary"].items()})

    if show_options and cc in {"CALL", "PUT"}:
        st.markdown("### Option Chain")
        opts = get_option_candidates(selected_symbol, cc)
        if opts.empty:
            st.info(f"No {cc} options available.")
        else:
            st.dataframe(opts, use_container_width=True, hide_index=True)
