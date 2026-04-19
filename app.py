"""
Stock Analyzer Ultimate v2.0
Merged from: Diamond Scanner Pro + Predictive Trading Analysis Engine

Features:
- Multi-timeframe (hourly/daily/weekly)
- TSI 25,13,7 as PRIMARY overbought/oversold indicator
- Percentile-based signals for all oscillators
- Option chain integration (CALL/PUT based on signal)
- Calendar backtesting (pick any date, see predictions)
- Monte Carlo simulation from analogs
- Pattern detection (bear kiss, dead cat, pinned, post-bottom thrust)
- Data source priority: Upload → DefeatBeta → Yahoo
- Professional visualizations with candlestick charts
"""

import hashlib
import io
import math
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Optional defeatbeta import
try:
    from defeatbeta_api import Ticker as DefeatTicker
except Exception:
    try:
        from defeatbeta_api.data.ticker import Ticker as DefeatTicker
    except Exception:
        DefeatTicker = None

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Stock Analyzer Ultimate",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sector ETFs for context
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLC": "Communication Services",
    "SMH": "Semiconductors",
}

# Feature columns for analog matching
FEATURE_COLS = [
    "tsi_pctile",           # TSI 25,13,7 percentile (PRIMARY)
    "tsi_slope_3",
    "rsi_14_pctile",
    "rsi_slope_3",
    "cci_20_pctile",
    "cci_slope_3",
    "pct_b_pctile",
    "atr_stretch_pctile",
    "adx_14_pctile",
    "dist_ema20_pctile",
    "rs_bench_slope_5",
    "rs_sector_slope_5",
    "price_slope_3",
    "volume_ratio_pctile",
]

# Initialize session state
for key in ["scan_results", "detail_rows", "analogs_map", "backtest_results", "calendar_predictions"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "scan_results" else {}


# ============================================================================
# INDICATOR FUNCTIONS
# ============================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13, signal_period: int = 7) -> Tuple[pd.Series, pd.Series]:
    """
    True Strength Index - DEFAULT 25,13,7 (PRIMARY overbought/oversold)
    This is the ONLY TSI used in the entire application.
    """
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

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()


def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """Expanding percentile rank (0 to 1)"""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if i + 1 < max(20, window // 5) or not np.isfinite(v):
            continue
        window_vals = values[max(0, i - window + 1):i + 1]
        window_vals = window_vals[np.isfinite(window_vals)]
        if len(window_vals) < max(10, window // 10):
            continue
        out[i] = float((window_vals <= v).mean())
    return pd.Series(out, index=series.index)


def slope(series: pd.Series, bars: int = 3) -> pd.Series:
    return series.diff(bars) / bars


def zscore(series: pd.Series, window: int = 63) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd.replace(0, np.nan)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP calculation"""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    cum_pv = pv.groupby(df.index.date).cumsum()
    cum_v = df["Volume"].groupby(df.index.date).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe"""
    if df.empty:
        return df.copy()
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df.resample(rule).agg(agg).dropna(how="any")


def safe_last(series: pd.Series):
    return None if series.empty else series.iloc[-1]


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_prices(
    ticker: str,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch data from Yahoo Finance"""
    ticker = str(ticker).strip().upper()
    if not ticker or not re.fullmatch(r"[A-Z][A-Z0-9.\-\^=]{0,9}", ticker):
        return pd.DataFrame()
    
    kwargs = {
        "tickers": ticker,
        "interval": interval,
        "progress": False,
        "auto_adjust": True,
        "threads": False,
        "group_by": "column",
        "prepost": False,
    }
    if start is not None or end is not None:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period or "1y"

    try:
        df = yf.download(**kwargs)
    except Exception:
        return pd.DataFrame()
    
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_defeat_history(ticker: str, years: int = 10) -> pd.DataFrame:
    """Load daily history from the defeatbeta_api package"""
    if DefeatTicker is None:
        return pd.DataFrame()
    try:
        t = DefeatTicker(ticker)
        df = t.price()
        if df is None or len(df) == 0:
            return pd.DataFrame()
        rename_map = {
            "report_date": "Date",
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
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
        if years:
            df = df.tail(int(years * 252 * 1.2))
        return df
    except Exception:
        return pd.DataFrame()


def normalize_history_text(text_blob: str, fallback_ticker: str = "") -> Dict[str, pd.DataFrame]:
    """Parse uploaded CSV/Excel files (supports StockCharts format)"""
    out: Dict[str, pd.DataFrame] = {}
    if not text_blob.strip():
        return out
    
    lines = [line.rstrip() for line in text_blob.splitlines() if line.strip()]
    if len(lines) < 3:
        return out
    
    first_parts = [x.strip() for x in lines[0].split(",")]
    second_parts = [x.strip() for x in lines[1].split(",")]
    stockcharts_like = (
        len(first_parts) == 2
        and first_parts[0]
        and first_parts[1].lower().startswith(("daily", "weekly", "monthly"))
        and len(second_parts) >= 5
        and second_parts[0].lower() == "date"
    )
    
    if stockcharts_like:
        ticker = first_parts[0].upper() or fallback_ticker.upper()
        body = "\n".join(lines[1:])
        df = pd.read_csv(io.StringIO(body), skipinitialspace=True)
        df = normalize_history_frame(df)
        if ticker and not df.empty:
            out[ticker] = df
        return out
    
    try:
        df = pd.read_csv(io.StringIO(text_blob), skipinitialspace=True)
    except Exception:
        return out
    
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    ticker_col = cols_lower.get("ticker") or cols_lower.get("symbol")
    if ticker_col is not None:
        for tkr, grp in df.groupby(ticker_col):
            norm = normalize_history_frame(grp.drop(columns=[ticker_col]))
            tkr = str(tkr).strip().upper()
            if tkr and not norm.empty:
                out[tkr] = norm
        return out
    
    norm = normalize_history_frame(df)
    ticker = fallback_ticker.upper()
    if ticker and not norm.empty:
        out[ticker] = norm
    return out


def normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded DataFrame to standard OHLCV format"""
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
        if "Date" not in out.columns and "index" in out.columns:
            out = out.rename(columns={"index": "Date"})
    
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in needed):
        return pd.DataFrame()
    
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).set_index("Date")
    return out[["Open", "High", "Low", "Close", "Volume"]]


def load_history_uploads(files) -> Dict[str, pd.DataFrame]:
    """Load multiple uploaded history files"""
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
        
        if name.endswith('.csv'):
            try:
                text_blob = file.getvalue().decode('utf-8', errors='ignore')
                parsed = normalize_history_text(text_blob, fallback_ticker=ticker)
            except Exception:
                parsed = {}
        else:
            try:
                raw = pd.read_excel(file)
                cols_lower = {str(c).strip().lower(): c for c in raw.columns}
                ticker_col = cols_lower.get('ticker') or cols_lower.get('symbol')
                if ticker_col is not None:
                    for tkr, grp in raw.groupby(ticker_col):
                        norm = normalize_history_frame(grp.drop(columns=[ticker_col]))
                        tkr = str(tkr).strip().upper()
                        if tkr and not norm.empty:
                            parsed[tkr] = norm
                else:
                    norm = normalize_history_frame(raw)
                    if ticker and not norm.empty:
                        parsed[ticker] = norm
            except Exception:
                parsed = {}
        
        for tkr, df in parsed.items():
            if not df.empty:
                if tkr in history_map:
                    history_map[tkr] = pd.concat([history_map[tkr], df]).drop_duplicates().sort_index()
                else:
                    history_map[tkr] = df
    
    return history_map


def merge_history_with_recent(base_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    """Merge historical data with recent Yahoo data"""
    if base_df is None or base_df.empty:
        return recent_df.copy() if recent_df is not None else pd.DataFrame()
    if recent_df is None or recent_df.empty:
        return base_df.copy()
    merged = pd.concat([base_df, recent_df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def fetch_recent_daily_gap_fill(ticker: str, base_df: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
    """Fill gaps with recent Yahoo data"""
    recent = fetch_yahoo_prices(ticker, "1d", period="3mo")
    if recent.empty:
        return base_df.copy()
    if base_df is None or base_df.empty:
        return recent.tail(lookback_days)
    last_base = pd.to_datetime(base_df.index.max())
    cutoff = last_base - pd.Timedelta(days=3)
    recent = recent[recent.index >= cutoff]
    return merge_history_with_recent(base_df, recent)


def fetch_daily_history_with_priority(ticker: str, uploaded_history: Optional[Dict[str, pd.DataFrame]] = None, years: int = 5) -> Tuple[pd.DataFrame, str]:
    """Priority: Upload → DefeatBeta → Yahoo"""
    uploaded_history = uploaded_history or {}
    tkr = ticker.upper()
    
    # Priority 1: Uploaded history
    if tkr in uploaded_history and not uploaded_history[tkr].empty:
        base = uploaded_history[tkr].copy()
        merged = fetch_recent_daily_gap_fill(tkr, base)
        source = "upload+yahoo_gap" if len(merged) > len(base) else "upload"
        return merged, source
    
    # Priority 2: DefeatBeta API
    defeat_df = fetch_defeat_history(tkr, years=max(years, 5))
    if not defeat_df.empty:
        merged = fetch_recent_daily_gap_fill(tkr, defeat_df)
        source = "defeatbeta+yahoo_gap" if len(merged) > len(defeat_df) else "defeatbeta"
        return merged, source
    
    # Priority 3: Yahoo Finance
    yahoo_df = fetch_yahoo_prices(tkr, "1d", period=f"{max(years, 5)}y")
    if not yahoo_df.empty:
        return yahoo_df, "yahoo"
    
    return pd.DataFrame(), "none"


def fetch_multi_timeframe(ticker: str, sector_etf: Optional[str], benchmark: str, uploaded_history: Optional[Dict[str, pd.DataFrame]] = None, years: int = 5) -> Dict[str, pd.DataFrame]:
    """Fetch hourly, daily, and weekly data with proper priority"""
    # Hourly from Yahoo only (real-time)
    hourly = fetch_yahoo_prices(ticker, "1h", period="60d")
    if hourly.empty:
        hourly = fetch_yahoo_prices(ticker, "60m", period="60d")
    
    # Daily with priority chain
    daily, history_source = fetch_daily_history_with_priority(ticker, uploaded_history=uploaded_history, years=years)
    if daily.empty:
        return {"hourly": pd.DataFrame(), "daily": pd.DataFrame(), "weekly": pd.DataFrame(), "history_source": history_source}
    
    # Fallback for hourly if needed
    if hourly.empty:
        hourly = resample_ohlcv(daily.tail(90), "B")
        history_source = f"{history_source}|hourly_fallback_from_daily"
    
    weekly = resample_ohlcv(daily, "W-FRI")
    
    return {
        "hourly": hourly,
        "daily": daily,
        "weekly": weekly,
        "history_source": history_source
    }


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def enrich_price_features(df: pd.DataFrame, timeframe_name: str, benchmark_df: pd.DataFrame = None, sector_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add all technical features - SINGLE TSI 25,13,7 as PRIMARY"""
    if df.empty:
        return df.copy()
    
    out = df.copy()
    
    # Returns
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_5"] = out["Close"].pct_change(5)
    
    # Moving averages
    out["ema_10"] = ema(out["Close"], 10)
    out["ema_20"] = ema(out["Close"], 20)
    out["sma_20"] = sma(out["Close"], 20)
    out["sma_50"] = sma(out["Close"], 50)
    
    # ATR
    out["atr_14"] = atr(out, 14)
    out["atr_stretch"] = (out["Close"] - out["ema_20"]) / out["atr_14"].replace(0, np.nan)
    
    # RSI
    out["rsi_14"] = rsi(out["Close"], 14)
    out["rsi_5"] = rsi(out["Close"], 5)
    out["rsi_slope_3"] = slope(out["rsi_14"], 3)
    
    # CCI
    out["cci_20"] = cci(out, 20)
    out["cci_15"] = cci(out, 15)
    out["cci_slope_3"] = slope(out["cci_20"], 3)
    
    # TSI 25,13,7 (PRIMARY - only one)
    out["tsi"], out["tsi_signal"] = tsi(out["Close"], 25, 13, 7)
    out["tsi_slope_3"] = slope(out["tsi"], 3)
    
    # MACD
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["Close"], 12, 26, 9)
    
    # Bollinger Bands %B
    out["pct_b"] = bollinger_pct_b(out["Close"], 20, 2)
    
    # ADX
    out["adx_14"] = adx(out, 14)
    
    # Distance from MAs
    out["dist_ema10_pct"] = (out["Close"] / out["ema_10"]) - 1
    out["dist_ema20_pct"] = (out["Close"] / out["ema_20"]) - 1
    out["dist_sma50_pct"] = (out["Close"] / out["sma_50"]) - 1
    
    # Volume
    out["volume_ma_20"] = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_ma_20"].replace(0, np.nan)
    out["rvol_20"] = out["volume_ratio"]  # alias
    
    # Z-scores
    out["z_20"] = zscore(out["Close"], 20)
    out["z_63"] = zscore(out["Close"], 63)
    
    # Candle features
    out["close_in_range"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["High"] - out[["Close", "Open"]].max(axis=1)) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["body_pct"] = (out["Close"] - out["Open"]).abs() / (out["High"] - out["Low"]).replace(0, np.nan)
    out["candle_score"] = 50 + out["upper_wick_pct"].fillna(0) * 25 - out["body_pct"].fillna(0) * 15
    
    # Price slope
    out["price_slope_3"] = slope(out["Close"], 3)
    out["price_slope_5"] = slope(out["Close"], 5)
    out["higher_high"] = (out["High"] > out["High"].shift(1)).astype(float)
    out["lower_high"] = (out["High"] < out["High"].shift(1)).astype(float)
    
    # Hourly-specific features
    if timeframe_name == "hourly":
        out["vwap"] = compute_vwap(out)
        out["dist_vwap_pct"] = (out["Close"] / out["vwap"]) - 1
        out["hours_from_10bar_low"] = out["Low"].rolling(10).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True)
        out["intraday_atr_pct"] = out["atr_14"] / out["Close"]
    else:
        out["dist_vwap_pct"] = np.nan
        out["hours_from_10bar_low"] = np.nan
        out["intraday_atr_pct"] = np.nan
    
    # Percentile ranks (for cross-stock comparison)
    window = 252 if timeframe_name != "hourly" else 120
    for col in ["rsi_14", "cci_20", "tsi", "pct_b", "atr_stretch", "adx_14", "volume_ratio"]:
        if col in out.columns:
            out[f"{col}_pctile"] = rolling_percentile(out[col], window)
    
    # Relative strength vs benchmark
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(out.index).ffill()
        out["rs_vs_benchmark"] = out["Close"] / aligned
        out["rs_bench_slope_5"] = slope(out["rs_vs_benchmark"], 5)
    else:
        out["rs_vs_benchmark"] = 1.0
        out["rs_bench_slope_5"] = 0.0
    
    # Relative strength vs sector
    if sector_df is not None and not sector_df.empty:
        aligned_sec = sector_df["Close"].reindex(out.index).ffill()
        out["rs_vs_sector"] = out["Close"] / aligned_sec
        out["rs_sector_slope_5"] = slope(out["rs_vs_sector"], 5)
    else:
        out["rs_vs_sector"] = 1.0
        out["rs_sector_slope_5"] = 0.0
    
    return out


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward returns for prediction"""
    if df.empty:
        return df.copy()
    out = df.copy()
    for n in [1, 2, 3, 5]:
        out[f"fwd_ret_{n}"] = out["Close"].shift(-n) / out["Close"] - 1
    out["fwd_low_1"] = out["Low"].shift(-1) / out["Close"] - 1
    out["fwd_high_1"] = out["High"].shift(-1) / out["Close"] - 1
    return out


# ============================================================================
# SIGNAL DETECTION (TSI 25,13,7 CENTRIC)
# ============================================================================

def classify_timeframe_state(row: pd.Series, timeframe: str) -> str:
    """Classify a single timeframe as Bullish/Neutral/Bearish/Stalling"""
    if row.empty:
        return "Unknown"
    
    tsi_pctile = row.get("tsi_pctile", 0.5)
    price_slope = row.get("price_slope_3", 0)
    tsi_slope = row.get("tsi_slope_3", 0)
    price_vs_ema = row.get("dist_ema20_pct", 0)
    rsi_pctile = row.get("rsi_14_pctile", 0.5)
    
    # PRIMARY SIGNAL: TSI 25,13,7 percentile based
    # Overheated (bearish): TSI > 70th percentile
    if tsi_pctile > 0.7 and price_slope > 0 and tsi_slope < 0:
        return "Overheated (Bearish)"
    
    # Washed out (bullish): TSI < 30th percentile
    if tsi_pctile < 0.3 and price_slope < 0 and tsi_slope > 0:
        return "Washed Out (Bullish)"
    
    # Bullish conditions
    if tsi_pctile > 0.5 and price_slope > 0 and price_vs_ema > -0.02:
        return "Bullish"
    
    # Bearish conditions
    if tsi_pctile < 0.5 and price_slope < 0 and price_vs_ema < 0.02:
        return "Bearish"
    
    # Stalling (price up but momentum down)
    if price_slope > 0 and tsi_slope < 0:
        return "Stalling"
    
    # Bear kiss (price making higher highs but momentum rolling)
    if price_slope > 0 and rsi_pctile > 0.7 and tsi_slope < 0:
        return "Bear Kiss"
    
    return "Neutral"


def detect_pattern_flags(hourly_row: pd.Series, daily_row: pd.Series, weekly_row: pd.Series) -> Dict[str, bool]:
    """Detect pattern flags across timeframes"""
    flags = {}
    
    # Bear kiss hourly
    flags["bear_kiss_hourly"] = bool(
        hourly_row.get("price_slope_3", 0) >= 0 and
        hourly_row.get("rsi_slope_3", 0) < 0 and
        hourly_row.get("cci_slope_3", 0) < 0 and
        hourly_row.get("tsi_slope_3", 0) <= 0
    )
    
    # Daily bearish divergence
    flags["daily_bearish_divergence"] = bool(
        daily_row.get("price_slope_3", 0) >= 0 and
        daily_row.get("rsi_slope_3", 0) < 0 and
        daily_row.get("cci_slope_3", 0) < 0
    )
    
    # Pinned continuation risk (overextended but still trending)
    flags["pinned_continuation_risk"] = bool(
        hourly_row.get("rsi_14_pctile", 0) > 0.85 and
        hourly_row.get("pct_b_pctile", 0) > 0.85 and
        hourly_row.get("adx_14", 0) > 22 and
        hourly_row.get("close_in_range", 0) > 0.65 and
        hourly_row.get("price_slope_3", 0) > 0
    )
    
    # Dead cat bounce risk
    flags["dead_cat_bounce_risk"] = bool(
        weekly_row.get("Close", np.nan) < weekly_row.get("ema_20", np.nan) and
        daily_row.get("price_slope_3", 0) > 0 and
        daily_row.get("rsi_14", 50) < 55 and
        daily_row.get("dist_ema20_pct", 0) < 0
    )
    
    # Post-bottom thrust (recovery signal)
    flags["post_bottom_thrust"] = bool(
        daily_row.get("Close", np.nan) > daily_row.get("ema_20", np.nan) and
        daily_row.get("dist_sma50_pct", -1) > -0.02 and
        daily_row.get("rsi_14", 0) > 55 and
        weekly_row.get("rsi_slope_3", 0) >= 0
    )
    
    # Overheated (TSI > 70th percentile) - PRIMARY
    flags["overheated"] = bool(daily_row.get("tsi_pctile", 0) > 0.7)
    
    # Washed out (TSI < 30th percentile) - PRIMARY
    flags["washed_out"] = bool(daily_row.get("tsi_pctile", 0) < 0.3)
    
    # TSI slope positive (momentum increasing)
    flags["tsi_momentum_up"] = bool(daily_row.get("tsi_slope_3", 0) > 0)
    
    # TSI slope negative (momentum decreasing)
    flags["tsi_momentum_down"] = bool(daily_row.get("tsi_slope_3", 0) < 0)
    
    return flags


def get_alignment_label(hourly_state: str, daily_state: str, weekly_state: str) -> str:
    """Get alignment description across timeframes"""
    if hourly_state in ["Overheated (Bearish)", "Bearish"] and daily_state in ["Overheated (Bearish)", "Bearish"]:
        return "Bearish alignment - consider PUTs"
    if hourly_state in ["Washed Out (Bullish)", "Bullish"] and daily_state in ["Washed Out (Bullish)", "Bullish"]:
        return "Bullish alignment - consider CALLs"
    if hourly_state in ["Overheated (Bearish)", "Bearish"] and daily_state == "Bullish":
        return "Short-term fade in larger uptrend - cautious PUTs"
    if hourly_state in ["Washed Out (Bullish)", "Bullish"] and daily_state == "Bearish":
        return "Countertrend bounce - cautious CALLs"
    if "Overheated" in hourly_state:
        return "Hourly overheated - short-term fade opportunity"
    if "Washed Out" in hourly_state:
        return "Hourly washed out - short-term bounce opportunity"
    return "Mixed alignment - wait for clearer signal"


def get_direction_from_tsi(tsi_pctile: float, daily_state: str) -> Tuple[str, float]:
    """Determine option direction based on TSI 25,13,7 percentile"""
    if tsi_pctile > 0.7 or daily_state in ["Overheated (Bearish)", "Bearish"]:
        return "PUT", min(1.0, (tsi_pctile - 0.7) / 0.3 + 0.5)
    elif tsi_pctile < 0.3 or daily_state in ["Washed Out (Bullish)", "Bullish"]:
        return "CALL", min(1.0, (0.3 - tsi_pctile) / 0.3 + 0.5)
    else:
        return None, 0.3


def get_confidence_from_analogs(analog_summary: Dict[str, float], flags: Dict[str, bool], tsi_pctile: float) -> float:
    """Calculate confidence score (0-100)"""
    confidence = 50.0
    
    # Analog-based confidence
    p_down = analog_summary.get("fwd_ret_1_p_down", 0.5)
    if tsi_pctile > 0.7:  # Expecting down
        confidence += (p_down - 0.5) * 50
    elif tsi_pctile < 0.3:  # Expecting up
        confidence += (0.5 - p_down) * 50
    
    # Pattern-based adjustments
    if flags.get("bear_kiss_hourly") and tsi_pctile > 0.7:
        confidence += 10
    if flags.get("daily_bearish_divergence") and tsi_pctile > 0.7:
        confidence += 15
    if flags.get("post_bottom_thrust") and tsi_pctile < 0.3:
        confidence += 10
    if flags.get("dead_cat_bounce_risk") and tsi_pctile > 0.7:
        confidence += 10
    
    # Clip and return
    return max(0, min(100, confidence))


# ============================================================================
# ANALOG MATCHING & MONTE CARLO
# ============================================================================

def nearest_analogs(df: pd.DataFrame, feature_cols: List[str], top_n: int = 15) -> pd.DataFrame:
    """Find nearest historical analogs using weighted features"""
    hist = df.dropna(subset=[c for c in feature_cols if c in df.columns]).copy()
    if len(hist) < max(30, top_n + 5):
        return pd.DataFrame()

    current = hist.iloc[-1]
    hist = hist.iloc[:-1].copy()
    
    usable = [c for c in feature_cols if c in hist.columns and c in current.index]
    if not usable:
        return pd.DataFrame()

    X = hist[usable].copy()
    cur = current[usable].copy()

    # Standardize
    std = X.std().replace(0, np.nan)
    z = (X - cur) / std
    hist["distance"] = np.sqrt((z.fillna(0) ** 2).sum(axis=1))
    
    analogs = hist.sort_values("distance").head(top_n).copy()
    analogs["similarity"] = 1 / (1 + analogs["distance"])
    
    cols = ["Close", "distance", "similarity"] + \
           [c for c in ["fwd_ret_1", "fwd_ret_2", "fwd_ret_3", "fwd_ret_5", "fwd_low_1", "fwd_high_1"] 
            if c in analogs.columns]
    return analogs[cols]


def summarize_analogs(analogs: pd.DataFrame) -> Dict[str, float]:
    """Summarize analog statistics"""
    if analogs.empty:
        return {}
    
    out = {}
    for c in ["fwd_ret_1", "fwd_ret_2", "fwd_ret_3", "fwd_ret_5", "fwd_low_1", "fwd_high_1"]:
        if c in analogs.columns:
            out[f"{c}_mean"] = analogs[c].mean()
            out[f"{c}_median"] = analogs[c].median()
            out[f"{c}_std"] = analogs[c].std()
            out[f"{c}_p_down"] = float((analogs[c] < 0).mean()) if "ret" in c else np.nan
            out[f"{c}_p_up"] = float((analogs[c] > 0).mean()) if "ret" in c else np.nan
    
    return out


def monte_carlo_from_analogs(analogs: pd.DataFrame, horizon_days: int = 5, n_sims: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """Monte Carlo simulation conditioned on analogs"""
    if analogs.empty:
        return pd.DataFrame(), {"mean": 0, "median": 0, "p10": 0, "p90": 0, "prob_negative": 0.5}
    
    return_cols = [c for c in [f"fwd_ret_{i}" for i in range(1, horizon_days + 1)] if c in analogs.columns]
    if not return_cols:
        return pd.DataFrame(), {"mean": 0, "median": 0, "p10": 0, "p90": 0, "prob_negative": 0.5}
    
    rng = np.random.default_rng(seed)
    vals = analogs[return_cols].dropna().values
    
    if len(vals) == 0:
        return pd.DataFrame(), {"mean": 0, "median": 0, "p10": 0, "p90": 0, "prob_negative": 0.5}
    
    boot = []
    for _ in range(n_sims):
        row = vals[rng.integers(0, len(vals))]
        path = [1.0]
        for r in row:
            path.append(path[-1] * (1 + r))
        boot.append(path)
    
    sim_paths = pd.DataFrame(boot).T
    sim_paths.index.name = "step"
    
    terminal = sim_paths.iloc[-1] - 1
    summary = {
        "mean": terminal.mean(),
        "median": terminal.median(),
        "p10": terminal.quantile(0.10),
        "p90": terminal.quantile(0.90),
        "prob_negative": float((terminal < 0).mean()),
    }
    
    return sim_paths, summary


# ============================================================================
# OPTION CHAIN
# ============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_option_candidates(symbol: str, option_type: str, max_expirations: int = 3) -> Optional[pd.DataFrame]:
    """Fetch option chain from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        exps = list(ticker.options)
        if not exps:
            return None
        
        hist = ticker.history(period="5d")
        if hist is None or hist.empty:
            return None
        
        current_price = float(hist["Close"].iloc[-1])
        all_options = []
        now = pd.Timestamp.now('UTC').tz_localize(None)
        
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
            options["expiration"] = exp_date
            options["days_to_expiry"] = dte
            options["option_type"] = option_type
            
            rel_spread = (options["spread"] / options["mid"].replace(0, np.nan)).clip(upper=1).fillna(1)
            options["liq_score"] = (
                options["volume"].fillna(0).clip(upper=5000) / 5000 * 0.35 +
                options["openInterest"].fillna(0).clip(upper=10000) / 10000 * 0.35 +
                (1 - rel_spread) * 0.30
            )
            all_options.append(options)
        
        if not all_options:
            return None
        
        result = pd.concat(all_options, ignore_index=True)
        result = result.sort_values(["liq_score", "volume", "openInterest"], ascending=False)
        
        cols = ["contractSymbol", "expiration", "days_to_expiry", "strike", "option_type", 
                "bid", "ask", "mid", "spread", "volume", "openInterest", "impliedVolatility", "liq_score"]
        cols = [c for c in cols if c in result.columns]
        return result[cols].head(12)
    except Exception:
        return None


# ============================================================================
# CALENDAR BACKTESTING
# ============================================================================

def get_prediction_for_date(daily_df: pd.DataFrame, target_date: pd.Timestamp, analog_count: int = 15) -> Dict:
    """Get what the model would have predicted on a specific date"""
    df_before = daily_df[daily_df.index <= target_date].copy()
    if df_before.empty or len(df_before) < 100:
        return {"error": "Insufficient data before this date"}
    
    # Use the last row before target date as current
    current_row = df_before.iloc[-1].copy()
    
    # Find analogs from history BEFORE current date
    hist_for_analogs = df_before.iloc[:-1].copy()
    analogs = nearest_analogs(hist_for_analogs, FEATURE_COLS, top_n=analog_count)
    
    if analogs.empty:
        return {"error": "No analogs found"}
    
    analog_summary = summarize_analogs(analogs)
    
    # Get actual forward returns (what actually happened after the prediction date)
    future_df = daily_df[daily_df.index > target_date].copy()
    if not future_df.empty:
        actual_ret_1 = future_df.iloc[0]["Close"] / current_row["Close"] - 1 if len(future_df) >= 1 else np.nan
        actual_ret_2 = future_df.iloc[1]["Close"] / current_row["Close"] - 1 if len(future_df) >= 2 else np.nan
        actual_ret_5 = future_df.iloc[4]["Close"] / current_row["Close"] - 1 if len(future_df) >= 5 else np.nan
    else:
        actual_ret_1 = actual_ret_2 = actual_ret_5 = np.nan
    
    # Get TSI-based signal
    tsi_pctile = current_row.get("tsi_pctile", 0.5)
    daily_state = classify_timeframe_state(current_row, "daily")
    direction, strength = get_direction_from_tsi(tsi_pctile, daily_state)
    
    return {
        "date": target_date,
        "close": current_row["Close"],
        "tsi_pctile": tsi_pctile,
        "tsi_raw": current_row.get("tsi", np.nan),
        "daily_state": daily_state,
        "predicted_direction": direction,
        "predicted_strength": strength,
        "analog_pred_1d": analog_summary.get("fwd_ret_1_median", np.nan),
        "analog_pred_2d": analog_summary.get("fwd_ret_2_median", np.nan),
        "analog_pred_5d": analog_summary.get("fwd_ret_5_median", np.nan),
        "actual_ret_1d": actual_ret_1,
        "actual_ret_2d": actual_ret_2,
        "actual_ret_5d": actual_ret_5,
        "num_analogs": len(analogs),
    }


def run_calendar_backtest(daily_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, analog_count: int = 15) -> pd.DataFrame:
    """Run backtest on a date range"""
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    results = []
    
    for date in date_range:
        if date < daily_df.index.min() + pd.Timedelta(days=100):
            continue
        pred = get_prediction_for_date(daily_df, date, analog_count)
        if "error" not in pred:
            results.append(pred)
    
    return pd.DataFrame(results)


# ============================================================================
# SCORING
# ============================================================================

def score_setup(states: Dict[str, str], flags: Dict[str, bool], analog_summary: Dict[str, float]) -> Dict[str, float]:
    """Calculate long and short scores (0-100) based on TSI 25,13,7 primarily"""
    long_score = 50.0
    short_score = 50.0
    
    # Timeframe alignment (TSI-centric)
    if states["hourly"] in ["Washed Out (Bullish)", "Bullish"]:
        long_score += 5
        short_score -= 5
    if states["daily"] in ["Washed Out (Bullish)", "Bullish"]:
        long_score += 10
        short_score -= 10
    if states["weekly"] in ["Washed Out (Bullish)", "Bullish"]:
        long_score += 8
        short_score -= 8
    
    if states["hourly"] in ["Overheated (Bearish)", "Bearish"]:
        long_score -= 5
        short_score += 5
    if states["daily"] in ["Overheated (Bearish)", "Bearish"]:
        long_score -= 10
        short_score += 10
    
    # Pattern flags
    if flags.get("bear_kiss_hourly"):
        long_score -= 10
        short_score += 10
    if flags.get("daily_bearish_divergence"):
        long_score -= 12
        short_score += 12
    if flags.get("pinned_continuation_risk"):
        long_score -= 5
        short_score += 5
    if flags.get("dead_cat_bounce_risk"):
        long_score -= 15
        short_score += 15
    if flags.get("post_bottom_thrust"):
        long_score += 10
        short_score -= 10
    if flags.get("overheated"):
        long_score -= 15
        short_score += 15
    if flags.get("washed_out"):
        long_score += 15
        short_score -= 15
    
    # Analog forward returns
    p_down = analog_summary.get("fwd_ret_1_p_down", 0.5)
    long_score += (0.5 - p_down) * 30
    short_score -= (0.5 - p_down) * 30
    
    mean_ret = analog_summary.get("fwd_ret_2_mean", 0)
    long_score += max(-15, min(15, mean_ret * 1000))
    short_score -= max(-15, min(15, mean_ret * 1000))
    
    long_score = float(np.clip(long_score, 0, 100))
    short_score = float(np.clip(short_score, 0, 100))
    
    return {"long_score": round(long_score, 1), "short_score": round(short_score, 1)}


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_price_and_oscillator(ticker: str, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, hourly_df: pd.DataFrame):
    """Plot price and oscillators across timeframes with TSI 25,13,7 emphasis"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"{ticker} Daily Price", "Daily Oscillators (TSI 25,13,7 highlighted)", "Hourly Oscillators"],
        row_heights=[0.45, 0.27, 0.28],
    )
    
    if not daily_df.empty:
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=daily_df.index,
                open=daily_df["Open"],
                high=daily_df["High"],
                low=daily_df["Low"],
                close=daily_df["Close"],
                name="Daily",
            ),
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["ema_20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["sma_50"], name="SMA50", line=dict(color="blue")), row=1, col=1)
        
        # Percentile-based oscillators with TSI highlighted
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["rsi_14_pctile"] * 100, name="RSI14 %ile", line=dict(color="gray", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["cci_20_pctile"] * 100, name="CCI20 %ile", line=dict(color="lightblue", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["tsi_pctile"] * 100, name="TSI 25,13,7 %ile (PRIMARY)", line=dict(color="red", width=3)), row=2, col=1)
        
        # Add 70/30 threshold lines (overheated/washed out)
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overheated (70th)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Washed Out (30th)", row=2, col=1)
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, row=2, col=1)
    
    if not hourly_df.empty:
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["rsi_14"], name="Hourly RSI14", line=dict(color="gray")), row=3, col=1)
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["cci_20"], name="Hourly CCI20", line=dict(color="lightblue")), row=3, col=1)
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["tsi"], name="Hourly TSI 25,13,7", line=dict(color="red", width=2)), row=3, col=1)
    
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend_orientation="h")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Percentile (%)", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    
    st.plotly_chart(fig, width='stretch')


def plot_monte_carlo(sim_paths: pd.DataFrame, mc_summary: Dict, ticker: str):
    """Plot Monte Carlo simulation results"""
    if sim_paths.empty:
        st.info("Not enough analog data for Monte Carlo.")
        return
    
    fig = go.Figure()
    x = sim_paths.index
    
    # Add confidence bands
    fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.9, axis=1), name="90th %ile", line=dict(color="lightgreen", width=1), fill=None))
    fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.1, axis=1), name="10th %ile", line=dict(color="lightcoral", width=1), fill='tonexty'))
    fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.5, axis=1), name="Median Path", line=dict(color="blue", width=2)))
    
    fig.update_layout(
        height=400,
        title=f"{ticker} Conditional Monte Carlo (analog-conditioned, {sim_paths.shape[1]} simulations)",
        xaxis_title="Days Forward",
        yaxis_title="Cumulative Return",
    )
    st.plotly_chart(fig, width='stretch')
    
    # Display stats
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean", f"{mc_summary.get('mean', 0)*100:.2f}%")
    col2.metric("Median", f"{mc_summary.get('median', 0)*100:.2f}%")
    col3.metric("10th %ile", f"{mc_summary.get('p10', 0)*100:.2f}%")
    col4.metric("90th %ile", f"{mc_summary.get('p90', 0)*100:.2f}%")
    col5.metric("Downside Prob", f"{mc_summary.get('prob_negative', 0)*100:.1f}%")


def plot_analog_distribution(analogs: pd.DataFrame):
    """Plot analog forward return distribution"""
    if analogs.empty:
        return
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["1-Day Forward Returns", "5-Day Forward Returns"])
    
    if "fwd_ret_1" in analogs.columns:
        fig.add_trace(go.Histogram(x=analogs["fwd_ret_1"] * 100, nbinsx=20, name="1-day", marker_color="blue"), row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    
    if "fwd_ret_5" in analogs.columns:
        fig.add_trace(go.Histogram(x=analogs["fwd_ret_5"] * 100, nbinsx=20, name="5-day", marker_color="orange"), row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(height=400, title="Analog Forward Return Distributions")
    fig.update_xaxes(title_text="Return (%)")
    fig.update_yaxes(title_text="Frequency")
    
    st.plotly_chart(fig, width='stretch')


def describe_setup(states: Dict[str, str], flags: Dict[str, bool], analog_summary: Dict[str, float], tsi_pctile: float) -> str:
    """Generate narrative description of the setup"""
    bits = []
    
    # TSI 25,13,7 primary signal
    if tsi_pctile > 0.7:
        bits.append(f"TSI 25,13,7 is at {tsi_pctile*100:.1f}th percentile (OVERHEATED). This suggests a bearish/short setup.")
    elif tsi_pctile < 0.3:
        bits.append(f"TSI 25,13,7 is at {tsi_pctile*100:.1f}th percentile (WASHED OUT). This suggests a bullish/long setup.")
    else:
        bits.append(f"TSI 25,13,7 is at {tsi_pctile*100:.1f}th percentile (neutral range).")
    
    # Alignment
    bits.append(f"Alignment: {states['alignment']}.")
    
    # Pattern flags
    if flags.get("bear_kiss_hourly"):
        bits.append("⚠️ Hourly momentum shows bear-kiss / rollover pattern.")
    if flags.get("daily_bearish_divergence"):
        bits.append("⚠️ Daily momentum is diverging bearishly against price.")
    if flags.get("pinned_continuation_risk"):
        bits.append("📌 Pinned continuation risk - trend may continue.")
    if flags.get("dead_cat_bounce_risk"):
        bits.append("🐱 This bounce has dead-cat / countertrend characteristics.")
    if flags.get("post_bottom_thrust"):
        bits.append("🚀 Post-bottom thrust detected - recovery signal.")
    
    # Analog stats
    if analog_summary:
        p_down = analog_summary.get("fwd_ret_1_p_down", np.nan)
        med2 = analog_summary.get("fwd_ret_2_median", np.nan)
        if not pd.isna(p_down):
            bits.append(f"Analogs show {p_down:.0%} probability of negative next day.")
        if not pd.isna(med2):
            bits.append(f"Median 2-day analog return: {med2:.2%}.")
    
    return " ".join(bits)


# ============================================================================
# MAIN APP
# ============================================================================

st.title("📊 Stock Analyzer Ultimate v2.0")
st.caption("Multi-timeframe | TSI 25,13,7 PRIMARY | Percentile-based | Option Chains | Calendar Backtesting")

# Sidebar
with st.sidebar:
    st.header("📁 Input")
    
    input_method = st.radio("Symbol input method", ["Paste tickers", "Upload watchlist"], index=0)
    
    symbols = []
    if input_method == "Paste tickers":
        ticker_text = st.text_area("Stock symbols (comma or line separated)", 
                                   value="AAPL, MSFT, NVDA, AMD, INTC", height=100)
        symbols = [s.strip().upper() for s in ticker_text.replace("\n", ",").split(",") if s.strip()]
    else:
        uploaded_watchlist = st.file_uploader("Upload watchlist CSV/XLSX", type=["csv", "xlsx"])
        if uploaded_watchlist:
            name = uploaded_watchlist.name.lower()
            try:
                if name.endswith(".csv"):
                    df = pd.read_csv(uploaded_watchlist)
                else:
                    df = pd.read_excel(uploaded_watchlist)
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                preferred_cols = [c for c in df.columns if str(c).strip().lower() in {"ticker", "symbol", "tickers", "symbols"}]
                col = preferred_cols[0] if preferred_cols else df.columns[0]
                symbols = df[col].dropna().astype(str).str.upper().str.strip().tolist()
                symbols = [s for s in symbols if re.fullmatch(r"[A-Z][A-Z0-9.\-\^=]{0,9}", s)]
                st.success(f"Loaded {len(symbols)} symbols")
    
    st.header("📂 Historical Data (Optional)")
    uploaded_history_files = st.file_uploader(
        "Upload OHLCV files (CSV/XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help="Priority: Uploaded files → DefeatBeta → Yahoo"
    )
    
    st.header("⚙️ Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=0)
    sector_etf = st.selectbox("Sector context", ["None"] + list(SECTOR_ETFS.keys()), index=0)
    history_years = st.selectbox("Historical years", [3, 5, 10], index=1)
    
    st.header("🔬 Analysis Options")
    top_analogs = st.slider("Number of analogs", 5, 30, 15)
    mc_sims = st.slider("Monte Carlo simulations", 500, 10000, 1000, step=500)
    
    st.header("🕐 Mode")
    analysis_mode = st.radio("Analysis mode", ["Current", "Calendar Backtest"], index=0)
    
    if analysis_mode == "Calendar Backtest":
        col1, col2 = st.columns(2)
        with col1:
            backtest_start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=180))
        with col2:
            backtest_end = st.date_input("End date", value=pd.Timestamp.today().date())
    
    st.header("📦 Options")
    show_options = st.checkbox("Show option chains (current mode only)", value=True)
    rebuild_cache = st.checkbox("Rebuild cache", value=False)
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", width='stretch')

# Main content
if not run_analysis:
    st.info("👈 Configure settings and click **Run Analysis**")
    st.stop()

if not symbols:
    st.error("Please enter at least one symbol")
    st.stop()

# Clear cache if requested
if rebuild_cache:
    for cache_file in CACHE_DIR.glob("*.parquet"):
        cache_file.unlink()
    st.success("Cache cleared")

# Load uploaded history
uploaded_history_map = load_history_uploads(uploaded_history_files) if uploaded_history_files else {}

# Run analysis for all symbols
progress_bar = st.progress(0)
status_text = st.empty()

results_rows = []
detail_data = {}

for i, symbol in enumerate(symbols):
    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
    progress_bar.progress((i + 1) / len(symbols))
    
    # Fetch data with priority chain
    data = fetch_multi_timeframe(
        symbol, 
        sector_etf if sector_etf != "None" else None, 
        benchmark, 
        uploaded_history=uploaded_history_map, 
        years=history_years
    )
    
    hourly_df = data["hourly"]
    daily_df = data["daily"]
    weekly_df = data["weekly"]
    history_source = data.get("history_source", "unknown")
    
    if daily_df.empty:
        results_rows.append({
            "Symbol": symbol, 
            "Status": "No data", 
            "History Source": history_source,
            "Long Score": 0, 
            "Short Score": 0
        })
        continue
    
    # Fetch benchmark and sector data for features
    benchmark_df, _ = fetch_daily_history_with_priority(benchmark, uploaded_history=uploaded_history_map, years=history_years)
    sector_df = None
    if sector_etf != "None":
        sector_df, _ = fetch_daily_history_with_priority(sector_etf, uploaded_history=uploaded_history_map, years=history_years)
    
    # Enrich features
    hourly_df = enrich_price_features(hourly_df, "hourly", benchmark_df, sector_df)
    daily_df = enrich_price_features(daily_df, "daily", benchmark_df, sector_df)
    weekly_df = enrich_price_features(weekly_df, "weekly", benchmark_df, sector_df)
    
    # Add forward returns
    daily_df = add_forward_returns(daily_df)
    
    if analysis_mode == "Calendar Backtest":
        # Run calendar backtest
        backtest_results = run_calendar_backtest(
            daily_df, 
            pd.Timestamp(backtest_start), 
            pd.Timestamp(backtest_end), 
            analog_count=top_analogs
        )
        detail_data[symbol] = {
            "backtest_results": backtest_results,
            "history_source": history_source,
        }
        
        if not backtest_results.empty:
            accuracy = (backtest_results["predicted_direction"].notna() & 
                       ((backtest_results["predicted_direction"] == "CALL" & backtest_results["actual_ret_1d"] > 0) |
                        (backtest_results["predicted_direction"] == "PUT" & backtest_results["actual_ret_1d"] < 0))).mean()
            
            results_rows.append({
                "Symbol": symbol,
                "Status": "OK",
                "History Source": history_source,
                "Backtest Period": f"{backtest_start} to {backtest_end}",
                "Predictions": len(backtest_results),
                "Accuracy (1d)": f"{accuracy*100:.1f}%",
            })
        else:
            results_rows.append({
                "Symbol": symbol,
                "Status": "Insufficient data",
                "History Source": history_source,
            })
    else:
        # Current mode - use latest data
        current_hourly = hourly_df.iloc[-1] if not hourly_df.empty else pd.Series()
        current_daily = daily_df.iloc[-1]
        current_weekly = weekly_df.iloc[-1] if not weekly_df.empty else pd.Series()
        
        # Classify states (TSI-centric)
        states = {
            "hourly": classify_timeframe_state(current_hourly, "hourly") if not current_hourly.empty else "No data",
            "daily": classify_timeframe_state(current_daily, "daily"),
            "weekly": classify_timeframe_state(current_weekly, "weekly") if not current_weekly.empty else "No data",
        }
        states["alignment"] = get_alignment_label(states["hourly"], states["daily"], states["weekly"])
        
        # Detect patterns
        flags = detect_pattern_flags(current_hourly, current_daily, current_weekly)
        
        # Find analogs
        analogs = nearest_analogs(daily_df, FEATURE_COLS, top_n=top_analogs)
        analog_summary = summarize_analogs(analogs)
        
        # Monte Carlo
        sim_paths, mc_summary = monte_carlo_from_analogs(analogs, horizon_days=5, n_sims=mc_sims)
        
        # Get TSI percentile (PRIMARY)
        tsi_pctile = current_daily.get("tsi_pctile", 0.5)
        
        # Get direction and confidence
        direction, strength = get_direction_from_tsi(tsi_pctile, states["daily"])
        confidence = get_confidence_from_analogs(analog_summary, flags, tsi_pctile)
        
        # Calculate scores
        scores = score_setup(states, flags, analog_summary)
        
        # Narrative
        narrative = describe_setup(states, flags, analog_summary, tsi_pctile)
        
        results_rows.append({
            "Symbol": symbol,
            "Status": "OK",
            "History Source": history_source,
            "Signal": "🟢 CALL" if direction == "CALL" else ("🔴 PUT" if direction == "PUT" else "⚪ NEUTRAL"),
            "Confidence": f"{confidence:.0f}%",
            "Long Score": scores["long_score"],
            "Short Score": scores["short_score"],
            "Alignment": states["alignment"],
            "Hourly": states["hourly"],
            "Daily": states["daily"],
            "Weekly": states["weekly"],
            "TSI %ile": round(tsi_pctile * 100, 1),
            "TSI Raw": round(current_daily.get("tsi", 0), 2),
            "Bear Kiss": flags.get("bear_kiss_hourly", False),
            "Daily Div": flags.get("daily_bearish_divergence", False),
            "Overheated": flags.get("overheated", False),
            "Washed Out": flags.get("washed_out", False),
            "Fwd 1d Med": round(analog_summary.get("fwd_ret_1_median", 0) * 100, 2),
            "Fwd 5d Med": round(analog_summary.get("fwd_ret_5_median", 0) * 100, 2),
            "MC 5d Mean": round(mc_summary.get("mean", 0) * 100, 2),
        })
        
        detail_data[symbol] = {
            "hourly": hourly_df,
            "daily": daily_df,
            "weekly": weekly_df,
            "history_source": history_source,
            "states": states,
            "flags": flags,
            "analogs": analogs,
            "analog_summary": analog_summary,
            "sim_paths": sim_paths,
            "mc_summary": mc_summary,
            "scores": scores,
            "current_hourly": current_hourly,
            "current_daily": current_daily,
            "current_weekly": current_weekly,
            "direction": direction,
            "confidence": confidence,
            "narrative": narrative,
        }

progress_bar.empty()
status_text.empty()

# Display results
if analysis_mode == "Calendar Backtest":
    st.subheader("📅 Calendar Backtest Results")
else:
    st.subheader("🏆 Ranked Results (Current Mode)")

results_df = pd.DataFrame(results_rows)
if not results_df.empty:
    st.dataframe(results_df, width='stretch', hide_index=True)
    
    # Download button
    csv_buffer = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download results CSV", csv_buffer, "stock_analysis_results.csv", "text/csv")

# Detailed view for selected symbol (current mode only)
if analysis_mode != "Calendar Backtest" and detail_data:
    st.subheader("🔬 Detailed Analysis")
    valid_symbols = [r["Symbol"] for r in results_rows if r.get("Status") == "OK"]
    
    if valid_symbols:
        selected_symbol = st.selectbox("Select symbol for detailed view", valid_symbols)
        
        if selected_symbol and selected_symbol in detail_data:
            data = detail_data[selected_symbol]
            
            # Score metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Signal", "🟢 CALL" if data["direction"] == "CALL" else ("🔴 PUT" if data["direction"] == "PUT" else "⚪ NEUTRAL"))
            col2.metric("Confidence", f"{data['confidence']:.0f}%")
            col3.metric("Long Score", data["scores"]["long_score"])
            col4.metric("Short Score", data["scores"]["short_score"])
            col5.metric("TSI Percentile", f"{data['current_daily'].get('tsi_pctile', 0.5)*100:.1f}%")
            
            # TSI raw value
            st.metric("TSI 25,13,7 Raw Value", f"{data['current_daily'].get('tsi', 0):.2f}", 
                     delta=f"Slope: {data['current_daily'].get('tsi_slope_3', 0):.2f}")
            
            # State summary
            st.markdown(f"""
            **Timeframe States:** Weekly: {data['states']['weekly']} | Daily: {data['states']['daily']} | Hourly: {data['states']['hourly']}
            """)
            
            # Alignment
            st.info(f"**Alignment:** {data['states']['alignment']}")
            
            # Narrative
            st.markdown(f"**Interpretation:** {data['narrative']}")
            
            # Flags
            flags_df = pd.DataFrame([{"Pattern": k.replace("_", " ").title(), "Detected": v} for k, v in data["flags"].items()])
            st.dataframe(flags_df, width='stretch', hide_index=True)
            
            # Price chart
            st.subheader("📈 Price & Oscillator Charts")
            plot_price_and_oscillator(selected_symbol, data["daily"].tail(220), data["weekly"].tail(140), data["hourly"].tail(120))
            
            # Analog analysis
            st.subheader("📊 Historical Analog Analysis")
            
            if not data["analogs"].empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **Analog Summary (n={len(data['analogs'])})**
                    - 1-day forward mean: {data['analog_summary'].get('fwd_ret_1_mean', 0)*100:.2f}%
                    - 1-day forward median: {data['analog_summary'].get('fwd_ret_1_median', 0)*100:.2f}%
                    - 1-day down probability: {data['analog_summary'].get('fwd_ret_1_p_down', 0)*100:.1f}%
                    - 5-day forward median: {data['analog_summary'].get('fwd_ret_5_median', 0)*100:.2f}%
                    """)
                with col2:
                    st.markdown(f"""
                    **Key Statistics**
                    - Analog similarity range: {data['analogs']['similarity'].min():.2f} - {data['analogs']['similarity'].max():.2f}
                    - Most similar date: {data['analogs'].iloc[0].name.strftime('%Y-%m-%d') if not data['analogs'].empty else 'N/A'}
                    - Distance: {data['analogs'].iloc[0].get('distance', 0):.2f}
                    """)
                
                # Display analog table
                display_analogs = data["analogs"].copy()
                if not display_analogs.empty:
                    display_analogs.index = display_analogs.index.strftime("%Y-%m-%d")
                    st.dataframe(display_analogs[["Close", "similarity", "fwd_ret_1", "fwd_ret_2", "fwd_ret_5"]].head(10), width='stretch')
                
                # Analog distribution plot
                plot_analog_distribution(data["analogs"])
            else:
                st.info("No sufficient analog data for this symbol")
            
            # Monte Carlo
            st.subheader("🎲 Monte Carlo Simulation (5-day forecast)")
            plot_monte_carlo(data["sim_paths"], data["mc_summary"], selected_symbol)
            
            # Option chain
            if show_options and data["direction"] is not None:
                st.subheader("📦 Option Chain")
                opts = get_option_candidates(selected_symbol, data["direction"])
                if opts is not None and not opts.empty:
                    st.markdown(f"**{data['direction']} options based on TSI 25,13,7 signal**")
                    st.dataframe(opts, width='stretch')
                else:
                    st.info(f"No {data['direction']} options available for {selected_symbol}")

elif analysis_mode == "Calendar Backtest" and detail_data:
    st.subheader("📅 Calendar Backtest Details")
    selected_symbol = st.selectbox("Select symbol to view backtest", list(detail_data.keys()))
    
    if selected_symbol and selected_symbol in detail_data:
        backtest_df = detail_data[selected_symbol].get("backtest_results", pd.DataFrame())
        
        if not backtest_df.empty:
            st.write(f"**Backtest Period:** {backtest_df['date'].min().date()} to {backtest_df['date'].max().date()}")
            st.write(f"**Total Predictions:** {len(backtest_df)}")
            
            # Calculate accuracy
            backtest_df["correct"] = (
                (backtest_df["predicted_direction"] == "CALL") & (backtest_df["actual_ret_1d"] > 0) |
                (backtest_df["predicted_direction"] == "PUT") & (backtest_df["actual_ret_1d"] < 0)
            )
            accuracy = backtest_df["correct"].mean() * 100
            
            st.metric("Overall Accuracy (1-day)", f"{accuracy:.1f}%")
            
            # Display backtest results
            display_cols = ["date", "close", "tsi_pctile", "predicted_direction", "analog_pred_1d", "actual_ret_1d", "correct"]
            st.dataframe(backtest_df[display_cols].tail(50), width='stretch')
            
            # Download backtest results
            csv_buffer = backtest_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download backtest results CSV", csv_buffer, f"{selected_symbol}_backtest.csv", "text/csv")
        else:
            st.info("No backtest results available for this symbol")

st.success("Analysis complete")
