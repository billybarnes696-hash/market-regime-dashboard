"""
Stock Analyzer Ultimate
Multi-timeframe analysis | Single TSI 25,13,7 | Percentile-based signals | Option chains | Calendar backtesting
"""

import hashlib
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

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
    "tsi_pctile",           # TSI 25,13,7 percentile (primary)
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
    "volume_pctile",
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
    """True Strength Index - DEFAULT 25,13,7 (primary overbought/oversold)"""
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


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_data(ticker: str, interval: str, period: str = "3y") -> pd.DataFrame:
    """Fetch data from Yahoo Finance"""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={c: c.title() for c in df.columns})
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
        return df
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_multi_timeframe(ticker: str, sector_etf: Optional[str], benchmark: str) -> Dict[str, pd.DataFrame]:
    """Fetch hourly, daily, and weekly data"""
    hourly = fetch_yahoo_data(ticker, "1h", period="60d")
    daily = fetch_yahoo_data(ticker, "1d", period="3y")
    weekly = resample_ohlcv(daily, "W-FRI") if not daily.empty else pd.DataFrame()
    
    return {"hourly": hourly, "daily": daily, "weekly": weekly}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def enrich_price_features(df: pd.DataFrame, timeframe: str, benchmark_df: pd.DataFrame = None, sector_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add all technical features - SINGLE TSI 25,13,7 as primary"""
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
    
    # TSI 25,13,7 (PRIMARY)
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
    
    # Candle features
    out["close_in_range"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["High"] - out[["Close", "Open"]].max(axis=1)) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["body_pct"] = (out["Close"] - out["Open"]).abs() / (out["High"] - out["Low"]).replace(0, np.nan)
    out["candle_score"] = 50 + out["upper_wick_pct"].fillna(0) * 25 - out["body_pct"].fillna(0) * 15
    
    # Price slope
    out["price_slope_3"] = slope(out["Close"], 3)
    out["price_slope_5"] = slope(out["Close"], 5)
    
    # Hourly-specific features
    if timeframe == "hourly":
        out["vwap"] = compute_vwap(out)
        out["dist_vwap_pct"] = (out["Close"] / out["vwap"]) - 1
        out["intraday_atr_pct"] = out["atr_14"] / out["Close"]
    
    # Percentile ranks (for cross-stock comparison)
    window = 252 if timeframe != "hourly" else 120
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
# SIGNAL DETECTION
# ============================================================================

def classify_timeframe_state(row: pd.Series, timeframe: str) -> str:
    """Classify a single timeframe as Bullish/Neutral/Bearish/Stalling"""
    if row.empty:
        return "Unknown"
    
    tsi_val = row.get("tsi", 0)
    tsi_pctile = row.get("tsi_pctile", 0.5)
    rsi_pctile = row.get("rsi_14_pctile", 0.5)
    price_slope = row.get("price_slope_3", 0)
    tsi_slope = row.get("tsi_slope_3", 0)
    price_vs_ema = row.get("dist_ema20_pct", 0)
    
    # Bullish conditions
    if tsi_pctile > 0.7 and price_slope > 0 and price_vs_ema > -0.02:
        return "Bullish"
    
    # Bearish conditions
    if tsi_pctile < 0.3 and price_slope < 0 and price_vs_ema < 0.02:
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
        hourly_row.get("price_slope_3", 0) > 0
    )
    
    # Dead cat bounce risk
    flags["dead_cat_bounce_risk"] = bool(
        weekly_row.get("dist_ema20_pct", 0) < 0 and
        daily_row.get("price_slope_3", 0) > 0 and
        daily_row.get("rsi_14_pctile", 0.5) < 0.55 and
        daily_row.get("dist_ema20_pct", 0) < 0
    )
    
    # Post-bottom thrust (recovery signal)
    flags["post_bottom_thrust"] = bool(
        daily_row.get("dist_ema20_pct", 0) > 0 and
        daily_row.get("dist_sma50_pct", -1) > -0.02 and
        daily_row.get("rsi_14_pctile", 0) > 0.55 and
        weekly_row.get("price_slope_3", 0) >= 0
    )
    
    # Overheated (TSI > 70th percentile)
    flags["overheated"] = bool(daily_row.get("tsi_pctile", 0) > 0.7)
    
    # Washed out (TSI < 30th percentile)
    flags["washed_out"] = bool(daily_row.get("tsi_pctile", 0) < 0.3)
    
    return flags


def get_alignment_label(hourly_state: str, daily_state: str, weekly_state: str) -> str:
    """Get alignment description across timeframes"""
    if hourly_state == "Bullish" and daily_state in ["Bearish", "Stalling"]:
        return "Hourly thrust vs weak daily backdrop"
    if hourly_state == "Bullish" and daily_state == "Bullish" and weekly_state == "Bullish":
        return "Full bullish alignment"
    if hourly_state in ["Stalling", "Bearish"] and daily_state == "Bullish":
        return "Short-term fade in larger uptrend"
    if weekly_state == "Bearish" and hourly_state == "Bullish":
        return "Countertrend bounce / fakeout risk"
    if hourly_state == "Bear Kiss":
        return "Bear kiss pattern - potential rollover"
    return "Mixed alignment"


# ============================================================================
# ANALOG MATCHING & MONTE CARLO
# ============================================================================

def find_analogs(df: pd.DataFrame, feature_cols: List[str], top_n: int = 15) -> pd.DataFrame:
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
    
    # Add similarity score
    analogs["similarity"] = 1 / (1 + analogs["distance"])
    
    # Return columns
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


def monte_carlo_from_analogs(analogs: pd.DataFrame, horizon_days: int = 5, n_sims: int = 10000) -> Dict:
    """Monte Carlo simulation conditioned on analogs"""
    if analogs.empty:
        return {"mean": 0, "median": 0, "p5": 0, "p25": 0, "p75": 0, "p95": 0, "prob_negative": 0.5}
    
    return_cols = [c for c in [f"fwd_ret_{i}" for i in range(1, horizon_days + 1)] if c in analogs.columns]
    if not return_cols:
        return {"mean": 0, "median": 0, "p5": 0, "p25": 0, "p75": 0, "p95": 0, "prob_negative": 0.5}
    
    rng = np.random.default_rng(42)
    vals = analogs[return_cols].dropna().values
    
    if len(vals) == 0:
        return {"mean": 0, "median": 0, "p5": 0, "p25": 0, "p75": 0, "p95": 0, "prob_negative": 0.5}
    
    terminal_returns = []
    for _ in range(n_sims):
        row = vals[rng.integers(0, len(vals))]
        cumulative = np.prod(1 + row) - 1
        terminal_returns.append(cumulative)
    
    terminal_returns = np.array(terminal_returns)
    
    return {
        "mean": float(np.mean(terminal_returns)),
        "median": float(np.median(terminal_returns)),
        "p5": float(np.percentile(terminal_returns, 5)),
        "p25": float(np.percentile(terminal_returns, 25)),
        "p75": float(np.percentile(terminal_returns, 75)),
        "p95": float(np.percentile(terminal_returns, 95)),
        "prob_negative": float((terminal_returns < 0).mean()),
    }


def calculate_predictive_accuracy(df: pd.DataFrame, signal_col: str, return_col: str) -> Dict:
    """Calculate how well a signal predicts future returns"""
    if df.empty or signal_col not in df.columns or return_col not in df.columns:
        return {"accuracy": 0, "precision": 0, "recall": 0, "sharpe": 0, "trades": 0}
    
    signals = df[signal_col].fillna(0).astype(int)
    future_returns = df[return_col].fillna(0)
    
    valid_mask = (signals.notna()) & (future_returns.notna())
    signals = signals[valid_mask]
    future_returns = future_returns[valid_mask]
    
    if len(signals) == 0 or signals.sum() == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "sharpe": 0, "trades": 0}
    
    # Directional accuracy
    correct = ((signals == 1) & (future_returns > 0)) | ((signals == 0) & (future_returns <= 0))
    accuracy = correct.sum() / len(signals)
    
    # Precision
    signal_true = signals == 1
    if signal_true.sum() > 0:
        precision = ((signals == 1) & (future_returns > 0)).sum() / signal_true.sum()
    else:
        precision = 0
    
    # Recall
    positive_returns = future_returns > 0
    if positive_returns.sum() > 0:
        recall = ((signals == 1) & positive_returns).sum() / positive_returns.sum()
    else:
        recall = 0
    
    # Sharpe
    strategy_returns = future_returns * signals
    if strategy_returns.std() > 0:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "sharpe": round(sharpe, 2),
        "trades": int(signal_true.sum()),
    }


# ============================================================================
# SCORING
# ============================================================================

def calculate_scores(states: Dict[str, str], flags: Dict[str, bool], analog_summary: Dict[str, float]) -> Dict[str, float]:
    """Calculate long and short scores (0-100)"""
    long_score = 50.0
    short_score = 50.0
    
    # Timeframe alignment
    if states.get("hourly") == "Bullish":
        long_score += 5
        short_score -= 5
    if states.get("daily") == "Bullish":
        long_score += 8
        short_score -= 8
    if states.get("weekly") == "Bullish":
        long_score += 10
        short_score -= 10
    
    # Pattern flags
    if flags.get("bear_kiss_hourly"):
        long_score -= 10
        short_score += 10
    if flags.get("daily_bearish_divergence"):
        long_score -= 10
        short_score += 10
    if flags.get("pinned_continuation_risk"):
        long_score -= 3
        short_score += 3
    if flags.get("dead_cat_bounce_risk"):
        long_score -= 15
        short_score += 15
    if flags.get("post_bottom_thrust"):
        long_score += 7
        short_score -= 7
    if flags.get("overheated"):
        long_score -= 8
        short_score += 8
    if flags.get("washed_out"):
        long_score += 8
        short_score -= 8
    
    # Analog forward returns
    p_down = analog_summary.get("fwd_ret_1_p_down", np.nan)
    if not pd.isna(p_down):
        long_score += (0.5 - p_down) * 20
        short_score -= (0.5 - p_down) * 20
    
    mean_ret = analog_summary.get("fwd_ret_2_mean", 0)
    if not pd.isna(mean_ret):
        long_score += max(-10, min(10, mean_ret * 1000))
        short_score -= max(-10, min(10, mean_ret * 1000))
    
    long_score = float(np.clip(long_score, 0, 100))
    short_score = float(np.clip(short_score, 0, 100))
    
    return {"long_score": round(long_score, 1), "short_score": round(short_score, 1)}


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
        now = pd.Timestamp.utcnow().tz_localize(None)
        
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
# VISUALIZATION
# ============================================================================

def plot_multi_timeframe(ticker: str, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, hourly_df: pd.DataFrame):
    """Plot price and oscillators across timeframes"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"{ticker} Daily Price", "Daily Oscillators (Percentile Ranks)", "Hourly Oscillators"],
        row_heights=[0.45, 0.27, 0.28],
    )
    
    if not daily_df.empty:
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
        
        # Percentile-based oscillators
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["rsi_14_pctile"] * 100, name="RSI14 %ile"), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["cci_20_pctile"] * 100, name="CCI20 %ile"), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["tsi_pctile"] * 100, name="TSI 25,13,7 %ile", line=dict(color="red", width=2)), row=2, col=1)
        
        # Add 70/30 threshold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    if not hourly_df.empty:
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["rsi_14"], name="Hourly RSI14"), row=3, col=1)
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["cci_20"], name="Hourly CCI20"), row=3, col=1)
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["tsi"], name="Hourly TSI"), row=3, col=1)
    
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend_orientation="h")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Percentile (%)", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_monte_carlo(mc_results: Dict, ticker: str):
    """Plot Monte Carlo distribution"""
    if not mc_results or mc_results.get("mean") == 0:
        st.info("Insufficient data for Monte Carlo simulation")
        return
    
    # Create distribution visualization
    fig = go.Figure()
    
    # Add confidence intervals as a bar chart
    categories = ["p5", "p25", "median", "p75", "p95"]
    values = [mc_results.get("p5", 0) * 100, mc_results.get("p25", 0) * 100, 
              mc_results.get("median", 0) * 100, mc_results.get("p75", 0) * 100, 
              mc_results.get("p95", 0) * 100]
    colors = ["red", "orange", "green", "orange", "red"]
    
    fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors, name="Return %"))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title=f"{ticker} - Monte Carlo Forecast (5-day)",
        xaxis_title="Percentile",
        yaxis_title="Expected Return (%)",
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display stats
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean", f"{mc_results.get('mean', 0)*100:.2f}%")
    col2.metric("Median", f"{mc_results.get('median', 0)*100:.2f}%")
    col3.metric("P5 (Bear)", f"{mc_results.get('p5', 0)*100:.2f}%")
    col4.metric("P95 (Bull)", f"{mc_results.get('p95', 0)*100:.2f}%")
    col5.metric("Downside Prob", f"{mc_results.get('prob_negative', 0)*100:.1f}%")


def plot_analog_distribution(analogs: pd.DataFrame):
    """Plot analog forward return distribution"""
    if analogs.empty:
        return
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["1-Day Forward Returns", "5-Day Forward Returns"])
    
    if "fwd_ret_1" in analogs.columns:
        fig.add_trace(go.Histogram(x=analogs["fwd_ret_1"] * 100, nbinsx=20, name="1-day"), row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    
    if "fwd_ret_5" in analogs.columns:
        fig.add_trace(go.Histogram(x=analogs["fwd_ret_5"] * 100, nbinsx=20, name="5-day"), row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(height=400, title="Analog Forward Return Distributions")
    fig.update_xaxes(title_text="Return (%)")
    fig.update_yaxes(title_text="Frequency")
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================

st.set_page_config(layout="wide", page_title="Stock Analyzer Ultimate", page_icon="📊")

st.title("📊 Stock Analyzer Ultimate")
st.caption("Multi-timeframe | TSI 25,13,7 Primary | Percentile-based | Option Chains | Calendar Backtesting")

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
        uploaded_file = st.file_uploader("Upload CSV/XLSX with symbols", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            first_col = df.columns[0]
            symbols = df[first_col].dropna().astype(str).str.upper().tolist()
            st.success(f"Loaded {len(symbols)} symbols")
    
    st.header("⚙️ Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=0)
    sector_etf = st.selectbox("Sector context", ["None"] + list(SECTOR_ETFS.keys()), index=0)
    
    st.header("🔬 Analysis Options")
    top_analogs = st.slider("Number of analogs", 5, 30, 15)
    mc_sims = st.slider("Monte Carlo simulations", 500, 20000, 10000, step=500)
    
    st.header("🕐 Mode")
    analysis_mode = st.radio("Analysis mode", ["Current", "Calendar Backtest"], index=0)
    
    if analysis_mode == "Calendar Backtest":
        backtest_date = st.date_input("Test date", value=pd.Timestamp.today().date() - pd.Timedelta(days=30))
    
    show_options = st.checkbox("Show option chains (current mode only)", value=True)
    rebuild_cache = st.checkbox("Rebuild cache", value=False)
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

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

# Run analysis for all symbols
progress_bar = st.progress(0)
status_text = st.empty()

results_rows = []
detail_data = {}

for i, symbol in enumerate(symbols):
    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
    progress_bar.progress((i + 1) / len(symbols))
    
    # Fetch data
    data = fetch_multi_timeframe(symbol, sector_etf if sector_etf != "None" else None, benchmark)
    
    hourly_df = data["hourly"]
    daily_df = data["daily"]
    weekly_df = data["weekly"]
    
    if daily_df.empty:
        results_rows.append({"Symbol": symbol, "Status": "No data", "Long Score": 0, "Short Score": 0})
        continue
    
    # Fetch benchmark and sector data for features
    benchmark_df = fetch_yahoo_data(benchmark, "1d", "3y")
    sector_df = fetch_yahoo_data(sector_etf, "1d", "3y") if sector_etf != "None" else None
    
    # Enrich features
    hourly_df = enrich_price_features(hourly_df, "hourly", benchmark_df, sector_df)
    daily_df = enrich_price_features(daily_df, "daily", benchmark_df, sector_df)
    weekly_df = enrich_price_features(weekly_df, "weekly", benchmark_df, sector_df)
    
    # Add forward returns
    daily_df = add_forward_returns(daily_df)
    
    if analysis_mode == "Calendar Backtest":
        # Get row as of specific date
        target_date = pd.Timestamp(backtest_date)
        daily_df_filtered = daily_df[daily_df.index <= target_date]
        if daily_df_filtered.empty:
            results_rows.append({"Symbol": symbol, "Status": f"No data for {backtest_date}", "Long Score": 0, "Short Score": 0})
            continue
        
        # Get last row before target date for features
        current_hourly = hourly_df[hourly_df.index <= target_date].iloc[-1] if not hourly_df.empty else pd.Series()
        current_daily = daily_df_filtered.iloc[-1]
        current_weekly = weekly_df[weekly_df.index <= target_date].iloc[-1] if not weekly_df.empty else pd.Series()
    else:
        # Current mode - use latest data
        current_hourly = hourly_df.iloc[-1] if not hourly_df.empty else pd.Series()
        current_daily = daily_df.iloc[-1]
        current_weekly = weekly_df.iloc[-1] if not weekly_df.empty else pd.Series()
    
    # Classify states
    states = {
        "hourly": classify_timeframe_state(current_hourly, "hourly") if not current_hourly.empty else "No data",
        "daily": classify_timeframe_state(current_daily, "daily"),
        "weekly": classify_timeframe_state(current_weekly, "weekly") if not current_weekly.empty else "No data",
    }
    states["alignment"] = get_alignment_label(states["hourly"], states["daily"], states["weekly"])
    
    # Detect patterns
    flags = detect_pattern_flags(current_hourly, current_daily, current_weekly)
    
    # Find analogs
    analogs = find_analogs(daily_df, FEATURE_COLS, top_n=top_analogs)
    analog_summary = summarize_analogs(analogs)
    
    # Monte Carlo
    mc_results = monte_carlo_from_analogs(analogs, horizon_days=5, n_sims=mc_sims)
    
    # Calculate scores
    scores = calculate_scores(states, flags, analog_summary)
    
    # Store results
    results_rows.append({
        "Symbol": symbol,
        "Status": "OK",
        "Long Score": scores["long_score"],
        "Short Score": scores["short_score"],
        "Alignment": states["alignment"],
        "Hourly": states["hourly"],
        "Daily": states["daily"],
        "Weekly": states["weekly"],
        "Bear Kiss": flags.get("bear_kiss_hourly", False),
        "Daily Div": flags.get("daily_bearish_divergence", False),
        "Overheated": flags.get("overheated", False),
        "Washed Out": flags.get("washed_out", False),
        "TSI %ile": round(current_daily.get("tsi_pctile", 0.5) * 100, 1),
        "Fwd 1d Med": round(analog_summary.get("fwd_ret_1_median", 0) * 100, 2),
        "Fwd 5d Med": round(analog_summary.get("fwd_ret_5_median", 0) * 100, 2),
        "MC 5d Mean": round(mc_results.get("mean", 0) * 100, 2),
    })
    
    detail_data[symbol] = {
        "hourly": hourly_df,
        "daily": daily_df,
        "weekly": weekly_df,
        "states": states,
        "flags": flags,
        "analogs": analogs,
        "analog_summary": analog_summary,
        "mc_results": mc_results,
        "scores": scores,
        "current_hourly": current_hourly,
        "current_daily": current_daily,
        "current_weekly": current_weekly,
    }

progress_bar.empty()
status_text.empty()

# Display results
results_df = pd.DataFrame(results_rows).sort_values("Long Score", ascending=False)

st.subheader("🏆 Ranked Results")
st.dataframe(results_df, use_container_width=True, hide_index=True)

# Download button
csv_buffer = results_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download results CSV", csv_buffer, "stock_analysis_results.csv", "text/csv")

# Detailed view for selected symbol
st.subheader("🔬 Detailed Analysis")
selected_symbol = st.selectbox("Select symbol for detailed view", [r["Symbol"] for r in results_rows if r["Status"] == "OK"])

if selected_symbol and selected_symbol in detail_data:
    data = detail_data[selected_symbol]
    
    # Score metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Long Score", data["scores"]["long_score"])
    col2.metric("Short Score", data["scores"]["short_score"])
    col3.metric("Alignment", data["states"]["alignment"])
    col4.metric("TSI Percentile", f"{data['current_daily'].get('tsi_pctile', 0.5)*100:.1f}%")
    col5.metric("5-day MC Mean", f"{data['mc_results'].get('mean', 0)*100:.2f}%")
    
    # State summary
    st.markdown(f"""
    **Timeframe States:** Weekly: {data['states']['weekly']} | Daily: {data['states']['daily']} | Hourly: {data['states']['hourly']}
    """)
    
    # Flags
    flags_df = pd.DataFrame([{"Pattern": k.replace("_", " ").title(), "Detected": v} for k, v in data["flags"].items()])
    st.dataframe(flags_df, use_container_width=True, hide_index=True)
    
    # Price chart
    st.subheader("📈 Price & Oscillator Charts")
    plot_multi_timeframe(selected_symbol, data["daily"].tail(220), data["weekly"].tail(140), data["hourly"].tail(120))
    
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
            st.dataframe(display_analogs[["Close", "similarity", "fwd_ret_1", "fwd_ret_2", "fwd_ret_5"]].head(10), use_container_width=True)
        
        # Analog distribution plot
        plot_analog_distribution(data["analogs"])
    else:
        st.info("No sufficient analog data for this symbol")
    
    # Monte Carlo
    st.subheader("🎲 Monte Carlo Simulation (5-day forecast)")
    plot_monte_carlo(data["mc_results"], selected_symbol)
    
    # Option chain (current mode only)
    if show_options and analysis_mode != "Calendar Backtest":
        st.subheader("📦 Option Chain")
        
        direction = "CALL" if data["scores"]["long_score"] > data["scores"]["short_score"] else "PUT"
        
        opts = get_option_candidates(selected_symbol, direction)
        if opts is not None and not opts.empty:
            st.markdown(f"**{direction} options based on signal direction**")
            st.dataframe(opts, use_container_width=True)
        else:
            st.info(f"No {direction} options available for {selected_symbol}")

st.success("Analysis complete")
