import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

try:
    from defeatbeta_api import Ticker
except Exception:
    from defeatbeta_api.data.ticker import Ticker

st.set_page_config(layout="wide", page_title="Diamond Scanner - Bull/Bear with File Upload")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# LIQUIDITY FILTER CONFIGURATION
# ============================================================================

LIQUIDITY_CONFIG = {
    "min_avg_volume": 500000,       # Minimum 500k average daily volume
    "min_price": 5.00,              # Minimum $5 stock price
    "max_spread_pct": 0.02,         # Maximum 2% bid-ask spread
    "min_market_cap_billions": 1.0, # Minimum $1B market cap
    "exclude_etf": True,            # Skip ETFs
    "etf_min_volume": 1000000,      # If including ETFs, require 1M volume
    "min_sctr": 20,                 # Minimum SCTR score (if available)
}

# ============================================================================
# FEATURE COLUMNS
# ============================================================================

FEATURE_COLS = [
    "TSI_424", "TSI_424_minus_sig", "TSI_424_slope_1", "TSI_424_slope_3",
    "TSI_747", "TSI_747_minus_sig", "TSI_747_slope_1", "TSI_747_slope_3",
    "TSI_1377", "TSI_1377_minus_sig", "TSI_1377_slope_1", "TSI_1377_slope_3",
    "TSI_25137", "TSI_25137_minus_sig", "TSI_25137_slope_1", "TSI_25137_slope_3",
    "CCI15", "CCI20", "CCI15_days_fading", "CCI20_days_fading",
    "BBP", "BBP_delta_1", "BBP_days_falling",
    "RSI14", "RSI14_slope_3",
    "EXT_SMA10", "EXT_SMA20", "EXT_VWAP",
    "RS_SLOPE_5", "RS_SLOPE_10",
    "candle_score", "candle_close_loc", "upper_wick_pct", "body_pct",
    "pinned_424", "pinned_747", "pinned_1377", "pinned_25137",
    "state_424_code", "state_747_code", "state_1377_code", "state_25137_code",
    "is_bear_kiss", "is_hot", "is_ripping", "is_fading_stack", "bear_kiss_score",
    "bull_kiss_score", "is_bull_kiss",
    "ADX14", "ADX14_slope_3",
]

STATE_CODE_MAP = {
    "below_zero": 0,
    "bull_repair": 1,
    "bull_continuation": 2,
    "near_pinned": 3,
    "pinned": 4,
    "bear_kiss": 5,
    "rolling": 6,
}


# ============================================================================
# INDICATOR FUNCTIONS
# ============================================================================

def tsi(series: pd.Series, r: int, s: int, signal: int = 7) -> Tuple[pd.Series, pd.Series]:
    m = series.diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    abs_m = m.abs()
    ema3 = abs_m.ewm(span=r, adjust=False).mean()
    ema4 = ema3.ewm(span=s, adjust=False).mean()
    tsi_val = 100 * ema2 / ema4.replace(0, np.nan)
    sig = tsi_val.ewm(span=signal, adjust=False).mean()
    return tsi_val, sig


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def bbpct(close: pd.Series, n: int = 20) -> pd.Series:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def rolling_vwap(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    num = (tp * vol).rolling(lookback, min_periods=max(5, lookback // 2)).sum()
    den = vol.rolling(lookback, min_periods=max(5, lookback // 2)).sum()
    return num / den


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


def count_consecutive(cond: pd.Series) -> pd.Series:
    out = np.zeros(len(cond), dtype=int)
    count = 0
    vals = cond.fillna(False).astype(bool).tolist()
    for i, ok in enumerate(vals):
        count = count + 1 if ok else 0
        out[i] = count
    return pd.Series(out, index=cond.index)


# ============================================================================
# LIQUIDITY SCREENING
# ============================================================================

def check_liquidity(symbol: str) -> Tuple[bool, Dict]:
    """Check if a symbol meets liquidity requirements"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get liquidity metrics
        avg_volume = info.get('averageVolume', 0)
        current_price = info.get('regularMarketPrice', 0)
        market_cap = info.get('marketCap', 0)
        bid = info.get('bid', 0)
        ask = info.get('ask', 0)
        quote_type = info.get('quoteType', '')
        
        # Calculate spread percentage
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2)
        else:
            spread_pct = 0.05  # Default high spread if unknown
        
        # Check ETF status
        is_etf = quote_type == 'ETF'
        
        # Apply filters
        checks = {
            'min_volume': avg_volume >= LIQUIDITY_CONFIG['min_avg_volume'],
            'min_price': current_price >= LIQUIDITY_CONFIG['min_price'],
            'max_spread': spread_pct <= LIQUIDITY_CONFIG['max_spread_pct'],
            'min_market_cap': market_cap >= LIQUIDITY_CONFIG['min_market_cap_billions'] * 1e9,
        }
        
        # ETF handling
        if is_etf and LIQUIDITY_CONFIG['exclude_etf']:
            checks['etf_allowed'] = False
        elif is_etf:
            checks['etf_allowed'] = avg_volume >= LIQUIDITY_CONFIG['etf_min_volume']
        else:
            checks['etf_allowed'] = True
        
        passed = all(checks.values())
        
        metrics = {
            'symbol': symbol,
            'avg_volume': avg_volume,
            'price': current_price,
            'market_cap_b': market_cap / 1e9,
            'spread_pct': spread_pct,
            'is_etf': is_etf,
            'checks': checks,
        }
        
        return passed, metrics
        
    except Exception as e:
        return False, {'symbol': symbol, 'error': str(e)}


# ============================================================================
# BULL / BEAR SIGNAL DETECTION
# ============================================================================

def calculate_bull_score(row: pd.Series) -> Tuple[float, List[str]]:
    """Calculate bullish signal score with reasons"""
    score = 0.0
    reasons = []
    
    # TSI Bullish patterns
    if row.get("TSI_747", 0) > 70 and row.get("TSI_747_slope_1", 0) > 0:
        score += 2.0
        reasons.append("TSI_747 bullish trending")
    if row.get("TSI_747_bull_kiss", 0):
        score += 3.0
        reasons.append("TSI_747 bull kiss")
    if row.get("TSI_424", 0) > 95 and row.get("TSI_424_slope_1", 0) > 0:
        score += 2.0
        reasons.append("TSI_424 pinned bullish")
    
    # RSI Bullish
    if row.get("RSI14", 0) < 30:
        score += 2.0
        reasons.append("RSI oversold")
    if 30 < row.get("RSI14", 0) < 40 and row.get("RSI14_slope_3", 0) > 0:
        score += 2.0
        reasons.append("RSI recovering")
    
    # CCI Bullish
    if row.get("CCI15", 0) < -100:
        score += 2.0
        reasons.append("CCI extreme oversold")
    
    # BB% Bullish
    if row.get("BBP", 0) < 0:
        score += 2.0
        reasons.append("BB% below lower band")
    
    # Candle Bullish
    if row.get("candle_score", 0) >= 68:
        score += 1.0
        reasons.append("strong rejection candle")
    
    # Bear kiss absence
    if row.get("bear_kiss_score", 0) < 5:
        score += 1.0
        reasons.append("no bear kiss pressure")
    
    return min(score, 10.0), reasons[:5]


def calculate_bear_score(row: pd.Series) -> Tuple[float, List[str]]:
    """Calculate bearish signal score with reasons"""
    score = 0.0
    reasons = []
    
    # TSI Bearish patterns
    if row.get("TSI_747", 0) > 70 and row.get("TSI_747_slope_1", 0) <= 0:
        score += 2.0
        reasons.append("TSI_747 rolling over")
    if row.get("TSI_747_bear_kiss", 0):
        score += 3.0
        reasons.append("TSI_747 bear kiss")
    
    # RSI Bearish
    if row.get("RSI14", 0) > 70:
        score += 2.0
        reasons.append("RSI overbought")
    if 60 < row.get("RSI14", 0) < 70 and row.get("RSI14_slope_3", 0) < 0:
        score += 2.0
        reasons.append("RSI rolling from overbought")
    
    # CCI Bearish
    if row.get("CCI15", 0) > 100:
        score += 2.0
        reasons.append("CCI extreme overbought")
    
    # BB% Bearish
    if row.get("BBP", 0) > 1:
        score += 2.0
        reasons.append("BB% above upper band")
    
    # Existing bear kiss score
    bear_kiss = row.get("bear_kiss_score", 0)
    if bear_kiss >= 8:
        score += 3.0
        reasons.append(f"bear kiss score {bear_kiss:.0f}")
    elif bear_kiss >= 5:
        score += 1.5
        reasons.append(f"bear kiss score {bear_kiss:.0f}")
    
    return min(score, 10.0), reasons[:5]


def get_signal_type(bull_score: float, bear_score: float) -> Tuple[str, str, float]:
    """Determine overall signal type and confidence"""
    if bull_score >= 6 and bull_score > bear_score + 2:
        return "🟢 BULL", "CALL", bull_score / 10
    elif bear_score >= 6 and bear_score > bull_score + 2:
        return "🔴 BEAR", "PUT", bear_score / 10
    elif bull_score >= 4 and bull_score > bear_score:
        return "🟡 BULLISH BIAS", "CALL", bull_score / 10 * 0.7
    elif bear_score >= 4 and bear_score > bull_score:
        return "🟠 BEARISH BIAS", "PUT", bear_score / 10 * 0.7
    else:
        return "⚪ NEUTRAL", None, 0.3


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_history(symbol: str, years: int = 3) -> Optional[pd.DataFrame]:
    """Get historical data - tries defeatbeta first, falls back to yfinance"""
    # Try defeatbeta first
    try:
        t = Ticker(symbol)
        df = t.price()
        df = df.rename(columns={"report_date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        needed = ["date", "open", "high", "low", "close", "volume"]
        if all(c in df.columns for c in needed):
            df = df[needed].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            if not df.empty:
                return df.tail(252 * years)
    except Exception:
        pass
    
    # Fallback to yfinance
    try:
        df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        needed = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            return None
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["open", "high", "low", "close"])
        return df.tail(252 * years)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_benchmark(symbol: str = "SPY", years: int = 5) -> Optional[pd.Series]:
    """Get benchmark data for relative strength"""
    hist = get_history(symbol, years=years)
    if hist is None or hist.empty:
        return None
    return hist["close"].rename("bench_close")


# ============================================================================
# OPTIONS CHAIN WITH EXPIRATION DATES
# ============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_option_candidates(symbol: str, option_type: str) -> Optional[pd.DataFrame]:
    """
    Get option candidates with full expiration details
    option_type: "CALL" for bull, "PUT" for bear
    """
    try:
        ticker = yf.Ticker(symbol)
        exps = list(ticker.options)
        if not exps:
            return None
        
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
        today = datetime.now()
        
        all_options = []
        
        # Get options for up to 3 expiration dates
        for exp_date in exps[:3]:
            chain = ticker.option_chain(exp_date)
            
            if option_type == "CALL":
                options = chain.calls.copy()
            else:
                options = chain.puts.copy()
            
            if options is None or options.empty:
                continue
            
            # Parse expiration date
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            days_to_expiry = (exp_dt - today).days
            
            # Calculate moneyness - focus on OTM options for cheaper entry
            if option_type == "CALL":
                options["moneyness"] = options["strike"] / current_price
                options["itm"] = options["strike"] < current_price
                # Focus on OTM calls (strike > price) or slightly ITM
                options = options[options["strike"] >= current_price * 0.95]
            else:
                options["moneyness"] = current_price / options["strike"]
                options["itm"] = options["strike"] > current_price
                # Focus on OTM puts (strike < price) or slightly ITM
                options = options[options["strike"] <= current_price * 1.05]
            
            if options.empty:
                continue
            
            # Calculate spread and liquidity
            options["spread"] = options["ask"] - options["bid"]
            options["mid"] = (options["ask"] + options["bid"]) / 2
            options["spread_pct"] = options["spread"] / options["mid"].replace(0, np.nan)
            
            # Add expiration info
            options["expiration"] = exp_date
            options["days_to_expiry"] = days_to_expiry
            options["expiry_date_str"] = exp_dt.strftime("%b %d, %Y")
            
            # Liquidity score
            options["liq_score"] = (
                options["volume"].fillna(0).clip(upper=5000) / 5000 * 0.35 +
                options["openInterest"].fillna(0).clip(upper=10000) / 10000 * 0.35 +
                (1 - options["spread_pct"].clip(upper=1).fillna(1)) * 0.30
            )
            
            options["option_type"] = option_type
            
            all_options.append(options)
        
        if not all_options:
            return None
        
        result = pd.concat(all_options, ignore_index=True)
        result = result.sort_values(["liq_score", "volume", "openInterest"], ascending=False)
        
        # Select top 10 by liquidity
        return result.head(10)[[
            "contractSymbol", "expiration", "expiry_date_str", "days_to_expiry",
            "strike", "option_type", "bid", "ask", "mid", "spread",
            "volume", "openInterest", "impliedVolatility", "liq_score"
        ]]
        
    except Exception as e:
        return None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["high"] - out[["open", "close"]].max(axis=1)) / rng
    out["lower_wick_pct"] = (out[["open", "close"]].min(axis=1) - out["low"]) / rng
    out["body_pct"] = (out["close"] - out["open"]).abs() / rng
    out["candle_close_loc"] = (out["close"] - out["low"]) / rng
    out["failed_gap"] = ((out["open"] > out["close"].shift(1)) & (out["close"] < out["open"]))
    out["outside_reversal"] = ((out["high"] > out["high"].shift(1)) & (out["close"] < out["high"].shift(1)))
    out["candle_score"] = 50.0
    out["candle_score"] += out["upper_wick_pct"].fillna(0) * 25
    out["candle_score"] -= out["lower_wick_pct"].fillna(0) * 10
    out["candle_score"] += np.where(out["close"] < out["open"], out["body_pct"].fillna(0) * 10, -out["body_pct"].fillna(0) * 15)
    out["candle_score"] += out["failed_gap"].astype(int) * 12
    out["candle_score"] += out["outside_reversal"].astype(int) * 10
    out["candle_score"] += (0.5 - out["candle_close_loc"].fillna(0.5)) * 20
    out["candle_score"] = out["candle_score"].clip(0, 100)
    return out


def build_features(df: pd.DataFrame, bench: Optional[pd.Series]) -> pd.DataFrame:
    out = df.copy()
    
    # TSI calculations
    out["TSI_424"], out["TSI_424_sig"] = tsi(out["close"], 4, 2, 4)
    out["TSI_747"], out["TSI_747_sig"] = tsi(out["close"], 7, 4, 7)
    out["TSI_1377"], out["TSI_1377_sig"] = tsi(out["close"], 13, 7, 7)
    out["TSI_25137"], out["TSI_25137_sig"] = tsi(out["close"], 25, 13, 7)

    for col in ["TSI_424", "TSI_747", "TSI_1377", "TSI_25137"]:
        out[f"{col}_minus_sig"] = out[col] - out[f"{col}_sig"]
        out[f"{col}_slope_1"] = out[col].diff(1)
        out[f"{col}_slope_3"] = out[col].diff(3)
    
    # TSI Kiss patterns
    for tsi_col, sig_col in [("TSI_424", "TSI_424_sig"), ("TSI_747", "TSI_747_sig"), 
                              ("TSI_1377", "TSI_1377_sig"), ("TSI_25137", "TSI_25137_sig")]:
        distance = abs(out[tsi_col] - out[sig_col])
        kissing = distance < 1.5
        was_above = out[tsi_col].shift(1) > out[sig_col].shift(1)
        was_below = out[tsi_col].shift(1) < out[sig_col].shift(1)
        out[f"{tsi_col}_bull_kiss"] = (was_above & kissing & (out[tsi_col] < out[tsi_col].shift(1))).astype(int)
        out[f"{tsi_col}_bear_kiss"] = (was_below & kissing & (out[tsi_col] > out[tsi_col].shift(1))).astype(int)

    out["CCI15"] = cci(out, 15)
    out["CCI20"] = cci(out, 20)
    out["CCI15_days_fading"] = count_consecutive(out["CCI15"].diff() < 0)
    out["CCI20_days_fading"] = count_consecutive(out["CCI20"].diff() < 0)
    
    out["BBP"] = bbpct(out["close"])
    out["BBP_delta_1"] = out["BBP"].diff(1)
    out["BBP_days_falling"] = count_consecutive(out["BBP"].diff() < 0)
    
    out["RSI14"] = rsi(out["close"], 14)
    out["RSI14_slope_3"] = out["RSI14"].diff(3)

    out["VWAP"] = rolling_vwap(out, 20)
    out["EXT_VWAP"] = (out["close"] - out["VWAP"]) / out["VWAP"]
    out["SMA10"] = out["close"].rolling(10).mean()
    out["SMA20"] = out["close"].rolling(20).mean()
    out["EXT_SMA10"] = (out["close"] - out["SMA10"]) / out["SMA10"]
    out["EXT_SMA20"] = (out["close"] - out["SMA20"]) / out["SMA20"]
    out["ADX14"] = adx(out, 14)
    out["ADX14_slope_3"] = out["ADX14"].diff(3)
    out = compute_candle_features(out)

    out["pinned_424"] = count_consecutive((out["TSI_424"] >= 95) & (out["TSI_424"].diff().abs() <= 1.0))
    out["pinned_747"] = count_consecutive((out["TSI_747"] >= 85) & (out["TSI_747"].diff().abs() <= 1.0))
    out["pinned_1377"] = count_consecutive((out["TSI_1377"] >= 55) & (out["TSI_1377"].diff().abs() <= 1.0))
    out["pinned_25137"] = count_consecutive((out["TSI_25137"] >= 20) & (out["TSI_25137"].diff().abs() <= 0.6))

    if bench is not None and not bench.empty:
        aligned = bench.reindex(out.index).ffill()
        out["bench_close"] = aligned
        out["RS_LINE"] = out["close"] / out["bench_close"].replace(0, np.nan)
        out["RS_SLOPE_5"] = out["RS_LINE"].diff(5) / out["RS_LINE"].shift(5)
        out["RS_SLOPE_10"] = out["RS_LINE"].diff(10) / out["RS_LINE"].shift(10)
    else:
        out["RS_LINE"] = 1.0
        out["RS_SLOPE_5"] = 0.0
        out["RS_SLOPE_10"] = 0.0

    for tcol, pcol, scol, codecol in [
        ("TSI_424", "pinned_424", "state_424", "state_424_code"),
        ("TSI_747", "pinned_747", "state_747", "state_747_code"),
        ("TSI_1377", "pinned_1377", "state_1377", "state_1377_code"),
        ("TSI_25137", "pinned_25137", "state_25137", "state_25137_code"),
    ]:
        state_vals = []
        for _, row in out.iterrows():
            state_vals.append(classify_tsi_state(row[tcol], row[f"{tcol}_sig"], row[f"{tcol}_slope_1"], int(row[pcol]) if pd.notna(row[pcol]) else 0))
        out[scol] = state_vals
        out[codecol] = out[scol].map(STATE_CODE_MAP)

    # Bull and bear kiss scores
    out["bull_kiss_score"] = (
        out["TSI_424_bull_kiss"] + out["TSI_747_bull_kiss"] + 
        out["TSI_1377_bull_kiss"] + out["TSI_25137_bull_kiss"]
    )
    out["is_bull_kiss"] = (out["bull_kiss_score"] >= 2).astype(int)

    price_higher = (out["close"] >= out["close"].shift(1) * 0.998)
    tsi_747_fail = out["TSI_747"] <= out["TSI_747"].shift(1) + 0.3
    tsi_1377_supportive = out["TSI_1377"] > 45
    tsi_25137_flat = out["TSI_25137_slope_3"] <= 1.0
    cci_roll = (out["CCI15_days_fading"] >= 2) | ((out["CCI15"] > 100) & (out["CCI15"].diff() < 0))
    bb_stall = (out["BBP"] > 0.95) & (out["BBP_days_falling"] >= 1)
    candle_ok = out["candle_score"] >= 55
    out["bear_kiss_score"] = (
        price_higher.astype(int) * 2.0 +
        tsi_747_fail.astype(int) * 3.0 +
        tsi_1377_supportive.astype(int) * 1.0 +
        tsi_25137_flat.astype(int) * 1.5 +
        cci_roll.astype(int) * 2.5 +
        bb_stall.astype(int) * 2.0 +
        candle_ok.astype(int) * 2.0
    )
    out["is_bear_kiss"] = (out["bear_kiss_score"] >= 8.0).astype(int)
    out["is_hot"] = ((out["TSI_424"] > 95) & (out["TSI_747"] > 70) & (out["BBP"] > 0.95)).astype(int)
    out["is_ripping"] = ((out["TSI_424"] > 95) & (out["TSI_747_slope_1"] > 0.8) & (out["RS_SLOPE_5"].fillna(0) > 0)).astype(int)
    out["is_fading_stack"] = ((out["CCI15_days_fading"] >= 3) & (out["TSI_424_slope_1"] < 0) & (out["TSI_747_slope_1"] <= 0) & (out["BBP_days_falling"] >= 2)).astype(int)
    
    return out


def classify_tsi_state(tsi_val: float, sig_val: float, slope1: float, pinned_bars: int) -> str:
    if pd.isna(tsi_val) or pd.isna(sig_val):
        return "below_zero"
    if tsi_val < 0:
        return "below_zero"
    if pinned_bars >= 3 and abs(slope1) <= 0.5 and tsi_val >= 95:
        return "pinned"
    if tsi_val >= 90 and abs(slope1) <= 1.0:
        return "near_pinned"
    if tsi_val > 70 and slope1 <= 0 and tsi_val <= sig_val + 2:
        return "bear_kiss"
    if tsi_val > sig_val and tsi_val > 0 and slope1 > 0:
        return "bull_continuation"
    if tsi_val > sig_val and tsi_val <= 40:
        return "bull_repair"
    if tsi_val > 0 and slope1 < 0:
        return "rolling"
    return "bull_repair"


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["close"].shift(-1) / out["close"] - 1
    out["ret2"] = out["close"].shift(-2) / out["close"] - 1
    out["ret5"] = out["close"].shift(-5) / out["close"] - 1
    out["dip1"] = (out["ret1"] < 0).astype(float)
    out["dip2"] = (out["ret2"] < -0.005).astype(float)
    out["dip5"] = (out["ret5"] < -0.01).astype(float)
    return out


def feature_store_path(symbol: str, years: int, benchmark_symbol: str) -> Path:
    key = hashlib.md5(f"{symbol}|{years}|{benchmark_symbol}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"feature_store_{symbol}_{years}_{benchmark_symbol}_{key}.parquet"


def load_or_build_feature_store(symbol: str, years: int, benchmark_symbol: str) -> Optional[pd.DataFrame]:
    path = feature_store_path(symbol, years, benchmark_symbol)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    df = get_history(symbol, years=years)
    if df is None or df.empty:
        return None
    bench = get_benchmark(benchmark_symbol, years=years)
    feat = build_features(df, bench)
    feat = add_returns(feat)
    for col in ["RS_LINE", "RS_SLOPE_5", "RS_SLOPE_10"]:
        if col not in feat.columns:
            feat[col] = 0.0 if col != "RS_LINE" else 1.0
    feat["RS_LINE"] = feat["RS_LINE"].fillna(1.0)
    feat["RS_SLOPE_5"] = feat["RS_SLOPE_5"].fillna(0.0)
    feat["RS_SLOPE_10"] = feat["RS_SLOPE_10"].fillna(0.0)
    try:
        feat.to_parquet(path)
    except Exception:
        pass
    return feat


def find_analogs(df: pd.DataFrame, n: int = 30, exclusion_gap: int = 20) -> pd.DataFrame:
    working = df.copy()
    fill_defaults = {
        "RS_LINE": 1.0,
        "RS_SLOPE_5": 0.0,
        "RS_SLOPE_10": 0.0,
        "ADX14": 0.0,
        "ADX14_slope_3": 0.0,
    }
    for col, default in fill_defaults.items():
        if col in working.columns:
            working[col] = working[col].fillna(default)
    base = working.dropna(subset=FEATURE_COLS + ["ret1", "ret2", "ret5"]).copy()
    if len(base) < 100:
        return pd.DataFrame()
    current = base.iloc[-1]
    pool = base.iloc[:-exclusion_gap].copy()
    if pool.empty:
        return pd.DataFrame()
    X = pool[FEATURE_COLS]
    cur = current[FEATURE_COLS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cur_scaled = scaler.transform(pd.DataFrame([cur], columns=FEATURE_COLS))
    dist = cdist(cur_scaled, X_scaled)[0]
    pool["distance"] = dist
    pool["state_penalty"] = (
        (pool["state_424_code"] != current["state_424_code"]).astype(float) * 0.15 +
        (pool["state_747_code"] != current["state_747_code"]).astype(float) * 0.35 +
        (pool["state_1377_code"] != current["state_1377_code"]).astype(float) * 0.20 +
        (pool["state_25137_code"] != current["state_25137_code"]).astype(float) * 0.10 +
        (pool["is_bear_kiss"] != current["is_bear_kiss"]).astype(float) * 0.20
    )
    pool["total_distance"] = pool["distance"] + pool["state_penalty"]
    analogs = pool.nsmallest(n, "total_distance").copy()
    analogs["similarity"] = 1 / (1 + analogs["total_distance"])
    return analogs


def weighted_stats(analogs: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if analogs.empty:
        return {}
    w = analogs["similarity"].clip(lower=1e-9)
    stats: Dict[str, Dict[str, float]] = {}
    for ret_col, dip_col in [("ret1", "dip1"), ("ret2", "dip2"), ("ret5", "dip5")]:
        r = analogs[ret_col]
        d = analogs[dip_col]
        stats[ret_col] = {
            "prob": float(np.average(d, weights=w)),
            "mean": float(np.average(r, weights=w)),
            "median": float(r.median()),
            "p10": float(r.quantile(0.10)),
            "p90": float(r.quantile(0.90)),
        }
    return stats


def confidence_score(analogs: pd.DataFrame, current: pd.Series, stats: Dict[str, Dict[str, float]]) -> float:
    if analogs.empty:
        return 0.0
    size_score = min(len(analogs) / 30.0, 1.0)
    closeness = float(np.clip(1 / (1 + analogs["total_distance"].mean()), 0, 1))
    agreement = 0.0
    if current.get("is_hot", 0):
        agreement += 0.12
    if current.get("is_bear_kiss", 0):
        agreement += 0.28
    if current.get("bear_kiss_score", 0) >= 10:
        agreement += 0.12
    if current.get("candle_score", 0) >= 68:
        agreement += 0.16
    if current.get("CCI15_days_fading", 0) >= 2:
        agreement += 0.10
    if current.get("BBP_days_falling", 0) >= 1:
        agreement += 0.08
    if current.get("state_747") in {"bear_kiss", "pinned", "near_pinned", "rolling"}:
        agreement += 0.08
    if current.get("RS_SLOPE_5", 0) < 0:
        agreement += 0.06
    outcome_consistency = 0.0
    if stats:
        probs = [stats["ret1"]["prob"], stats["ret2"]["prob"], stats["ret5"]["prob"]]
        outcome_consistency = max(0.0, 1 - float(np.std(probs)) * 2)
    conf = 100 * (0.20 * size_score + 0.35 * closeness + 0.25 * agreement + 0.20 * outcome_consistency)
    return float(np.clip(conf, 0, 100))


def setup_label(row: pd.Series) -> str:
    if row.get("is_bear_kiss", 0) and row.get("candle_score", 0) >= 68 and row.get("TSI_747", 0) > 70:
        return "🔥 Diamond"
    if row.get("bear_kiss_score", 0) >= 8 and row.get("TSI_747", 0) > 70:
        return "🟠 Near Diamond"
    if row.get("pinned_424", 0) >= 2 or row.get("state_747") in {"pinned", "near_pinned"}:
        return "🟡 Pinned Extreme"
    if row.get("is_ripping", 0):
        return "🚀 Ripping Extreme"
    return "⚪ Watch"


def why_text(row: pd.Series) -> str:
    reasons: List[str] = []
    if row.get("pinned_424", 0) >= 2:
        reasons.append(f"{int(row['pinned_424'])}x 4,2,4 pinned")
    if row.get("state_747") == "bear_kiss":
        reasons.append("7,4,7 bear kiss")
    if row.get("CCI15_days_fading", 0) >= 2:
        reasons.append(f"CCI15 fading {int(row['CCI15_days_fading'])}d")
    if row.get("BBP_days_falling", 0) >= 1 and row.get("BBP", 0) > 0.95:
        reasons.append("BB% stalled high")
    if row.get("candle_score", 0) >= 68:
        reasons.append("good rejection candle")
    if row.get("RS_SLOPE_5", 0) < 0:
        reasons.append("RS fading")
    if row.get("TSI_25137_slope_3", 0) <= 0.5:
        reasons.append("25,13,7 flattening")
    return ", ".join(reasons[:6]) if reasons else "No stacked trigger yet"


# ============================================================================
# MAIN UI
# ============================================================================

st.title("🔍 Diamond Scanner v4 - Bull/Bear with File Upload")
st.caption("Upload CSV with symbols | Automatic liquidity filtering | Bull/Bear signals | Options with expiration dates")

with st.sidebar:
    st.header("📁 Upload Ticker File")
    uploaded_file = st.file_uploader("CSV with 'Symbol' column", type=["csv"])
    
    st.header("⚙️ Scan Settings")
    years = st.selectbox("Years of historical data", [2, 3, 5], index=1)
    analog_count = st.slider("Number of analogs", 10, 60, 30, 5)
    benchmark_symbol = st.selectbox("Relative strength benchmark", ["SPY", "QQQ", "IWM"], index=0)
    
    st.header("💧 Liquidity Filters")
    st.caption(f"Min Avg Volume: {LIQUIDITY_CONFIG['min_avg_volume']:,}")
    st.caption(f"Min Price: ${LIQUIDITY_CONFIG['min_price']}")
    st.caption(f"Min Market Cap: ${LIQUIDITY_CONFIG['min_market_cap_billions']}B")
    st.caption(f"Exclude ETFs: {LIQUIDITY_CONFIG['exclude_etf']}")
    
    rebuild_cache = st.checkbox("Rebuild cached feature stores", value=False)
    run_scan = st.button("🚀 Run Scan", type="primary")

# Load symbols from uploaded file
symbols = []
uploaded_df = None

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if 'Symbol' in uploaded_df.columns:
        symbols = uploaded_df['Symbol'].tolist()
        st.success(f"Loaded {len(symbols)} symbols from file")
    else:
        st.error("CSV must contain a 'Symbol' column")
        st.stop()
else:
    # Default symbols if no file uploaded
    default_symbols = ["QQQ", "SMH", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AAPL"]
    symbols = default_symbols
    st.info(f"No file uploaded. Using default symbols: {', '.join(default_symbols)}")

if run_scan:
    if rebuild_cache:
        for symbol in symbols:
            path = feature_store_path(symbol, years, benchmark_symbol)
            if path.exists():
                path.unlink(missing_ok=True)
    
    # First, check liquidity for all symbols
    st.subheader("💧 Liquidity Screening")
    liquidity_results = []
    
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        passed, metrics = check_liquidity(symbol)
        metrics['passed'] = passed
        liquidity_results.append(metrics)
        progress.progress((i + 1) / len(symbols))
    progress.empty()
    
    liquidity_df = pd.DataFrame(liquidity_results)
    passed_symbols = liquidity_df[liquidity_df['passed'] == True]['symbol'].tolist()
    
    st.write(f"**Liquidity Screen Results:** {len(passed_symbols)} / {len(symbols)} passed")
    
    if not passed_symbols:
        st.error("No symbols passed liquidity filters. Adjust filters or check your symbols.")
        st.dataframe(liquidity_df, width='stretch')
        st.stop()
    
    with st.expander("View liquidity details"):
        st.dataframe(liquidity_df, width='stretch')
    
    # Now scan only liquid symbols
    st.subheader("📊 Scanning Liquid Symbols")
    
    rows = []
    history_map = {}
    analog_map = {}
    
    progress = st.progress(0)
    debug_rows = []
    
    for i, symbol in enumerate(passed_symbols):
        feat = load_or_build_feature_store(symbol, years, benchmark_symbol)
        if feat is None or feat.empty:
            debug_rows.append({"symbol": symbol, "status": "no history returned"})
            continue
        
        # Get live data for current signal
        live_df = yf.download(symbol, period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        if live_df is None or live_df.empty:
            debug_rows.append({"symbol": symbol, "status": "no live data"})
            continue
        
        if isinstance(live_df.columns, pd.MultiIndex):
            live_df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in live_df.columns]
        else:
            live_df.columns = [str(c).lower() for c in live_df.columns]
        
        live_bench = get_benchmark(benchmark_symbol, years=years)
        live_feat = build_features(live_df, live_bench)
        live_feat = add_returns(live_feat)
        
        if live_feat.empty:
            debug_rows.append({"symbol": symbol, "status": "no live features"})
            continue
        
        current = live_feat.iloc[-1].copy()
        
        # Get analog matches from historical data
        analogs = find_analogs(feat, n=analog_count)
        
        if analogs.empty:
            debug_rows.append({"symbol": symbol, "status": "no analogs", "hist_rows": len(feat)})
            continue
        
        stats = weighted_stats(analogs)
        conf = confidence_score(analogs, current, stats)
        
        # Calculate bull/bear scores
        bull_score, bull_reasons = calculate_bull_score(current)
        bear_score, bear_reasons = calculate_bear_score(current)
        signal_type, option_type, signal_conf = get_signal_type(bull_score, bear_score)
        
        rows.append({
            "symbol": symbol,
            "Signal": signal_type,
            "Bull Score": bull_score,
            "Bear Score": bear_score,
            "Confidence": conf,
            "TSI747": float(current.get("TSI_747", 0)),
            "RSI14": float(current.get("RSI14", 0)),
            "CCI15": float(current.get("CCI15", 0)),
            "BBP": float(current.get("BBP", 0)),
            "Candle": float(current.get("candle_score", 0)),
            "Bull Kiss": int(current.get("bull_kiss_score", 0)),
            "Bear Kiss": int(current.get("bear_kiss_score", 0)),
            "State": setup_label(current),
            "Why": why_text(current)[:100],
        })
        
        history_map[symbol] = feat
        analog_map[symbol] = analogs
        progress.progress((i + 1) / len(passed_symbols))
    
    progress.empty()
    
    if not rows:
        st.error("No valid symbols were processed.")
        if debug_rows:
            st.subheader("Debug Info")
            st.dataframe(pd.DataFrame(debug_rows), width='stretch')
        st.stop()
    
    results = pd.DataFrame(rows).sort_values(["Confidence", "Bull Score"], ascending=[False, False])
    
    st.subheader("🏆 Ranked Setups")
    st.dataframe(results, width='stretch', use_container_width=True)
    
    if debug_rows:
        with st.expander("Skipped symbols / debug details"):
            st.dataframe(pd.DataFrame(debug_rows), width='stretch')
    
    # Detailed inspection for selected symbol
    st.subheader("🔬 Detailed Analysis")
    selected = st.selectbox("Select symbol to inspect", results["symbol"].tolist())
    
    if selected:
        selected_row = results[results["symbol"] == selected].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Signal", selected_row["Signal"])
        col2.metric("Bull Score", f"{selected_row['Bull Score']:.1f}")
        col3.metric("Bear Score", f"{selected_row['Bear Score']:.1f}")
        col4.metric("Confidence", f"{selected_row['Confidence']:.0f}%")
        
        # Show bull/bear reasons
        # Need to recalc for display
        current_hist = history_map[selected]
        live_df = yf.download(selected, period="3mo", interval="1d", auto_adjust=False, progress=False)
        if isinstance(live_df.columns, pd.MultiIndex):
            live_df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in live_df.columns]
        else:
            live_df.columns = [str(c).lower() for c in live_df.columns]
        live_bench = get_benchmark(benchmark_symbol, years=years)
        live_feat = build_features(live_df, live_bench)
        current = live_feat.iloc[-1] if not live_feat.empty else None
        
        if current is not None:
            bull_score, bull_reasons = calculate_bull_score(current)
            bear_score, bear_reasons = calculate_bear_score(current)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🟢 Bullish Signals**")
                if bull_reasons:
                    for r in bull_reasons:
                        st.write(f"  ✅ {r}")
                else:
                    st.write("No bullish signals")
            
            with col2:
                st.markdown("**🔴 Bearish Signals**")
                if bear_reasons:
                    for r in bear_reasons:
                        st.write(f"  ⚠️ {r}")
                else:
                    st.write("No bearish signals")
        
        # Options chain
        signal_type = selected_row["Signal"]
        if "BULL" in signal_type:
            option_type = "CALL"
            st.subheader(f"📈 Bull Signal - Call Option Candidates for {selected}")
        elif "BEAR" in signal_type:
            option_type = "PUT"
            st.subheader(f"📉 Bear Signal - Put Option Candidates for {selected}")
        else:
            option_type = None
            st.subheader(f"⚪ Neutral Signal - Options for {selected} (both sides)")
        
        if option_type:
            opt = get_option_candidates(selected, option_type)
            if opt is not None:
                st.dataframe(opt, width='stretch', use_container_width=True)
            else:
                st.info(f"No {option_type} options available for {selected}")
        
        # Also show opposite side options
        other_type = "PUT" if option_type == "CALL" else "CALL"
        opt_other = get_option_candidates(selected, other_type)
        if opt_other is not None:
            with st.expander(f"Show {other_type} options (opposite side)"):
                st.dataframe(opt_other, width='stretch', use_container_width=True)
        
        # Show analog matches
        st.subheader(f"📊 Historical Analog Matches for {selected}")
        analogs = analog_map[selected]
        if analogs is not None and not analogs.empty:
            analog_show = analogs[["close", "ret1", "ret2", "ret5", "similarity", "state_747", "bear_kiss_score", "candle_score"]].tail(15)
            st.dataframe(analog_show, width='stretch', use_container_width=True)
        else:
            st.info("No analog matches found")

else:
    st.info("👈 Upload a CSV file with a 'Symbol' column and click 'Run Scan'")
    
    # Show expected file format
    with st.expander("📄 Expected CSV Format"):
        st.code("""
Symbol,Name,Exchange,Sector,Industry,SCTR,Universe,Close,Volume
AAPL,Apple Inc.,NASD,Technology,Consumer Electronics,85.2,lrg,175.50,50000000
MSFT,Microsoft Corp.,NASD,Technology,Software,92.1,lrg,420.75,22000000
NVDA,NVIDIA Corp.,NASD,Technology,Semiconductors,98.5,lrg,950.25,45000000
        """)
        st.caption("Only the 'Symbol' column is required. Other columns are optional.")
