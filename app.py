import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

try:
    from defeatbeta_api import Ticker as DefeatTicker
except Exception:
    try:
        from defeatbeta_api.data.ticker import Ticker as DefeatTicker
    except Exception:
        DefeatTicker = None

st.set_page_config(page_title="Stable Market Engine", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Basic indicators
# -----------------------------
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


def centered_pct(series: pd.Series) -> pd.Series:
    return (series.fillna(0.5) - 0.5) * 2


# -----------------------------
# Data fetch
# -----------------------------
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    rename = {str(c): str(c).title() for c in out.columns}
    out = out.rename(columns=rename)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    if len(keep) < 4:
        return pd.DataFrame()
    out = out[keep].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    try:
        out.index = out.index.tz_localize(None)
    except Exception:
        try:
            out.index = out.index.tz_convert(None)
        except Exception:
            pass
    out = out[~out.index.isna()].sort_index()
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_daily(symbol: str, years: int = 10) -> pd.DataFrame:
    periods = [f"{max(years, 5)}y", "10y", "5y"]
    for period in periods:
        for attempt in range(4):
            try:
                df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False, prepost=False)
                df = normalize_ohlcv(df)
                if not df.empty:
                    return df
            except Exception:
                pass
            time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpha_vantage_daily(symbol: str, api_key: str) -> pd.DataFrame:
    api_key = (api_key or "").strip()
    if not api_key:
        return pd.DataFrame()
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
        r = requests.get(url, params=params, timeout=20)
        payload = r.json()
        key = "Time Series (Daily)"
        if key not in payload:
            return pd.DataFrame()
        rows = []
        for ds, item in payload[key].items():
            rows.append(
                {
                    "Date": pd.to_datetime(ds, errors="coerce"),
                    "Open": float(item.get("1. open", np.nan)),
                    "High": float(item.get("2. high", np.nan)),
                    "Low": float(item.get("3. low", np.nan)),
                    "Close": float(item.get("4. close", np.nan)),
                    "Volume": float(item.get("6. volume", np.nan)),
                }
            )
        df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
        return normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_defeatbeta_history(symbol: str, years: int = 5) -> pd.DataFrame:
    if DefeatTicker is None:
        return pd.DataFrame()
    try:
        t = DefeatTicker(symbol)
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
        return normalize_ohlcv(df).tail(int(max(years, 5) * 252 * 1.3))
    except Exception:
        return pd.DataFrame()


def merge_with_recent(base_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty:
        return recent_df.copy()
    if recent_df.empty:
        return base_df.copy()
    cutoff = pd.to_datetime(base_df.index.max()) - pd.Timedelta(days=10)
    recent_only = recent_df[recent_df.index >= cutoff]
    merged = pd.concat([base_df, recent_only])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def fetch_daily_priority(symbol: str, years: int, alpha_key: str) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    meta = {"yahoo_status": "not_checked", "alpha_status": "not_checked", "defeat_status": "not_checked"}

    yahoo_df = fetch_yahoo_daily(symbol, years=max(years, 5))
    if not yahoo_df.empty:
        meta["yahoo_status"] = f"ok:{len(yahoo_df)}"
        meta["daily_rows"] = str(len(yahoo_df))
        return yahoo_df, "yahoo_daily", meta
    meta["yahoo_status"] = "empty_or_rate_limited"

    alpha_df = fetch_alpha_vantage_daily(symbol, alpha_key)
    if not alpha_df.empty:
        meta["alpha_status"] = f"ok:{len(alpha_df)}"
        meta["daily_rows"] = str(len(alpha_df))
        return alpha_df, "alpha_vantage_daily", meta
    meta["alpha_status"] = "empty_or_limited"

    defeat_df = fetch_defeatbeta_history(symbol, years=max(years, 5))
    if not defeat_df.empty:
        meta["defeat_status"] = f"ok:{len(defeat_df)}"
        meta["daily_rows"] = str(len(defeat_df))
        return defeat_df, "defeatbeta_fallback", meta
    meta["defeat_status"] = "empty"
    meta["daily_rows"] = "0"
    return pd.DataFrame(), "none", meta


# -----------------------------
# Feature engineering
# -----------------------------
def add_ultimate_oscillator(out: pd.DataFrame, timeframe_name: str) -> pd.DataFrame:
    spans = {
        "proxy_hourly": (5, 13, 5),
        "proxy_2hour": (6, 18, 5),
        "daily": (8, 21, 7),
        "weekly": (5, 13, 5),
    }

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
    out["uo_pctile"] = rolling_percentile(out["uo"], 120 if timeframe_name in {"proxy_hourly", "proxy_2hour"} else 252)
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
    out["dist_vwap_pct"] = np.nan
    if benchmark_df is not None and not benchmark_df.empty:
        aligned = benchmark_df["Close"].reindex(out.index).ffill()
        out["rs_vs_benchmark"] = out["Close"] / aligned
        out["rs_bench_slope_5"] = slope(out["rs_vs_benchmark"], 5)
    else:
        out["rs_vs_benchmark"] = 1.0
        out["rs_bench_slope_5"] = 0.0
    win = 120 if timeframe_name in {"proxy_hourly", "proxy_2hour"} else 252
    for col in ["rsi_14", "cci_20", "tsi", "pct_b", "atr_stretch", "adx_14", "dist_ema20_pct", "volume_ratio", "dist_vwap_pct"]:
        if col in out.columns:
            out[f"{col}_pctile"] = rolling_percentile(out[col], win)
    out = add_ultimate_oscillator(out, timeframe_name)
    return out


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("W-FRI").agg(agg).dropna(how="any")


def build_proxy_from_daily(df: pd.DataFrame, proxy_mode: str) -> Tuple[pd.DataFrame, str, str]:
    # Daily-based tactical proxy to replace unstable true hourly fetches.
    label = "Hourly Proxy" if proxy_mode == "hourly" else "2-Hour Proxy"
    timeframe_name = "proxy_hourly" if proxy_mode == "hourly" else "proxy_2hour"
    return df.copy(), timeframe_name, label


def compute_distance_to_cross(row: pd.Series, frame: pd.DataFrame) -> Dict[str, float]:
    if row is None or row.empty or frame is None or frame.empty:
        return {"gap": np.nan, "abs_gap": np.nan, "range_pct": np.nan}
    gap = float(row.get("uo", np.nan) - row.get("uo_signal", np.nan))
    abs_gap = abs(gap) if pd.notna(gap) else np.nan
    recent = (frame["uo"] - frame["uo_signal"]).dropna().tail(40)
    if recent.empty:
        range_pct = np.nan
    else:
        denom = max(float(recent.abs().max()), 1e-9)
        range_pct = float(np.clip(abs_gap / denom, 0, 1) * 100) if pd.notna(abs_gap) else np.nan
    return {"gap": gap, "abs_gap": abs_gap, "range_pct": range_pct}


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
    if timeframe in {"proxy_hourly", "proxy_2hour"} and rsi_val > 75 and cci_val > 90 and pct_b > 0.90 and (uo_gap <= 0 or tsi_gap <= 0):
        return "PUT", 78.0, "Proxy overheated and rolling"
    if timeframe in {"proxy_hourly", "proxy_2hour"} and rsi_val > 75 and cci_val > 90 and pct_b > 0.90 and adx_val > 25 and uo_gap > 0:
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


def combine_calls(proxy_call: str, daily_call: str, weekly_call: str) -> Tuple[str, str]:
    score_map = {"CALL": 1.0, "PUT": -1.0, "NEUTRAL": 0.0, "AVOID CHASE": -0.35, "NO DATA": 0.0}
    score = 0.30 * score_map.get(proxy_call, 0.0) + 0.45 * score_map.get(daily_call, 0.0) + 0.25 * score_map.get(weekly_call, 0.0)
    if proxy_call in {"PUT", "AVOID CHASE"} and daily_call == "CALL" and weekly_call == "CALL":
        return "WAIT / PROXY TOO HOT", "Bullish trend, but short-term timing is poor"
    if score >= 0.55:
        return "CALL", "Timeframes supportive"
    if score <= -0.55:
        return "PUT", "Timeframes lean bearish"
    if daily_call == "CALL" and weekly_call == "CALL":
        return "CALL ON PULLBACK", "Higher timeframe trend constructive, but entry needs reset"
    return "NEUTRAL", "Mixed timeframe signals"


def align_asof(index: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    try:
        if index.tz is not None and ts.tzinfo is None:
            return ts.tz_localize(index.tz)
        if index.tz is None and ts.tzinfo is not None:
            return ts.tz_localize(None)
    except Exception:
        pass
    return ts


def slice_asof(df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    end_ts = align_asof(df.index, pd.Timestamp(analysis_date) + pd.Timedelta(hours=23, minutes=59, seconds=59))
    return df.loc[df.index <= end_ts].copy()


def plot_dashboard(symbol: str, proxy_df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, asof_date: pd.Timestamp, proxy_label: str) -> None:
    fig = make_subplots(
        rows=4, cols=1, vertical_spacing=0.06,
        subplot_titles=[f"{symbol} Daily Price (as of {pd.Timestamp(asof_date).date()})", f"{proxy_label} Oscillator", "Daily Ultimate Oscillator", "Weekly Ultimate Oscillator"],
        row_heights=[0.42, 0.19, 0.19, 0.20],
    )
    if not daily_df.empty:
        d = daily_df.tail(220)
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Daily"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["ema_20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["sma_50"], name="SMA50", line=dict(color="blue")), row=1, col=1)
    for row_num, frame in zip([2, 3, 4], [proxy_df.tail(220), daily_df.tail(220), weekly_df.tail(150)]):
        if frame.empty:
            continue
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo"], name=f"UO {row_num}", line=dict(color="red", width=2)), row=row_num, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["uo_signal"], name=f"Signal {row_num}", line=dict(color="black", width=1)), row=row_num, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row_num, col=1)
    fig.update_layout(height=950, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, width='stretch')


# -----------------------------
# App
# -----------------------------
st.title("📈 Stable Market Engine")
st.caption("Daily-first | fast daily proxy instead of true hourly | calendar lookback with as-of values")

with st.sidebar:
    st.header("Inputs")
    manual_symbols = st.text_area("Paste tickers (comma or line separated)", value="SMH, QQQ, INTC", height=110)
    alpha_vantage_key = st.text_input("Alpha Vantage API key (optional)", type="password")

    st.header("Settings")
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "RSP", "IWM"], index=0)
    history_years = st.selectbox("Historical years", [3, 5, 10], index=1)
    analysis_mode = st.radio("Analysis mode", ["Current", "Historical"], index=0)
    default_date = pd.Timestamp.today().date()
    analysis_date = st.date_input("Calendar lookback", value=default_date, disabled=(analysis_mode == "Current"))
    proxy_mode = st.radio("Tactical proxy", ["hourly", "2hour"], index=0, format_func=lambda x: "Hourly Proxy" if x == "hourly" else "2-Hour Proxy")
    run_analysis = st.button("Run Analysis", type="primary", width='stretch')

st.info("This build uses daily data only. The tactical panel is a switchable daily-built proxy, not true intraday bars.")

if not run_analysis:
    st.stop()

symbols = [s.strip().upper() for s in manual_symbols.replace("\n", ",").split(",") if s.strip()]
symbols = list(dict.fromkeys(symbols))
if not symbols:
    st.error("Provide at least one symbol.")
    st.stop()

benchmark_daily_raw, bench_source, bench_meta = fetch_daily_priority(benchmark, history_years, alpha_vantage_key)
benchmark_daily = normalize_ohlcv(benchmark_daily_raw)
if benchmark_daily.empty:
    st.error(f"Could not fetch benchmark {benchmark}.")
    st.stop()

st.caption(f"Benchmark source: {bench_source} | Yahoo={bench_meta.get('yahoo_status')} | Alpha={bench_meta.get('alpha_status')} | Defeat={bench_meta.get('defeat_status')}")

rows: List[Dict[str, object]] = []
detail: Dict[str, Dict[str, object]] = {}
progress = st.progress(0.0)

for idx, symbol in enumerate(symbols):
    progress.progress((idx + 1) / len(symbols))
    daily_raw, source, meta = fetch_daily_priority(symbol, history_years, alpha_vantage_key)
    daily_raw = normalize_ohlcv(daily_raw)
    if daily_raw.empty:
        rows.append({
            "Symbol": symbol,
            "Status": "No data",
            "History Source": source,
            "Yahoo Daily": meta.get("yahoo_status", "n/a"),
            "Alpha Daily": meta.get("alpha_status", "n/a"),
            "DefeatBeta": meta.get("defeat_status", "n/a"),
            "Fetch Detail": f"yahoo={meta.get('yahoo_status','n/a')} | alpha={meta.get('alpha_status','n/a')} | defeat={meta.get('defeat_status','n/a')}",
        })
        continue

    proxy_raw, proxy_timeframe, proxy_label = build_proxy_from_daily(daily_raw, proxy_mode)
    weekly_raw = resample_weekly(daily_raw)

    proxy_df = enrich_price_features(proxy_raw, proxy_timeframe, benchmark_daily)
    daily_df = enrich_price_features(daily_raw, "daily", benchmark_daily)
    weekly_df = enrich_price_features(weekly_raw, "weekly", benchmark_daily)

    asof = pd.Timestamp.today().normalize() if analysis_mode == "Current" else pd.Timestamp(analysis_date)
    proxy_view = slice_asof(proxy_df, asof)
    daily_view = slice_asof(daily_df, asof)
    weekly_view = slice_asof(weekly_df, asof)

    if daily_view.empty:
        rows.append({
            "Symbol": symbol,
            "Status": "No data as of date",
            "History Source": source,
            "Yahoo Daily": meta.get("yahoo_status", "n/a"),
            "Alpha Daily": meta.get("alpha_status", "n/a"),
            "DefeatBeta": meta.get("defeat_status", "n/a"),
            "Fetch Detail": f"asof={asof.date()} | no rows before date",
        })
        continue

    proxy_row = proxy_view.iloc[-1] if not proxy_view.empty else pd.Series(dtype=float)
    daily_row = daily_view.iloc[-1]
    weekly_row = weekly_view.iloc[-1] if not weekly_view.empty else pd.Series(dtype=float)

    proxy_call, proxy_conf, proxy_reason = classify_timeframe_call(proxy_row, proxy_timeframe)
    proxy_cross = compute_distance_to_cross(proxy_row, proxy_view)
    daily_call, daily_conf, daily_reason = classify_timeframe_call(daily_row, "daily")
    weekly_call, weekly_conf, weekly_reason = classify_timeframe_call(weekly_row, "weekly")
    combined_call, combined_reason = combine_calls(proxy_call, daily_call, weekly_call)

    rows.append({
        "Symbol": symbol,
        "Status": "OK",
        "History Source": source,
        "Yahoo Daily": meta.get("yahoo_status", "n/a"),
        "Alpha Daily": meta.get("alpha_status", "n/a"),
        "DefeatBeta": meta.get("defeat_status", "n/a"),
        "As Of": str(daily_row.name.date()),
        "Close": round(float(daily_row.get("Close", np.nan)), 2),
        "Proxy Mode": proxy_label,
        "Proxy Call": proxy_call,
        "Daily Call": daily_call,
        "Weekly Call": weekly_call,
        "Combined": combined_call,
        "Proxy UO": round(float(proxy_row.get("uo", np.nan)), 4) if not proxy_row.empty else np.nan,
        "Proxy Sig": round(float(proxy_row.get("uo_signal", np.nan)), 4) if not proxy_row.empty else np.nan,
        "Proxy %ile": round(float(proxy_row.get("uo_pctile", np.nan)) * 100, 1) if not proxy_row.empty else np.nan,
        "Cross Gap": round(float(proxy_cross.get("gap", np.nan)), 4) if proxy_cross else np.nan,
        "Cross Dist %": round(float(proxy_cross.get("range_pct", np.nan)), 1) if proxy_cross else np.nan,
        "Daily UO": round(float(daily_row.get("uo", np.nan)), 4),
        "Daily Sig": round(float(daily_row.get("uo_signal", np.nan)), 4),
        "Daily %ile": round(float(daily_row.get("uo_pctile", np.nan)) * 100, 1),
        "Weekly UO": round(float(weekly_row.get("uo", np.nan)), 4) if not weekly_row.empty else np.nan,
        "Weekly %ile": round(float(weekly_row.get("uo_pctile", np.nan)) * 100, 1) if not weekly_row.empty else np.nan,
        "Candle Score": round(float(daily_row.get("candle_score", np.nan)), 1),
        "RSI14": round(float(daily_row.get("rsi_14", np.nan)), 1),
        "CCI20": round(float(daily_row.get("cci_20", np.nan)), 1),
        "TSI": round(float(daily_row.get("tsi", np.nan)), 2),
        "Fetch Detail": f"yahoo={meta.get('yahoo_status','n/a')} | alpha={meta.get('alpha_status','n/a')} | defeat={meta.get('defeat_status','n/a')}",
    })

    detail[symbol] = {
        "proxy": proxy_view,
        "daily": daily_view,
        "weekly": weekly_view,
        "proxy_call": (proxy_call, proxy_conf, proxy_reason),
        "proxy_label": proxy_label,
        "proxy_cross": proxy_cross,
        "daily_call": (daily_call, daily_conf, daily_reason),
        "weekly_call": (weekly_call, weekly_conf, weekly_reason),
        "combined": (combined_call, combined_reason),
        "fetch_meta": meta,
        "history_source": source,
        "asof": asof,
    }

progress.empty()
results_df = pd.DataFrame(rows)

st.subheader("Ranked Results")
if results_df.empty:
    st.warning("No results.")
    st.stop()

sort_col = "Daily %ile" if "Daily %ile" in results_df.columns else None
if sort_col:
    results_df = results_df.sort_values(sort_col, ascending=False)
st.dataframe(results_df, width='stretch', hide_index=True)
st.download_button("Download results CSV", results_df.to_csv(index=False).encode("utf-8"), "stable_market_engine_results.csv", "text/csv")

valid_symbols = results_df.loc[results_df["Status"] == "OK", "Symbol"].tolist()
if not valid_symbols:
    st.stop()

st.subheader("Detailed Analysis")
selected = st.selectbox("Select symbol", valid_symbols)
item = detail[selected]
proxy_call, proxy_conf, proxy_reason = item["proxy_call"]
proxy_label = item.get("proxy_label", "Proxy")
proxy_cross = item.get("proxy_cross", {})
daily_call, daily_conf, daily_reason = item["daily_call"]
weekly_call, weekly_conf, weekly_reason = item["weekly_call"]
combined_call, combined_reason = item["combined"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(proxy_label, proxy_call)
c2.metric("Daily", daily_call)
c3.metric("Weekly", weekly_call)
c4.metric("Combined", combined_call)
st.markdown(f"**{proxy_label} reason:** {proxy_reason}")
st.markdown(f"**Daily reason:** {daily_reason}")
st.markdown(f"**Weekly reason:** {weekly_reason}")
st.markdown(f"**Combined read:** {combined_reason}")
st.caption(f"History source: {item['history_source']} | Fetch detail: {item['fetch_meta']}")

cd1, cd2, cd3 = st.columns(3)
cd1.metric("Distance to cross", f"{proxy_cross.get('abs_gap', np.nan):.4f}" if pd.notna(proxy_cross.get('abs_gap', np.nan)) else "n/a")
cd2.metric("Cross gap sign", "Above signal" if proxy_cross.get("gap", np.nan) >= 0 else "Below signal") if pd.notna(proxy_cross.get("gap", np.nan)) else cd2.metric("Cross gap sign", "n/a")
cd3.metric("Cross distance %", f"{proxy_cross.get('range_pct', np.nan):.1f}%" if pd.notna(proxy_cross.get('range_pct', np.nan)) else "n/a")

proxy_last = item["proxy"].iloc[-1] if not item["proxy"].empty else pd.Series(dtype=float)
daily_last = item["daily"].iloc[-1] if not item["daily"].empty else pd.Series(dtype=float)
weekly_last = item["weekly"].iloc[-1] if not item["weekly"].empty else pd.Series(dtype=float)

st.markdown("### As-of values")
asof_table = pd.DataFrame(
    [
        {
            "Frame": proxy_label,
            "Date": str(proxy_last.name.date()) if not proxy_last.empty else "n/a",
            "Price": round(float(proxy_last.get("Close", np.nan)), 2) if not proxy_last.empty else np.nan,
            "UO": round(float(proxy_last.get("uo", np.nan)), 4) if not proxy_last.empty else np.nan,
            "Signal": round(float(proxy_last.get("uo_signal", np.nan)), 4) if not proxy_last.empty else np.nan,
            "Percentile": round(float(proxy_last.get("uo_pctile", np.nan)) * 100, 1) if not proxy_last.empty else np.nan,
            "Candle Score": round(float(proxy_last.get("candle_score", np.nan)), 1) if not proxy_last.empty else np.nan,
            "Cross Gap": round(float(proxy_cross.get("gap", np.nan)), 4) if pd.notna(proxy_cross.get("gap", np.nan)) else np.nan,
            "Cross Dist %": round(float(proxy_cross.get("range_pct", np.nan)), 1) if pd.notna(proxy_cross.get("range_pct", np.nan)) else np.nan,
        },
        {
            "Frame": "Daily",
            "Date": str(daily_last.name.date()) if not daily_last.empty else "n/a",
            "Price": round(float(daily_last.get("Close", np.nan)), 2) if not daily_last.empty else np.nan,
            "UO": round(float(daily_last.get("uo", np.nan)), 4) if not daily_last.empty else np.nan,
            "Signal": round(float(daily_last.get("uo_signal", np.nan)), 4) if not daily_last.empty else np.nan,
            "Percentile": round(float(daily_last.get("uo_pctile", np.nan)) * 100, 1) if not daily_last.empty else np.nan,
            "Candle Score": round(float(daily_last.get("candle_score", np.nan)), 1) if not daily_last.empty else np.nan,
        },
        {
            "Frame": "Weekly",
            "Date": str(weekly_last.name.date()) if not weekly_last.empty else "n/a",
            "Price": round(float(weekly_last.get("Close", np.nan)), 2) if not weekly_last.empty else np.nan,
            "UO": round(float(weekly_last.get("uo", np.nan)), 4) if not weekly_last.empty else np.nan,
            "Signal": round(float(weekly_last.get("uo_signal", np.nan)), 4) if not weekly_last.empty else np.nan,
            "Percentile": round(float(weekly_last.get("uo_pctile", np.nan)) * 100, 1) if not weekly_last.empty else np.nan,
            "Candle Score": round(float(weekly_last.get("candle_score", np.nan)), 1) if not weekly_last.empty else np.nan,
        },
    ]
)
st.dataframe(asof_table, width='stretch', hide_index=True)

plot_dashboard(selected, item["proxy"], item["daily"], item["weekly"], item["asof"], proxy_label)
