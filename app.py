
import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache_store"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Predictive Trading Analysis Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Utility / indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
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

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def tsi(series: pd.Series, long: int = 25, short: int = 13, signal: int = 7):
    delta = series.diff()
    abs_delta = delta.abs()
    double_smoothed = ema(ema(delta, long), short)
    double_abs = ema(ema(abs_delta, long), short)
    tsi_line = 100 * double_smoothed / double_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal)
    return tsi_line, signal_line

def bollinger_pct_b(series: pd.Series, window: int = 20, num_std: float = 2.0):
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

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_val = tr.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()

def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    def pct_rank(x: np.ndarray) -> float:
        if np.all(np.isnan(x)):
            return np.nan
        s = pd.Series(x)
        return s.rank(pct=True).iloc[-1]
    return series.rolling(window, min_periods=max(20, window // 5)).apply(pct_rank, raw=True)

def zscore(series: pd.Series, window: int = 63) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd.replace(0, np.nan)

def slope(series: pd.Series, bars: int = 3) -> pd.Series:
    return series.diff(bars) / bars

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    cum_pv = pv.groupby(df.index.date).cumsum()
    cum_v = df["Volume"].groupby(df.index.date).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)

def safe_last(series: pd.Series):
    return None if series.empty else series.iloc[-1]

# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_prices(
    ticker: str,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
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

    df = yf.download(**kwargs)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(how="all")
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_defeat_fallback(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Optional fallback adapter. Set DEFEAT_API_URL in Streamlit secrets or environment.
    Expected JSON list of records with Date/Open/High/Low/Close/Volume fields.
    """
    api_url = st.secrets.get("DEFEAT_API_URL", None) if hasattr(st, "secrets") else None
    if not api_url:
        return pd.DataFrame()

    try:
        params = {"ticker": ticker}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        r = requests.get(api_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
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

# -----------------------------
# Feature engine
# -----------------------------
def enrich_price_features(df: pd.DataFrame, timeframe_name: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ema_10"] = ema(out["Close"], 10)
    out["ema_20"] = ema(out["Close"], 20)
    out["sma_20"] = sma(out["Close"], 20)
    out["sma_50"] = sma(out["Close"], 50)
    out["atr_14"] = atr(out, 14)
    out["rsi_14"] = rsi(out["Close"], 14)
    out["rsi_5"] = rsi(out["Close"], 5)
    out["cci_20"] = cci(out, 20)
    out["cci_15"] = cci(out, 15)
    out["tsi"], out["tsi_signal"] = tsi(out["Close"], 25, 13, 7)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["Close"], 12, 26, 9)
    out["pct_b"] = bollinger_pct_b(out["Close"], 20, 2)
    out["adx_14"] = adx(out, 14)
    out["dist_ema10_pct"] = (out["Close"] / out["ema_10"]) - 1
    out["dist_ema20_pct"] = (out["Close"] / out["ema_20"]) - 1
    out["dist_sma50_pct"] = (out["Close"] / out["sma_50"]) - 1
    out["atr_stretch"] = (out["Close"] - out["ema_20"]) / out["atr_14"].replace(0, np.nan)
    out["z_20"] = zscore(out["Close"], 20)
    out["z_63"] = zscore(out["Close"], 63)
    out["rvol_20"] = out["Volume"] / out["Volume"].rolling(20).mean()
    out["close_in_range"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick_pct"] = (out["High"] - out[["Close", "Open"]].max(axis=1)) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["body_pct"] = (out["Close"] - out["Open"]).abs() / (out["High"] - out["Low"]).replace(0, np.nan)

    if timeframe_name == "hourly":
        out["vwap"] = compute_vwap(out)
        out["dist_vwap_pct"] = (out["Close"] / out["vwap"]) - 1
        out["hours_from_10bar_low"] = out["Low"].rolling(10).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True)
        out["intraday_atr_pct"] = out["atr_14"] / out["Close"]
    else:
        out["dist_vwap_pct"] = np.nan
        out["hours_from_10bar_low"] = np.nan
        out["intraday_atr_pct"] = np.nan

    for col in [
        "rsi_14", "cci_20", "tsi", "pct_b", "atr_stretch",
        "dist_ema10_pct", "dist_ema20_pct", "adx_14", "rvol_20"
    ]:
        out[f"{col}_pctile"] = rolling_percentile(out[col], 252 if timeframe_name != "hourly" else 120)

    out["rsi_slope_3"] = slope(out["rsi_14"], 3)
    out["cci_slope_3"] = slope(out["cci_20"], 3)
    out["tsi_slope_3"] = slope(out["tsi"], 3)
    out["price_slope_3"] = slope(out["Close"], 3)
    out["higher_high"] = (out["High"] > out["High"].shift(1)).astype(float)
    out["lower_high"] = (out["High"] < out["High"].shift(1)).astype(float)
    return out

def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [1, 2, 3, 5]:
        out[f"fwd_ret_{n}"] = out["Close"].shift(-n) / out["Close"] - 1
    out["fwd_low_1"] = out["Low"].shift(-1) / out["Close"] - 1
    out["fwd_high_1"] = out["High"].shift(-1) / out["Close"] - 1
    return out

def compute_relative_context(
    asset_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    sector_daily: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = asset_daily.copy()
    if not benchmark_daily.empty:
        aligned = benchmark_daily["Close"].reindex(out.index).ffill()
        out["rs_vs_benchmark"] = out["Close"] / aligned
        out["rs_bench_slope_5"] = slope(out["rs_vs_benchmark"], 5)
    else:
        out["rs_vs_benchmark"] = np.nan
        out["rs_bench_slope_5"] = np.nan

    if sector_daily is not None and not sector_daily.empty:
        aligned_sec = sector_daily["Close"].reindex(out.index).ffill()
        out["rs_vs_sector"] = out["Close"] / aligned_sec
        out["rs_sector_slope_5"] = slope(out["rs_vs_sector"], 5)
    else:
        out["rs_vs_sector"] = np.nan
        out["rs_sector_slope_5"] = np.nan
    return out

def classify_state(latest_h: pd.Series, latest_d: pd.Series, latest_w: pd.Series) -> Dict[str, str]:
    def tf_state(row: pd.Series, tf: str) -> str:
        if row.empty:
            return "Unknown"
        if row.get("rsi_14", np.nan) > 65 and row.get("tsi_slope_3", 0) > 0 and row.get("Close", 0) > row.get("ema_20", np.inf):
            return "Bullish"
        if row.get("rsi_14", np.nan) < 40 and row.get("tsi_slope_3", 0) < 0 and row.get("Close", np.inf) < row.get("ema_20", -np.inf):
            return "Bearish"
        if row.get("rsi_slope_3", 0) < 0 and row.get("price_slope_3", 0) >= 0:
            return "Stalling"
        return "Neutral"

    weekly = tf_state(latest_w, "weekly")
    daily = tf_state(latest_d, "daily")
    hourly = tf_state(latest_h, "hourly")

    if hourly == "Bullish" and daily in {"Bearish", "Stalling"}:
        alignment = "Hourly thrust vs weak daily backdrop"
    elif hourly == "Bullish" and daily == "Bullish" and weekly == "Bullish":
        alignment = "Full bullish alignment"
    elif hourly in {"Stalling", "Bearish"} and daily == "Bullish":
        alignment = "Short-term fade in larger uptrend"
    elif weekly == "Bearish" and hourly == "Bullish":
        alignment = "Countertrend bounce / fakeout risk"
    else:
        alignment = "Mixed alignment"

    return {"weekly": weekly, "daily": daily, "hourly": hourly, "alignment": alignment}

def detect_pattern_flags(latest_h: pd.Series, latest_d: pd.Series, latest_w: pd.Series) -> Dict[str, bool]:
    flags = {}
    flags["bear_kiss_hourly"] = bool(
        latest_h.get("price_slope_3", 0) >= 0
        and latest_h.get("rsi_slope_3", 0) < 0
        and latest_h.get("cci_slope_3", 0) < 0
        and latest_h.get("tsi_slope_3", 0) <= 0
    )
    flags["daily_bearish_divergence"] = bool(
        latest_d.get("price_slope_3", 0) >= 0
        and latest_d.get("rsi_slope_3", 0) < 0
        and latest_d.get("cci_slope_3", 0) < 0
    )
    flags["pinned_continuation_risk"] = bool(
        latest_h.get("rsi_14_pctile", 0) > 0.85
        and latest_h.get("pct_b_pctile", 0) > 0.85
        and latest_h.get("adx_14", 0) > 22
        and latest_h.get("close_in_range", 0) > 0.65
        and latest_h.get("price_slope_3", 0) > 0
    )
    flags["dead_cat_bounce_risk"] = bool(
        latest_w.get("Close", np.nan) < latest_w.get("ema_20", np.nan)
        and latest_d.get("price_slope_3", 0) > 0
        and latest_d.get("rsi_14", 50) < 55
        and latest_d.get("dist_ema20_pct", 0) < 0
    )
    flags["post_bottom_thrust"] = bool(
        latest_d.get("Close", np.nan) > latest_d.get("ema_20", np.nan)
        and latest_d.get("dist_sma50_pct", -1) > -0.02
        and latest_d.get("rsi_14", 0) > 55
        and latest_w.get("rsi_slope_3", 0) >= 0
    )
    return flags

def build_feature_snapshot(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    return {
        "hourly": hourly_df.iloc[-1] if not hourly_df.empty else pd.Series(dtype=float),
        "daily": daily_df.iloc[-1] if not daily_df.empty else pd.Series(dtype=float),
        "weekly": weekly_df.iloc[-1] if not weekly_df.empty else pd.Series(dtype=float),
    }

# -----------------------------
# Analogs and Monte Carlo
# -----------------------------
FEATURE_COLS = [
    "rsi_14_pctile", "cci_20_pctile", "tsi_pctile", "pct_b_pctile",
    "atr_stretch_pctile", "dist_ema10_pctile", "adx_14_pctile",
    "rsi_slope_3", "cci_slope_3", "tsi_slope_3", "price_slope_3",
    "rs_bench_slope_5", "rs_sector_slope_5",
]

def nearest_analogs(df: pd.DataFrame, feature_cols: List[str], top_n: int = 15) -> pd.DataFrame:
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

    std = X.std().replace(0, np.nan)
    z = (X - cur) / std
    hist["distance"] = np.sqrt((z.fillna(0) ** 2).sum(axis=1))
    analogs = hist.sort_values("distance").head(top_n).copy()
    cols = ["Close", "distance"] + [c for c in ["fwd_ret_1", "fwd_ret_2", "fwd_ret_3", "fwd_ret_5", "fwd_low_1", "fwd_high_1"] if c in analogs.columns]
    return analogs[cols]

def summarize_analogs(analogs: pd.DataFrame) -> Dict[str, float]:
    if analogs.empty:
        return {}
    out = {}
    for c in ["fwd_ret_1", "fwd_ret_2", "fwd_ret_3", "fwd_ret_5", "fwd_low_1", "fwd_high_1"]:
        if c in analogs.columns:
            out[f"{c}_mean"] = analogs[c].mean()
            out[f"{c}_median"] = analogs[c].median()
            out[f"{c}_p_down"] = float((analogs[c] < 0).mean()) if "ret" in c else np.nan
    return out

def monte_carlo_from_analogs(
    analogs: pd.DataFrame,
    horizon_days: int = 3,
    n_sims: int = 1000,
    seed: int = 7,
) -> pd.DataFrame:
    if analogs.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    return_cols = [c for c in [f"fwd_ret_{i}" for i in range(1, horizon_days + 1)] if c in analogs.columns]
    if not return_cols:
        return pd.DataFrame()

    boot = []
    vals = analogs[return_cols].dropna().values
    if len(vals) == 0:
        return pd.DataFrame()

    for _ in range(n_sims):
        row = vals[rng.integers(0, len(vals))]
        path = [1.0]
        for r in row:
            path.append(path[-1] * (1 + r))
        boot.append(path)

    sim = pd.DataFrame(boot).T
    sim.index.name = "step"
    return sim

def monte_carlo_summary(sim_paths: pd.DataFrame) -> Dict[str, float]:
    if sim_paths.empty:
        return {}
    terminal = sim_paths.iloc[-1] - 1
    return {
        "mc_mean": terminal.mean(),
        "mc_median": terminal.median(),
        "mc_p10": terminal.quantile(0.10),
        "mc_p90": terminal.quantile(0.90),
        "mc_prob_negative": float((terminal < 0).mean()),
    }

# -----------------------------
# Visualization
# -----------------------------
def plot_price_and_oscillator(ticker: str, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, hourly_df: pd.DataFrame):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"{ticker} Daily Price", "Daily Oscillators", "Hourly Oscillators"],
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
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["ema_20"], name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["sma_50"], name="SMA50"), row=1, col=1)

        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["rsi_14"], name="RSI14"), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["cci_20"], name="CCI20"), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_df.index, y=daily_df["tsi"], name="TSI"), row=2, col=1)

    if not hourly_df.empty:
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["rsi_14"], name="Hourly RSI14"), row=3, col=1)
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["cci_20"], name="Hourly CCI20"), row=3, col=1)
        fig.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df["tsi"], name="Hourly TSI"), row=3, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

def plot_monte_carlo(sim_paths: pd.DataFrame, ticker: str):
    if sim_paths.empty:
        st.info("Not enough analog data for Monte Carlo.")
        return
    fig = go.Figure()
    x = sim_paths.index
    fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.5, axis=1), name="Median Path"))
    fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.1, axis=1), name="10th %ile"))
    fig.add_trace(go.Scatter(x=x, y=sim_paths.quantile(0.9, axis=1), name="90th %ile"))
    fig.update_layout(height=350, title=f"{ticker} Conditional Monte Carlo (analog-conditioned)")
    st.plotly_chart(fig, use_container_width=True)

def score_setup(states: Dict[str, str], flags: Dict[str, bool], analog_summary: Dict[str, float]) -> Dict[str, float]:
    score = 50.0
    if states["hourly"] == "Bullish":
        score += 5
    if states["daily"] == "Bullish":
        score += 8
    if states["weekly"] == "Bullish":
        score += 10
    if flags.get("bear_kiss_hourly"):
        score -= 10
    if flags.get("daily_bearish_divergence"):
        score -= 10
    if flags.get("pinned_continuation_risk"):
        score -= 3
    if flags.get("dead_cat_bounce_risk"):
        score -= 15
    if flags.get("post_bottom_thrust"):
        score += 7

    p_down = analog_summary.get("fwd_ret_1_p_down", np.nan)
    if not pd.isna(p_down):
        score += (0.5 - p_down) * 20  # bullish if lower probability of next-day down
    mean_ret = analog_summary.get("fwd_ret_2_mean", 0)
    score += max(-10, min(10, mean_ret * 1000 / 10))
    score = float(np.clip(score, 0, 100))

    short_score = 100 - score
    return {
        "long_score": round(score, 1),
        "short_score": round(short_score, 1),
    }

# -----------------------------
# App workflow
# -----------------------------
SECTOR_DEFAULTS = {
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
    "SMH": "Semis",
}

def parse_ticker_input(raw_text: str) -> List[str]:
    cleaned = raw_text.replace("\n", ",").replace(";", ",").replace("\t", ",")
    tickers = [x.strip().upper() for x in cleaned.split(",") if x.strip()]
    return list(dict.fromkeys(tickers))

def load_watchlist_from_upload(uploaded_file) -> List[str]:
    if uploaded_file is None:
        return []
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    first_col = df.columns[0]
    vals = df[first_col].dropna().astype(str).str.upper().tolist()
    return list(dict.fromkeys(vals))

def get_market_context() -> Dict[str, pd.DataFrame]:
    spy = fetch_yahoo_prices("SPY", "1d", period="5y")
    rsp = fetch_yahoo_prices("RSP", "1d", period="5y")
    qqq = fetch_yahoo_prices("QQQ", "1d", period="5y")
    return {"SPY": spy, "RSP": rsp, "QQQ": qqq}

@st.cache_data(ttl=900, show_spinner=False)
def prepare_ticker_data(ticker: str, sector_etf: Optional[str], benchmark: str):
    hourly = fetch_yahoo_prices(ticker, "1h", period="60d")
    daily = fetch_yahoo_prices(ticker, "1d", period="5y")
    if daily.empty:
        daily = fetch_defeat_fallback(ticker, start="2018-01-01")
    if hourly.empty:
        hourly = fetch_yahoo_prices(ticker, "60m", period="60d")
    if hourly.empty or daily.empty:
        return {"hourly": pd.DataFrame(), "daily": pd.DataFrame(), "weekly": pd.DataFrame()}

    weekly = resample_ohlcv(daily, "W-FRI")
    hourly = enrich_price_features(hourly, "hourly")
    daily = enrich_price_features(daily, "daily")
    weekly = enrich_price_features(weekly, "weekly")

    benchmark_df = fetch_yahoo_prices(benchmark, "1d", period="5y")
    sector_df = fetch_yahoo_prices(sector_etf, "1d", period="5y") if sector_etf else pd.DataFrame()

    daily = compute_relative_context(daily, benchmark_df, sector_df)
    daily = add_forward_returns(daily)

    return {"hourly": hourly, "daily": daily, "weekly": weekly}

def describe_setup(states: Dict[str, str], flags: Dict[str, bool], analog_summary: Dict[str, float]) -> str:
    bits = []
    bits.append(f"Alignment: {states['alignment']}.")
    if flags.get("bear_kiss_hourly"):
        bits.append("Hourly momentum looks like a bear-kiss / rollover setup.")
    if flags.get("daily_bearish_divergence"):
        bits.append("Daily momentum is diverging against price.")
    if flags.get("pinned_continuation_risk"):
        bits.append("Pinning / continuation risk remains elevated.")
    if flags.get("dead_cat_bounce_risk"):
        bits.append("This bounce has dead-cat / countertrend characteristics.")
    if flags.get("post_bottom_thrust"):
        bits.append("The stock also shows post-bottom thrust behavior, so shorts may need extra confirmation.")
    if analog_summary:
        p_down = analog_summary.get("fwd_ret_1_p_down", np.nan)
        med2 = analog_summary.get("fwd_ret_2_median", np.nan)
        if not pd.isna(p_down):
            bits.append(f"Nearest analogs show about {p_down:.0%} odds of a negative next day.")
        if not pd.isna(med2):
            bits.append(f"Median 2-day analog return: {med2:.2%}.")
    return " ".join(bits)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Predictive Trading Analysis Engine")

benchmark = st.sidebar.selectbox("Benchmark", ["SPY", "RSP", "QQQ"], index=1)
default_sector = st.sidebar.selectbox("Default sector ETF context", ["", *SECTOR_DEFAULTS.keys()], index=0)
uploaded_watchlist = st.sidebar.file_uploader("Upload watchlist CSV/XLSX", type=["csv", "xlsx"])
ticker_text = st.sidebar.text_area(
    "Or paste tickers",
    value="AAPL, MSFT, NVDA, INTC, AMD",
    height=100,
)
top_analogs = st.sidebar.slider("Nearest analog count", 5, 30, 15)
mc_sims = st.sidebar.slider("Monte Carlo simulations", 250, 3000, 1000, step=250)
run_button = st.sidebar.button("Run analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Yahoo intraday is limited to recent history. The app can use an optional defeat-style fallback for longer daily history if configured."
)

# -----------------------------
# Main
# -----------------------------
st.title("Predictive Trading Analysis Engine")
st.caption(
    "Deep analysis for a shortlist of stocks: hourly timing, daily setup, weekly regime, analogs, and conditional Monte Carlo."
)

with st.expander("What this version does"):
    st.markdown(
        """
- Accepts a watchlist from upload or pasted tickers.
- Pulls hourly and daily data, then builds weekly bars.
- Detects timeframe conflict such as **hourly strong vs daily rolling**.
- Flags patterns including **bear kiss**, **daily divergence**, **pinning risk**, **dead-cat bounce risk**, and **post-bottom thrust**.
- Finds nearest historical analogs on the same ticker.
- Runs analog-conditioned Monte Carlo for forward path scenarios.
- Scores long vs short setup quality.
        """
    )

if run_button:
    watchlist = load_watchlist_from_upload(uploaded_watchlist)
    watchlist = watchlist if watchlist else parse_ticker_input(ticker_text)
    if not watchlist:
        st.error("Please upload a watchlist file or paste at least one ticker.")
        st.stop()

    tabs = st.tabs(["Summary", "Per-Ticker Deep Dive"])

    summary_rows = []
    per_ticker_payload = {}

    progress = st.progress(0)
    for i, ticker in enumerate(watchlist):
        data = prepare_ticker_data(ticker, default_sector or None, benchmark)
        hourly_df = data["hourly"]
        daily_df = data["daily"]
        weekly_df = data["weekly"]

        if hourly_df.empty or daily_df.empty or weekly_df.empty:
            summary_rows.append(
                {
                    "Ticker": ticker,
                    "Status": "No data",
                    "Alignment": "N/A",
                    "Long Score": np.nan,
                    "Short Score": np.nan,
                }
            )
            progress.progress((i + 1) / len(watchlist))
            continue

        snap = build_feature_snapshot(hourly_df, daily_df, weekly_df)
        states = classify_state(snap["hourly"], snap["daily"], snap["weekly"])
        flags = detect_pattern_flags(snap["hourly"], snap["daily"], snap["weekly"])
        analogs = nearest_analogs(daily_df, FEATURE_COLS, top_n=top_analogs)
        analog_summary = summarize_analogs(analogs)
        sim_paths = monte_carlo_from_analogs(analogs, horizon_days=3, n_sims=mc_sims, seed=11)
        mc_summary = monte_carlo_summary(sim_paths)
        scores = score_setup(states, flags, analog_summary)
        narrative = describe_setup(states, flags, analog_summary)

        summary_rows.append(
            {
                "Ticker": ticker,
                "Status": "OK",
                "Alignment": states["alignment"],
                "Weekly": states["weekly"],
                "Daily": states["daily"],
                "Hourly": states["hourly"],
                "Long Score": scores["long_score"],
                "Short Score": scores["short_score"],
                "Bear Kiss": flags["bear_kiss_hourly"],
                "Daily Div": flags["daily_bearish_divergence"],
                "Pinned": flags["pinned_continuation_risk"],
                "Dead Cat": flags["dead_cat_bounce_risk"],
                "Post-Bottom": flags["post_bottom_thrust"],
                "Analog P(Next Day Down)": analog_summary.get("fwd_ret_1_p_down", np.nan),
                "Analog Median 2D": analog_summary.get("fwd_ret_2_median", np.nan),
                "MC Median": mc_summary.get("mc_median", np.nan),
                "Narrative": narrative,
            }
        )

        per_ticker_payload[ticker] = {
            "hourly": hourly_df,
            "daily": daily_df,
            "weekly": weekly_df,
            "states": states,
            "flags": flags,
            "analogs": analogs,
            "analog_summary": analog_summary,
            "sim_paths": sim_paths,
            "mc_summary": mc_summary,
            "scores": scores,
            "narrative": narrative,
        }
        progress.progress((i + 1) / len(watchlist))

    with tabs[0]:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["Long Score", "Short Score"], ascending=[False, True]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download summary CSV",
            data=csv_bytes,
            file_name="predictive_trading_summary.csv",
            mime="text/csv",
        )

    with tabs[1]:
        if not per_ticker_payload:
            st.info("No tickers with usable data.")
        else:
            selected = st.selectbox("Select ticker", list(per_ticker_payload.keys()))
            payload = per_ticker_payload[selected]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Long Score", payload["scores"]["long_score"])
            c2.metric("Short Score", payload["scores"]["short_score"])
            c3.metric("Weekly / Daily / Hourly", f"{payload['states']['weekly']} / {payload['states']['daily']} / {payload['states']['hourly']}")
            c4.metric("Analog P(next day down)", f"{payload['analog_summary'].get('fwd_ret_1_p_down', np.nan):.0%}" if payload["analog_summary"] else "N/A")

            st.markdown(f"**Interpretation:** {payload['narrative']}")

            flags_display = pd.DataFrame(
                [{"Flag": k, "Triggered": v} for k, v in payload["flags"].items()]
            )
            st.dataframe(flags_display, use_container_width=True, hide_index=True)

            plot_price_and_oscillator(selected, payload["daily"].tail(220), payload["weekly"].tail(140), payload["hourly"].tail(120))

            st.subheader("Nearest historical analogs")
            if payload["analogs"].empty:
                st.info("Not enough analog history for this ticker.")
            else:
                analogs_display = payload["analogs"].copy()
                analogs_display.index = analogs_display.index.strftime("%Y-%m-%d")
                st.dataframe(analogs_display, use_container_width=True)
                st.markdown(
                    f"""
**Analog summary**
- Mean next-day return: {payload['analog_summary'].get('fwd_ret_1_mean', np.nan):.2%}
- Median next-day return: {payload['analog_summary'].get('fwd_ret_1_median', np.nan):.2%}
- Probability next day down: {payload['analog_summary'].get('fwd_ret_1_p_down', np.nan):.0%}
- Median 2-day return: {payload['analog_summary'].get('fwd_ret_2_median', np.nan):.2%}
                    """
                )

            st.subheader("Conditional Monte Carlo")
            plot_monte_carlo(payload["sim_paths"], selected)
            if payload["mc_summary"]:
                st.json(payload["mc_summary"])

else:
    st.info("Set your benchmark, paste or upload tickers, and click **Run analysis**.")
