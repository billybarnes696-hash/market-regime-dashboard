# nyad_nysi_nyhl_nymo_breadth_app_stockcharts_daily.py
import io
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Breadth Oscillator Dashboard", layout="wide")

SERIES = ["NYAD", "NYSI", "NYHL", "NYMO", "RSP"]


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace("$", "")
        .replace("^", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .upper()
    )


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(1, int(span)), adjust=False).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(10, window // 4)).mean()
    std = series.rolling(window, min_periods=max(10, window // 4)).std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).clip(-5, 5)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / max(1, length), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / max(1, length), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def tsi(series: pd.Series, long_len: int = 25, short_len: int = 13, signal_len: int = 7):
    momentum = series.diff()
    abs_momentum = momentum.abs()
    ds_m = ema(ema(momentum, long_len), short_len)
    ds_a = ema(ema(abs_momentum, long_len), short_len).replace(0, np.nan)
    out = 100 * (ds_m / ds_a)
    sig = ema(out, signal_len)
    return out, sig


def cci(series: pd.Series, length: int = 20) -> pd.Series:
    sma = series.rolling(length, min_periods=max(5, length // 3)).mean()
    mad = series.rolling(length, min_periods=max(5, length // 3)).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    mad = pd.Series(mad, index=series.index).replace(0, np.nan)
    return (series - sma) / (0.015 * mad)


def bb_percent(series: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(length, min_periods=max(5, length // 3)).mean()
    sd = series.rolling(length, min_periods=max(5, length // 3)).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = (upper - lower).replace(0, np.nan)
    return (series - lower) / width


def detect_symbol_table(df: pd.DataFrame) -> bool:
    cols = [normalize_name(c) for c in df.columns]
    return any(c in cols for c in ["SYMBOL", "TICKER"]) and any(c in cols for c in ["CLOSE", "VALUE", "LAST"])


def parse_symbol_table(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    temp = df.copy()
    rename_map = {}
    for c in temp.columns:
        nc = normalize_name(c)
        if nc in ["SYMBOL", "TICKER"]:
            rename_map[c] = "symbol"
        elif nc in ["CLOSE", "VALUE", "LAST"]:
            rename_map[c] = "value"
    temp = temp.rename(columns=rename_map)
    temp["symbol"] = temp["symbol"].astype(str).str.upper().str.replace("$", "", regex=False).str.strip()
    temp["value"] = to_num(temp["value"])
    keep = temp[temp["symbol"].isin(SERIES)][["symbol", "value"]].dropna()
    if keep.empty:
        return pd.DataFrame()
    row = {"Date": pd.to_datetime(snapshot_date)}
    for _, rec in keep.iterrows():
        row[rec["symbol"]] = float(rec["value"])
    return pd.DataFrame([row])


def parse_stockcharts_daily_csv(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return pd.DataFrame()

    first = [x.strip() for x in lines[0].split(",")]
    second = [x.strip() for x in lines[1].split(",")]

    symbol = None
    if first:
        symbol = normalize_name(first[0])
    if symbol not in SERIES:
        return pd.DataFrame()

    cols = []
    for i in range(max(len(first), len(second))):
        a = first[i].strip() if i < len(first) else ""
        b = second[i].strip() if i < len(second) else ""
        if normalize_name(b) == "DATE" or normalize_name(a) == "DATE":
            cols.append("Date")
        elif normalize_name(b) == "CLOSE":
            cols.append(symbol)
        else:
            cols.append(f"drop_{i}")

    data = []
    for ln in lines[2:]:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < len(cols):
            parts += [""] * (len(cols) - len(parts))
        row = dict(zip(cols, parts[: len(cols)]))
        data.append(row)

    df = pd.DataFrame(data)
    keep_cols = [c for c in ["Date", symbol] if c in df.columns]
    if not keep_cols or "Date" not in keep_cols:
        return pd.DataFrame()

    out = df[keep_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[symbol] = to_num(out[symbol])
    out = out.dropna(subset=["Date"]).sort_values("Date")
    return out


def parse_csv_bytes(raw: bytes, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    sc = parse_stockcharts_daily_csv(raw)
    if not sc.empty:
        return sc

    df = pd.read_csv(io.BytesIO(raw))
    if detect_symbol_table(df):
        if snapshot_date is None:
            snapshot_date = pd.Timestamp.today().normalize()
        return parse_symbol_table(df, snapshot_date)

    rename_map = {}
    for c in df.columns:
        nc = normalize_name(c)
        if nc == "DATE":
            rename_map[c] = "Date"
        elif nc in SERIES:
            rename_map[c] = nc
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Date" not in df.columns and len(df.columns) > 0:
        maybe_date = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if maybe_date.notna().sum() > 0:
            df["Date"] = maybe_date

    if "Date" not in df.columns:
        return pd.DataFrame()

    keep_cols = ["Date"] + [c for c in SERIES if c in df.columns]
    out = df[keep_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in SERIES:
        if c in out.columns:
            out[c] = to_num(out[c])
    return out.dropna(subset=["Date"]).sort_values("Date")


def load_uploaded(uploaded_file, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    parts = []

    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    try:
                        piece = parse_csv_bytes(zf.read(name), snapshot_date=snapshot_date)
                        if not piece.empty:
                            parts.append(piece)
                    except Exception:
                        continue
    elif suffix == ".csv":
        piece = parse_csv_bytes(raw, snapshot_date=snapshot_date)
        if not piece.empty:
            parts.append(piece)

    if not parts:
        return pd.DataFrame()

    merged = parts[0]
    for piece in parts[1:]:
        merged = pd.merge(merged, piece, on="Date", how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        for dc in dup_cols:
            base = dc[:-4]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[dc])
            else:
                merged[base] = merged[dc]
        merged = merged.drop(columns=dup_cols)

    return merged.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def append_snapshot(history: pd.DataFrame, snapshot_file, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    if snapshot_file is None:
        return history.copy()
    snap = load_uploaded(snapshot_file, snapshot_date=snapshot_date)
    if snap.empty:
        return history.copy()
    snap["Date"] = pd.to_datetime(snapshot_date)
    out = pd.concat([history, snap], ignore_index=True)
    return out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def prepare_series(df: pd.DataFrame, nyad_cumulative: bool, nyhl_cumulative: bool) -> pd.DataFrame:
    work = df.copy().sort_values("Date").reset_index(drop=True)
    for c in SERIES:
        if c in work.columns:
            work[c] = to_num(work[c])
    if "NYAD" in work.columns and not nyad_cumulative:
        work["NYAD"] = work["NYAD"].fillna(0).cumsum()
    if "NYHL" in work.columns and not nyhl_cumulative:
        work["NYHL"] = work["NYHL"].fillna(0).cumsum()
    return work


def build_composite(df: pd.DataFrame, weights: dict[str, float], z_window: int, smooth_span: int) -> pd.DataFrame:
    work = df.copy()
    normed = {}
    for key, wt in weights.items():
        if wt > 0 and key in work.columns:
            normed[key] = rolling_zscore(work[key], z_window)
    if not normed:
        return work

    norm_df = pd.DataFrame(normed, index=work.index)
    active_weight_sum = sum(w for k, w in weights.items() if k in norm_df.columns and w > 0)
    composite_raw = pd.Series(0.0, index=work.index)
    for key, wt in weights.items():
        if key in norm_df.columns and wt > 0:
            composite_raw = composite_raw.add(norm_df[key] * (wt / active_weight_sum), fill_value=0.0)

    work["Breadth_Composite_Raw"] = composite_raw
    work["Breadth_Composite"] = ema(composite_raw, smooth_span)
    return work


def add_oscillators(df: pd.DataFrame, rsi_len: int, tsi_long: int, tsi_short: int, tsi_signal: int, cci_len: int, bb_len: int, bb_std: float) -> pd.DataFrame:
    work = df.copy()
    base = work["Breadth_Composite"]
    work["Composite_RSI"] = rsi(base, rsi_len)
    work["Composite_TSI"], work["Composite_TSI_Signal"] = tsi(base, tsi_long, tsi_short, tsi_signal)
    work["Composite_CCI"] = cci(base, cci_len)
    work["Composite_BBP"] = bb_percent(base, bb_len, bb_std)
    return work


def regime_label(row: pd.Series) -> str:
    rsi_v = row.get("Composite_RSI", np.nan)
    tsi_v = row.get("Composite_TSI", np.nan)
    tsi_sig = row.get("Composite_TSI_Signal", np.nan)
    if pd.isna(rsi_v) or pd.isna(tsi_v) or pd.isna(tsi_sig):
        return "Insufficient data"
    improving = tsi_v > tsi_sig
    if rsi_v < 35:
        return "Repair / Improving" if improving else "Washout / Weakening"
    if rsi_v < 50:
        return "Neutral / Improving" if improving else "Neutral / Regressing"
    if rsi_v < 70:
        return "Constructive / Trending" if improving else "Constructive / Fading"
    return "Expansion / Strong" if improving else "Exhaustion Risk"


def trim_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    end = df["Date"].max()
    start = end - pd.DateOffset(years=years)
    return df[df["Date"] >= start].copy()


def detect_market_regime(price_series: pd.Series) -> pd.Series:
    """Simple regime detection based on trend and volatility"""
    ma50 = price_series.rolling(50).mean()
    ma200 = price_series.rolling(200).mean()
    trend = ma50 > ma200
    
    returns = price_series.pct_change()
    vol = returns.rolling(20).std()
    vol_percentile = vol.rank(pct=True)
    high_vol = vol_percentile > 0.7
    
    regime = pd.Series('Range', index=price_series.index)
    regime[trend & ~high_vol] = 'Bull'
    regime[~trend & ~high_vol] = 'Bear'
    regime[high_vol] = 'High Vol'
    
    return regime


def create_weighted_composite(df: pd.DataFrame, weights: dict, breadth_cols: list) -> pd.Series:
    """Create weighted composite signal for optimization"""
    composite = pd.Series(0, index=df.index)
    total_weight = sum(weights.values())
    
    for col in breadth_cols:
        if col in df.columns and col in weights:
            normalized = (df[col] - df[col].mean()) / df[col].std()
            composite += normalized * (weights[col] / total_weight)
    
    return composite


def calculate_robust_metrics(composite: pd.Series, forward_returns: pd.Series) -> dict:
    """Calculate multiple metrics for robust evaluation"""
    
    predicted_dir = np.sign(composite)
    actual_dir = np.sign(forward_returns)
    directional_acc = (predicted_dir == actual_dir).mean()
    
    strategy_returns = predicted_dir * forward_returns
    
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    gross_profits = strategy_returns[strategy_returns > 0].sum()
    gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
    
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    avg_return = strategy_returns.mean()
    median_return = strategy_returns.median()
    
    # Blended score: 30% accuracy, 25% sharpe, 20% profit factor, 15% calmar, 10% median return
    blended_score = (
        directional_acc * 0.30 +
        (sharpe / 2) * 0.25 +
        min(profit_factor / 2, 1.5) * 0.20 +
        min(calmar / 2, 1.0) * 0.15 +
        (median_return * 100) * 0.10
    )
    
    return {
        'directional_accuracy': directional_acc,
        'sharpe_ratio': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_return': avg_return,
        'median_return': median_return,
        'calmar_ratio': calmar,
        'blended_score': blended_score
    }


def run_breadth_optimization_robust(df: pd.DataFrame, optimization_years: int = 5) -> dict:
    """
    ROBUST optimization with walk-forward testing and guardrails against overfitting
    Uses only the last N years of data for optimization
    """
    
    # Limit data to optimization lookback period
    end_date = df["Date"].max()
    start_date = end_date - pd.DateOffset(years=optimization_years)
    opt_df = df[df["Date"] >= start_date].copy()
    
    print(f"Optimizing on {optimization_years} years of data ({len(opt_df)} days)")
    
    # Guardrail 1: Define bounds (prevent extreme weights)
    MIN_WEIGHT = 0.05   # 5% minimum
    MAX_WEIGHT = 0.50   # 50% maximum
    
    breadth_cols = ['NYAD', 'NYSI', 'NYHL', 'NYMO']
    available_cols = [c for c in breadth_cols if c in opt_df.columns]
    
    if len(available_cols) < 2:
        return {'error': 'Need at least 2 breadth series for optimization'}
    
    if 'RSP' not in opt_df.columns:
        return {'error': 'RSP column required for optimization'}
    
    def objective_function(weights, train_data, horizon):
        weight_dict = {col: weights[i] for i, col in enumerate(available_cols)}
        composite = create_weighted_composite(train_data, weight_dict, available_cols)
        
        forward_returns = train_data['RSP'].shift(-horizon) / train_data['RSP'] - 1
        
        valid_idx = composite.notna() & forward_returns.notna()
        if valid_idx.sum() < 50:
            return 1e6
        
        metrics = calculate_robust_metrics(composite[valid_idx], forward_returns[valid_idx])
        
        # Penalize unrealistic weights
        weight_penalty = 0
        if max(weights) > MAX_WEIGHT * 1.1:
            weight_penalty += (max(weights) - MAX_WEIGHT) * 10
        if min(weights) < MIN_WEIGHT * 0.9:
            weight_penalty += (MIN_WEIGHT - min(weights)) * 10
        
        return -metrics['blended_score'] + weight_penalty
    
    # Walk-forward testing
    window_size = min(252, len(opt_df) // 3)
    rebalance_freq = 21
    horizon = 5
    
    walk_forward_results = []
    weight_history = []
    regime_performance = {'Bull': [], 'Bear': [], 'Range': [], 'High Vol': []}
    
    opt_df['regime'] = detect_market_regime(opt_df['RSP'])
    
    for i in range(window_size, len(opt_df) - horizon, rebalance_freq):
        train_data = opt_df.iloc[i - window_size:i]
        test_data = opt_df.iloc[i:i + rebalance_freq]
        
        if len(test_data) < 10:
            continue
        
        bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in available_cols]
        
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                args=(train_data, horizon),
                maxiter=20,  # Reduced for speed
                popsize=6,   # Reduced for speed
                seed=42,
                disp=False
            )
            
            optimal_weights = result.x
            optimal_weights = optimal_weights / optimal_weights.sum()
            weights_dict = {col: optimal_weights[i] for i, col in enumerate(available_cols)}
            weight_history.append({'date': test_data['Date'].iloc[0], **weights_dict})
            
            composite_test = create_weighted_composite(test_data, weights_dict, available_cols)
            forward_returns_test = test_data['RSP'].shift(-horizon) / test_data['RSP'] - 1
            
            valid_idx = composite_test.notna() & forward_returns_test.notna()
            if valid_idx.sum() > 0:
                metrics = calculate_robust_metrics(composite_test[valid_idx], forward_returns_test[valid_idx])
                
                walk_forward_results.append({
                    'period_start': test_data['Date'].iloc[0],
                    'period_end': test_data['Date'].iloc[-1],
                    **metrics
                })
                
                test_regime = test_data.loc[valid_idx, 'regime'].iloc[0] if len(test_data) > 0 else 'Range'
                if test_regime in regime_performance:
                    regime_performance[test_regime].append(metrics['blended_score'])
        except Exception as e:
            continue
    
    if not walk_forward_results:
        return {'error': 'Insufficient data for walk-forward optimization'}
    
    # Weight stability analysis
    weight_df = pd.DataFrame(weight_history).set_index('date')
    weight_stability = {
        'mean_weights': weight_df.mean().to_dict(),
        'std_weights': weight_df.std().to_dict(),
        'max_swing': (weight_df.max() - weight_df.min()).to_dict(),
        'stability_score': 1 - (weight_df.std().mean() / weight_df.mean().mean()) if weight_df.mean().mean() > 0 else 0
    }
    
    # Regime consistency
    regime_consistency = {}
    for regime, scores in regime_performance.items():
        if scores:
            regime_consistency[regime] = np.mean(scores)
    
    # Final recommendation: median weights from walk-forward
    recommended_weights = weight_df.median().to_dict()
    
    results_df = pd.DataFrame(walk_forward_results)
    expected_performance = {
        'directional_accuracy': results_df['directional_accuracy'].mean(),
        'sharpe_ratio': results_df['sharpe_ratio'].mean(),
        'profit_factor': results_df['profit_factor'].mean(),
        'max_drawdown': results_df['max_drawdown'].mean(),
        'blended_score': results_df['blended_score'].mean(),
        'consistency_std': results_df['blended_score'].std()
    }
    
    confidence_score = (
        weight_stability['stability_score'] * 0.5 +
        (1 - min(1, expected_performance['consistency_std'] / 0.2)) * 0.3 +
        min(1, expected_performance['blended_score'] / 0.5) * 0.2
    )
    
    return {
        'recommended_weights': recommended_weights,
        'walk_forward_results': results_df,
        'weight_history': weight_df,
        'weight_stability': weight_stability,
        'regime_consistency': regime_consistency,
        'expected_performance': expected_performance,
        'confidence_score': confidence_score,
        'best_horizon': 5,
        'recommendations': {
            'expected_accuracy': expected_performance['directional_accuracy'],
            'best_horizon': 5
        }
    }


def normalize_for_oscillator(rsp_series: pd.Series, oscillator_series: pd.Series) -> pd.Series:
    """Normalize RSP to same scale as oscillator for visual comparison"""
    # Remove NaN values for scaling
    rsp_clean = rsp_series.dropna()
    osc_clean = oscillator_series.dropna()
    
    if len(rsp_clean) == 0 or len(osc_clean) == 0:
        return rsp_series
    
    # Scale RSP to match oscillator's range and mean
    rsp_normalized = (rsp_series - rsp_clean.min()) / (rsp_clean.max() - rsp_clean.min())
    rsp_normalized = rsp_normalized * (osc_clean.max() - osc_clean.min()) + osc_clean.min()
    return rsp_normalized


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📊 NYAD + NYSI + NYHL + NYMO Breadth Oscillator")
st.caption("Holistic breadth composite with adjustable weights and institutional-grade optimization")

# Initialize session state
if 'opt_results' not in st.session_state:
    st.session_state.opt_results = None
if 'combined_oscillator' not in st.session_state:
    st.session_state.combined_oscillator = False

with st.sidebar:
    st.header("📁 Uploads")
    historical_file = st.file_uploader("Historical ZIP or CSV", type=["zip", "csv"])
    snapshot_date = st.date_input("Daily snapshot date", value=pd.Timestamp.today().date())
    snapshot_file = st.file_uploader("Daily snapshot (optional)", type=["zip", "csv"])
    
    st.header("⚙️ Series handling")
    st.caption("NYAD and NYHL are daily values - will be cumulated")
    nyad_cumulative = st.checkbox("NYAD already cumulative", value=False)
    nyhl_cumulative = st.checkbox("NYHL already cumulative", value=False)
    
    st.header("🎯 Manual Weights")
    nyad_w = st.slider("NYAD %", 0, 100, 25, 1)
    nysi_w = st.slider("NYSI %", 0, 100, 25, 1)
    nyhl_w = st.slider("NYHL %", 0, 100, 25, 1)
    nymo_w = st.slider("NYMO %", 0, 100, 25, 1)
    
    st.header("🔬 Composite settings")
    z_window = st.slider("Normalization window", 20, 252, 126, 1)
    smooth_span = st.slider("Composite EMA smoothing", 1, 20, 5, 1)
    
    st.header("📈 Oscillator settings")
    rsi_len = st.slider("RSI length", 2, 50, 14, 1)
    tsi_long = st.slider("TSI long", 2, 60, 25, 1)
    tsi_short = st.slider("TSI short", 2, 40, 13, 1)
    tsi_signal = st.slider("TSI signal", 1, 20, 7, 1)
    cci_len = st.slider("CCI length", 2, 60, 20, 1)
    bb_len = st.slider("BB% length", 2, 60, 20, 1)
    bb_std = st.slider("BB% std dev", 0.5, 4.0, 2.0, 0.1)
    
    st.header("📺 Display")
    lookback_years = st.slider("Chart lookback (years)", 1, 20, 2, 1)
    
    st.header("🤖 Weight Optimizer")
    optimization_lookback = st.selectbox(
        "Optimization lookback (years)",
        options=[1, 2, 5, 20],
        index=2,  # Default to 5 years
        help="Use last N years of data to find optimal weights. 5 years recommended."
    )
    st.caption(f"Will use last {optimization_lookback} year(s) of data for optimization")
    
    run_opt = st.button("🔍 Find Optimal Weights", type="primary")
    
    st.header("🔄 Combined Signal")
    st.session_state.combined_oscillator = st.checkbox(
        "Show Combined Oscillator + RSP",
        value=False,
        help="Creates a blended signal weighting breadth composite and RSP momentum"
    )

if historical_file is None:
    st.info("📂 Upload a historical ZIP or CSV to begin.")
    st.markdown("""
    ### Required format:
    - **ZIP file** with CSVs: `NYAD_daily.csv`, `NYSI_daily.csv`, `NYHL_daily.csv`, `NYMO_daily.csv`, `RSP_daily.csv`
    - **Single CSV** with columns: `Date, NYAD, NYSI, NYHL, NYMO, RSP`
    
    Data can be downloaded from StockCharts.com
    """)
    st.stop()

# Load and process data
history = load_uploaded(historical_file, snapshot_date=pd.Timestamp(snapshot_date))
if history.empty:
    st.error("Could not parse the uploaded file. Include Date and at least one of NYAD, NYSI, NYHL, NYMO, or RSP.")
    st.stop()

merged = append_snapshot(history, snapshot_file, pd.Timestamp(snapshot_date))
merged = prepare_series(merged, nyad_cumulative=nyad_cumulative, nyhl_cumulative=nyhl_cumulative)

# Run optimization if button was clicked
if run_opt:
    with st.spinner(f"Running walk-forward optimization on last {optimization_lookback} year(s) of data... This may take 30-90 seconds..."):
        if 'RSP' in merged.columns and len(merged) > 100:
            opt_results = run_breadth_optimization_robust(merged, optimization_years=optimization_lookback)
            if 'error' not in opt_results:
                st.session_state.opt_results = opt_results
                st.success(f"Optimization complete using last {optimization_lookback} year(s) of data!")
            else:
                st.error(f"Optimization failed: {opt_results['error']}")
        else:
            st.error("Need at least 100 days of data with RSP column for optimization")

# Use optimized weights if available, otherwise use manual weights
if st.session_state.opt_results is not None and 'recommended_weights' in st.session_state.opt_results:
    opt_weights = st.session_state.opt_results['recommended_weights']
    weights = {
        "NYAD": opt_weights.get("NYAD", nyad_w / 100),
        "NYSI": opt_weights.get("NYSI", nysi_w / 100),
        "NYHL": opt_weights.get("NYHL", nyhl_w / 100),
        "NYMO": opt_weights.get("NYMO", nymo_w / 100),
    }
    st.sidebar.success(f"✨ Using optimized weights ({optimization_lookback}y lookback)")
else:
    weights = {"NYAD": nyad_w / 100, "NYSI": nysi_w / 100, "NYHL": nyhl_w / 100, "NYMO": nymo_w / 100}

# Build composite
model = build_composite(merged, weights=weights, z_window=z_window, smooth_span=smooth_span)

if "Breadth_Composite" not in model.columns:
    st.error("No composite could be built. Check that at least one breadth series was found and weighted above 0%.")
    st.stop()

model = add_oscillators(model, rsi_len, tsi_long, tsi_short, tsi_signal, cci_len, bb_len, bb_std)
model["State"] = model.apply(regime_label, axis=1)

# Add combined oscillator if checkbox is checked
if st.session_state.combined_oscillator:
    # Calculate RSP momentum (rate of change)
    rsp_momentum = model["RSP"].pct_change(10)  # 10-day momentum
    rsp_momentum_norm = (rsp_momentum - rsp_momentum.min()) / (rsp_momentum.max() - rsp_momentum.min()) * 2 - 1
    
    # Combined signal: 70% breadth composite + 30% RSP momentum
    model["Combined_Signal"] = model["Breadth_Composite"] * 0.7 + rsp_momentum_norm * 0.3
    model["Combined_RSI"] = rsi(model["Combined_Signal"], rsi_len)
    model["Combined_TSI"], model["Combined_TSI_Signal"] = tsi(model["Combined_Signal"], tsi_long, tsi_short, tsi_signal)

latest = model.iloc[-1]
view = trim_years(model, lookback_years)

# Calendar Date Picker for Historical View
st.subheader("📅 Historical Date Lookup")
col1, col2 = st.columns([2, 3])
with col1:
    lookup_date = st.date_input(
        "Select a date to view oscillator values",
        value=model["Date"].max().date(),
        min_value=model["Date"].min().date(),
        max_value=model["Date"].max().date()
    )
with col2:
    if lookup_date:
        lookup_data = model[model["Date"].dt.date == lookup_date]
        if not lookup_data.empty:
            lookup_row = lookup_data.iloc[0]
            st.metric("Market State on this date", lookup_row.get("State", "N/A"))
        else:
            st.warning("No data for selected date")

# Metrics row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Composite RSI", f"{latest['Composite_RSI']:.1f}" if pd.notna(latest["Composite_RSI"]) else "n/a")
m2.metric("Composite TSI", f"{latest['Composite_TSI']:.1f}" if pd.notna(latest["Composite_TSI"]) else "n/a")
m3.metric("TSI Signal", f"{latest['Composite_TSI_Signal']:.1f}" if pd.notna(latest["Composite_TSI_Signal"]) else "n/a")
m4.metric("Composite CCI", f"{latest['Composite_CCI']:.1f}" if pd.notna(latest["Composite_CCI"]) else "n/a")
m5.metric("Composite BB%", f"{latest['Composite_BBP']:.2f}" if pd.notna(latest["Composite_BBP"]) else "n/a")

st.markdown(f"**📌 Current state:** {latest['State']}")

# Main chart
fig_main = go.Figure()

if st.session_state.combined_oscillator and "Combined_Signal" in view.columns:
    fig_main.add_trace(go.Scatter(x=view["Date"], y=view["Combined_Signal"], mode="lines", name="Combined Signal", line=dict(color="purple", width=2)))
    fig_main.add_trace(go.Scatter(x=view["Date"], y=view["Breadth_Composite"], mode="lines", name="Breadth Composite", line=dict(color="blue", width=1.5, dash="dot")))
else:
    fig_main.add_trace(go.Scatter(x=view["Date"], y=view["Breadth_Composite"], mode="lines", name="Breadth Composite", line=dict(color="blue", width=2)))

if "RSP" in view.columns:
    # Normalize RSP for comparison
    rsp_norm = view["RSP"] / view["RSP"].iloc[0] * view["Breadth_Composite"].iloc[0]
    fig_main.add_trace(go.Scatter(x=view["Date"], y=rsp_norm, mode="lines", name="RSP (normalized)", line=dict(color="red", width=1.5), yaxis="y2"))

fig_main.update_layout(
    title="Breadth Composite vs RSP (normalized)",
    height=460,
    xaxis_title="Date",
    yaxis_title="Breadth Composite / Combined Signal",
    yaxis2=dict(title="RSP", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_main, width="stretch")

# Oscillator charts with RSP overlay
left, right = st.columns(2)

# Normalize RSP for each oscillator
rsp_for_rsi = normalize_for_oscillator(view["RSP"], view["Composite_RSI"])
rsp_for_tsi = normalize_for_oscillator(view["RSP"], view["Composite_TSI"])
rsp_for_cci = normalize_for_oscillator(view["RSP"], view["Composite_CCI"])
rsp_for_bbp = normalize_for_oscillator(view["RSP"], view["Composite_BBP"])

with left:
    # RSI Chart with RSP overlay
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=view["Date"], y=view["Composite_RSI"], mode="lines", name="Composite RSI", line=dict(color="blue", width=2)))
    fig_rsi.add_trace(go.Scatter(x=view["Date"], y=rsp_for_rsi, mode="lines", name="RSP (normalized)", line=dict(color="red", width=1.5, dash="dot")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    fig_rsi.update_layout(title="Composite RSI vs RSP", height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_rsi, width="stretch")
    
    # CCI Chart with RSP overlay
    fig_cci = go.Figure()
    fig_cci.add_trace(go.Scatter(x=view["Date"], y=view["Composite_CCI"], mode="lines", name="Composite CCI", line=dict(color="blue", width=2)))
    fig_cci.add_trace(go.Scatter(x=view["Date"], y=rsp_for_cci, mode="lines", name="RSP (normalized)", line=dict(color="red", width=1.5, dash="dot")))
    fig_cci.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5)
    fig_cci.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_cci.add_hline(y=-100, line_dash="dash", line_color="green", opacity=0.5)
    fig_cci.update_layout(title="Composite CCI vs RSP", height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_cci, width="stretch")

with right:
    # TSI Chart with RSP overlay
    fig_tsi = go.Figure()
    fig_tsi.add_trace(go.Scatter(x=view["Date"], y=view["Composite_TSI"], mode="lines", name="Composite TSI", line=dict(color="blue", width=2)))
    fig_tsi.add_trace(go.Scatter(x=view["Date"], y=view["Composite_TSI_Signal"], mode="lines", name="TSI Signal", line=dict(color="lightblue", width=1.5)))
    fig_tsi.add_trace(go.Scatter(x=view["Date"], y=rsp_for_tsi, mode="lines", name="RSP (normalized)", line=dict(color="red", width=1.5, dash="dot")))
    fig_tsi.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_tsi.update_layout(title="Composite TSI vs RSP", height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_tsi, width="stretch")
    
    # BB% Chart with RSP overlay
    fig_bbp = go.Figure()
    fig_bbp.add_trace(go.Scatter(x=view["Date"], y=view["Composite_BBP"], mode="lines", name="Composite BB%", line=dict(color="blue", width=2)))
    fig_bbp.add_trace(go.Scatter(x=view["Date"], y=rsp_for_bbp, mode="lines", name="RSP (normalized)", line=dict(color="red", width=1.5, dash="dot")))
    fig_bbp.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.5)
    fig_bbp.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig_bbp.add_hline(y=0.0, line_dash="dash", line_color="green", opacity=0.5)
    fig_bbp.update_layout(title="Composite BB% vs RSP", height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_bbp, width="stretch")

# Combined oscillator additional charts if enabled
if st.session_state.combined_oscillator and "Combined_RSI" in view.columns:
    st.subheader("📊 Combined Oscillator + RSP Signal")
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig_combined_rsi = go.Figure()
        fig_combined_rsi.add_trace(go.Scatter(x=view["Date"], y=view["Combined_RSI"], mode="lines", name="Combined RSI", line=dict(color="purple", width=2)))
        fig_combined_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_combined_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
        fig_combined_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_combined_rsi.update_layout(title="Combined Signal RSI", height=300)
        st.plotly_chart(fig_combined_rsi, width="stretch")
    
    with col_right:
        fig_combined_tsi = go.Figure()
        fig_combined_tsi.add_trace(go.Scatter(x=view["Date"], y=view["Combined_TSI"], mode="lines", name="Combined TSI", line=dict(color="purple", width=2)))
        fig_combined_tsi.add_trace(go.Scatter(x=view["Date"], y=view["Combined_TSI_Signal"], mode="lines", name="Signal", line=dict(color="lightpurple", width=1.5)))
        fig_combined_tsi.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_combined_tsi.update_layout(title="Combined Signal TSI", height=300)
        st.plotly_chart(fig_combined_tsi, width="stretch")

# Optimization results display
if st.session_state.opt_results is not None:
    st.subheader("🔬 Optimization Results")
    opt = st.session_state.opt_results
    
    confidence = opt['confidence_score']
    if confidence > 0.7:
        st.success(f"✅ High Confidence Strategy ({confidence:.0%})")
    elif confidence > 0.5:
        st.warning(f"⚠️ Moderate Confidence ({confidence:.0%})")
    else:
        st.error(f"❌ Low Confidence ({confidence:.0%}) - Consider equal weights")
    
    col1, col2, col3, col4 = st.columns(4)
    rec_weights = opt['recommended_weights']
    col1.metric("Recommended NYAD", f"{rec_weights.get('NYAD', 0):.1%}")
    col2.metric("Recommended NYSI", f"{rec_weights.get('NYSI', 0):.1%}")
    col3.metric("Recommended NYHL", f"{rec_weights.get('NYHL', 0):.1%}")
    col4.metric("Recommended NYMO", f"{rec_weights.get('NYMO', 0):.1%}")
    
    perf = opt['expected_performance']
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Accuracy", f"{perf['directional_accuracy']:.1%}")
    col2.metric("Out-of-Sample Sharpe", f"{perf['sharpe_ratio']:.2f}")
    col3.metric("Profit Factor", f"{perf['profit_factor']:.2f}")
    
    # Weight stability chart
    if len(opt['weight_history']) > 1:
        st.subheader("Weight Stability Over Time")
        st.line_chart(opt['weight_history'])
        st.caption(f"Weight Stability Score: {opt['weight_stability']['stability_score']:.2f} (higher = more stable)")
    
    # Regime consistency
    if opt['regime_consistency']:
        st.subheader("Performance by Market Regime")
        regime_df = pd.DataFrame([opt['regime_consistency']]).T
        regime_df.columns = ['Blended Score']
        st.dataframe(regime_df, width="stretch")

# Historical lookup table for selected date
if lookup_date and not lookup_data.empty:
    st.subheader(f"📊 Oscillator Values for {lookup_date}")
    lookup_row = lookup_data.iloc[0]
    
    hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
    hist_col1.metric("RSI", f"{lookup_row.get('Composite_RSI', 'n/a'):.1f}" if pd.notna(lookup_row.get('Composite_RSI')) else "n/a")
    hist_col2.metric("TSI", f"{lookup_row.get('Composite_TSI', 'n/a'):.1f}" if pd.notna(lookup_row.get('Composite_TSI')) else "n/a")
    hist_col3.metric("CCI", f"{lookup_row.get('Composite_CCI', 'n/a'):.1f}" if pd.notna(lookup_row.get('Composite_CCI')) else "n/a")
    hist_col4.metric("BB%", f"{lookup_row.get('Composite_BBP', 'n/a'):.2f}" if pd.notna(lookup_row.get('Composite_BBP')) else "n/a")
    
    st.write(f"**Market State:** {lookup_row.get('State', 'N/A')}")
    
    if "RSP" in lookup_row:
        st.write(f"**RSP Close:** ${lookup_row['RSP']:.2f}" if pd.notna(lookup_row['RSP']) else "RSP: n/a")

# Latest values table
st.subheader("Latest breadth values")
display_df = pd.DataFrame({
    "Series": ["NYAD", "NYSI", "NYHL", "NYMO", "RSP"],
    "Latest Value": [
        latest.get("NYAD", np.nan),
        latest.get("NYSI", np.nan),
        latest.get("NYHL", np.nan),
        latest.get("NYMO", np.nan),
        latest.get("RSP", np.nan),
    ],
    "Weight %": [weights.get("NYAD", 0) * 100, weights.get("NYSI", 0) * 100, 
                 weights.get("NYHL", 0) * 100, weights.get("NYMO", 0) * 100, np.nan],
})
st.dataframe(display_df, width="stretch")

# Footer
st.markdown("""
---
**Methodology notes:**
- `NYSI` is treated as already cumulative
- `NYMO` is treated as daily, non-cumulative
- `NYAD` and `NYHL` default to being cumulated automatically
- Optimizer uses walk-forward testing with 5-day prediction horizon
- Blended objective: directional accuracy (30%) + Sharpe (25%) + Profit Factor (20%) + Calmar (15%) + median return (10%)
- Weights constrained to 5-50% each to prevent overfitting
- Red dotted line on oscillators shows normalized RSP for divergence detection
- Combined signal blends 70% breadth composite + 30% RSP momentum
""")
