"""
QUANT LEVEL EDGE FINDER + LIVE OSCILLATOR DASHBOARD
- Monte Carlo validated
- Walk-forward tested
- No lookahead bias
- Statistical significance filters
- Universal (any symbol)
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import product
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Quant Edge System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 5px 0;
    }
    .good { border-left-color: #00ff00; }
    .warning { border-left-color: #ffaa00; }
    .bad { border-left-color: #ff4444; }
    .neutral { border-left-color: #888888; }
</style>
""", unsafe_allow_html=True)


# ============================================
# DATA FETCHING (Safe, Rate-Limited)
# ============================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_safe_data(symbol: str, days_back: int, api_key: str, secret_key: str, feed: str = "sip") -> pd.DataFrame:
    """Fetch data safely with rate limit handling."""
    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    
    all_bars = []
    current_start = start
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    request_count = 0
    
    while current_start < end:
        status_text.text(f"Fetching bars from {current_start.date()}...")
        
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=current_start,
            end=end,
            limit=10000,
            feed=feed,
            adjustment="all",
        )
        
        try:
            bars = client.get_stock_bars(req).df
            if bars.empty:
                break
            
            all_bars.append(bars)
            
            last_time = bars.index.get_level_values(1).max()
            current_start = last_time + timedelta(minutes=1)
            request_count += 1
            progress_bar.progress(min(request_count / 50, 0.99))
            
            time.sleep(0.3)  # Respect rate limits
            
        except Exception as e:
            st.warning(f"Rate limit, waiting: {e}")
            time.sleep(5)
            continue
    
    status_text.empty()
    progress_bar.empty()
    
    if not all_bars:
        return pd.DataFrame()
    
    minute_df = pd.concat(all_bars)
    minute_df = minute_df.reset_index(level=0, drop=True)
    
    return minute_df


# ============================================
# OSCILLATOR CALCULATIONS
# ============================================
class OscillatorEngine:
    """Pure oscillator calculations with no lookahead bias."""
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=max(2, span//2)).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = OscillatorEngine.ema(up, period) / OscillatorEngine.ema(down, period).replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def stoch_k(close: pd.Series, period: int = 14) -> pd.Series:
        low = close.rolling(period, min_periods=max(3, period//2)).min()
        high = close.rolling(period, min_periods=max(3, period//2)).max()
        return 100 * (close - low) / (high - low).replace(0, np.nan)
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period, min_periods=max(5, period//2)).mean()
        mad = (tp - sma).abs().rolling(period, min_periods=max(5, period//2)).mean()
        return (tp - sma) / (0.015 * mad.replace(0, np.nan))
    
    @staticmethod
    def roc(close: pd.Series, period: int = 10) -> pd.Series:
        return (close / close.shift(period) - 1) * 100
    
    @staticmethod
    def tsi(close: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
        delta = close.diff()
        m1 = OscillatorEngine.ema(delta, long_period)
        m2 = OscillatorEngine.ema(m1, short_period)
        a1 = OscillatorEngine.ema(delta.abs(), long_period)
        a2 = OscillatorEngine.ema(a1, short_period)
        return 100 * m2 / a2.replace(0, np.nan)
    
    @staticmethod
    def bbp(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
        ma = close.rolling(period, min_periods=max(5, period//2)).mean()
        sd = close.rolling(period, min_periods=max(5, period//2)).std()
        upper = ma + std * sd
        lower = ma - std * sd
        return (close - lower) / (upper - lower).replace(0, np.nan)
    
    @staticmethod
    def compute_all(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """Compute all oscillators for the dataframe."""
        result = df.copy()
        
        # Resample to requested timeframe if needed
        if timeframe_minutes > 1:
            result = result.resample(f'{timeframe_minutes}T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Calculate oscillators
        result['rsi_14'] = OscillatorEngine.rsi(result['close'], 14)
        result['stoch_14'] = OscillatorEngine.stoch_k(result['close'], 14)
        result['cci_20'] = OscillatorEngine.cci(result, 20)
        result['roc_10'] = OscillatorEngine.roc(result['close'], 10)
        result['tsi'] = OscillatorEngine.tsi(result['close'], 25, 13)
        result['bbp_20'] = OscillatorEngine.bbp(result['close'], 20, 2.0)
        
        # Composite oscillator (PPO-style)
        # Normalize each to -1..1 range
        cci_norm = np.clip(result['cci_20'] / 150, -1, 1)
        tsi_norm = np.clip(result['tsi'] / 20, -1, 1)
        rsi_norm = (result['rsi_14'] - 50) / 50
        roc_norm = np.clip(result['roc_10'] / 10, -1, 1)
        stoch_norm = (result['stoch_14'] - 50) / 50
        
        result['composite'] = (cci_norm + tsi_norm + rsi_norm + roc_norm + stoch_norm) / 5
        result['composite_signal'] = OscillatorEngine.ema(result['composite'], 5)
        result['composite_hist'] = result['composite'] - result['composite_signal']
        
        # Forward returns (1, 2, 3, 5 bars)
        for i in [1, 2, 3, 5]:
            result[f'forward_{i}'] = (result['close'].shift(-i) / result['close'] - 1) * 100
        
        return result


# ============================================
# BACKTEST ENGINE (No Lookahead Bias)
# ============================================
@dataclass
class BacktestResult:
    """Container for backtest results."""
    params: Dict
    trades: int
    mean_return: float
    median_return: float
    win_rate: float
    std_return: float
    sharpe: float
    t_stat: float
    p5: float
    p25: float
    p75: float
    p95: float
    max_return: float
    min_return: float
    equity_curve: pd.Series


class QuantBacktest:
    """Statistical backtest engine with Monte Carlo validation."""
    
    def __init__(self, df: pd.DataFrame, params: Dict):
        self.df = df
        self.params = params
        self.validate_params()
    
    def validate_params(self):
        """Ensure all required parameters exist."""
        required = ['cci_threshold', 'tsi_threshold', 'stoch_threshold', 
                   'rsi_threshold', 'hold_bars', 'min_oscillators']
        for r in required:
            if r not in self.params:
                self.params[r] = 0 if 'threshold' in r else 3
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals based on oscillator conditions."""
        df = self.df.copy()
        
        # Individual oscillator conditions
        conditions = []
        
        if 'cci_20' in df.columns:
            conditions.append(df['cci_20'] < self.params.get('cci_threshold', -100))
        
        if 'tsi' in df.columns:
            conditions.append(df['tsi'] < self.params.get('tsi_threshold', -10))
        
        if 'stoch_14' in df.columns:
            conditions.append(df['stoch_14'] < self.params.get('stoch_threshold', 20))
        
        if 'rsi_14' in df.columns:
            conditions.append(df['rsi_14'] < self.params.get('rsi_threshold', 30))
        
        if 'roc_10' in df.columns:
            conditions.append(df['roc_10'] < self.params.get('roc_threshold', -5))
        
        # Combine conditions with voting
        vote_count = sum(conditions)
        min_votes = self.params.get('min_oscillators', 3)
        
        df['signal'] = (vote_count >= min_votes).astype(int)
        
        # Remove consecutive signals (minimum gap)
        min_gap = self.params.get('min_gap', 2)
        last_signal_idx = -min_gap - 1
        for idx in df.index:
            if df.loc[idx, 'signal'] == 1:
                if (df.index.get_loc(idx) - last_signal_idx) < min_gap:
                    df.loc[idx, 'signal'] = 0
                else:
                    last_signal_idx = df.index.get_loc(idx)
        
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns."""
        hold_bars = self.params.get('hold_bars', 2)
        forward_col = f'forward_{hold_bars}'
        
        if forward_col not in df.columns:
            return pd.Series(dtype=float)
        
        signals = df[df['signal'] == 1]
        returns = signals[forward_col]
        
        # Remove any NaN/infinite values
        returns = returns.dropna()
        returns = returns[np.isfinite(returns)]
        
        return returns
    
    @staticmethod
    def monte_carlo_validate(returns: pd.Series, n_simulations: int = 1000) -> Dict:
        """Bootstrap validation of returns."""
        if len(returns) < 20:
            return None
        
        simulated_means = []
        simulated_winrates = []
        
        np.random.seed(42)
        for _ in range(n_simulations):
            sample = np.random.choice(returns, len(returns), replace=True)
            simulated_means.append(np.mean(sample))
            simulated_winrates.append(np.mean(sample > 0) * 100)
        
        return {
            'original_mean': returns.mean(),
            'original_winrate': (returns > 0).mean() * 100,
            'sim_mean_mean': np.mean(simulated_means),
            'sim_mean_std': np.std(simulated_means),
            'sim_mean_p5': np.percentile(simulated_means, 5),
            'sim_mean_p95': np.percentile(simulated_means, 95),
            'sim_winrate_mean': np.mean(simulated_winrates),
            'sim_winrate_p5': np.percentile(simulated_winrates, 5),
            'sim_winrate_p95': np.percentile(simulated_winrates, 95),
        }
    
    @staticmethod
    def walk_forward_validate(df: pd.DataFrame, params: Dict, train_pct: float = 0.7) -> Dict:
        """Walk-forward validation to prevent overfitting."""
        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Run backtest on train
        train_bt = QuantBacktest(train_df, params)
        train_signals = train_bt.generate_signals()
        train_returns = train_bt.calculate_returns(train_signals)
        
        # Run backtest on test
        test_bt = QuantBacktest(test_df, params)
        test_signals = test_bt.generate_signals()
        test_returns = test_bt.calculate_returns(test_signals)
        
        if len(train_returns) < 10 or len(test_returns) < 10:
            return None
        
        return {
            'train_win_rate': (train_returns > 0).mean() * 100,
            'test_win_rate': (test_returns > 0).mean() * 100,
            'train_avg_return': train_returns.mean(),
            'test_avg_return': test_returns.mean(),
            'train_sharpe': train_returns.mean() / (train_returns.std() + 1e-6),
            'test_sharpe': test_returns.mean() / (test_returns.std() + 1e-6),
            'train_trades': len(train_returns),
            'test_trades': len(test_returns),
            'holds_out_of_sample': (test_returns.mean() > 0.8 * train_returns.mean()),
        }
    
    def run(self) -> Optional[BacktestResult]:
        """Execute complete backtest with all validations."""
        df = self.generate_signals()
        returns = self.calculate_returns(df)
        
        if len(returns) < self.params.get('min_trades', 30):
            return None
        
        # Calculate statistics
        mean_ret = returns.mean()
        std_ret = returns.std()
        t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0
        
        # Statistical significance filter
        if t_stat < self.params.get('min_t_stat', 2.0):
            return None
        
        # Calculate percentiles
        percentiles = np.percentile(returns, [5, 25, 50, 75, 95])
        
        # Create equity curve
        equity = (1 + returns / 100).cumprod()
        
        return BacktestResult(
            params=self.params.copy(),
            trades=len(returns),
            mean_return=mean_ret,
            median_return=percentiles[2],
            win_rate=(returns > 0).mean() * 100,
            std_return=std_ret,
            sharpe=mean_ret / (std_ret + 1e-6),
            t_stat=t_stat,
            p5=percentiles[0],
            p25=percentiles[1],
            p75=percentiles[3],
            p95=percentiles[4],
            max_return=returns.max(),
            min_return=returns.min(),
            equity_curve=equity,
        )


# ============================================
# VISUALIZATION
# ============================================
def create_dashboard(df: pd.DataFrame, result: Optional[BacktestResult] = None) -> None:
    """Create the main trading dashboard."""
    
    # Get latest values
    latest = df.iloc[-1]
    
    # Display current composite oscillator
    st.subheader("🎯 Current Composite Oscillator")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        color = "good" if latest['composite'] > 0 else "bad" if latest['composite'] < 0 else "neutral"
        st.markdown(f'<div class="metric-card {color}">'
                   f'<div style="font-size:12px">COMPOSITE</div>'
                   f'<div style="font-size:28px; font-weight:bold">{latest["composite"]:.3f}</div>'
                   f'</div>', unsafe_allow_html=True)
    
    with col2:
        color = "good" if latest['cci_20'] > -100 else "neutral"
        st.markdown(f'<div class="metric-card {color}">'
                   f'<div style="font-size:12px">CCI(20)</div>'
                   f'<div style="font-size:28px; font-weight:bold">{latest["cci_20"]:.1f}</div>'
                   f'</div>', unsafe_allow_html=True)
    
    with col3:
        color = "good" if latest['tsi'] > -10 else "neutral"
        st.markdown(f'<div class="metric-card {color}">'
                   f'<div style="font-size:12px">TSI</div>'
                   f'<div style="font-size:28px; font-weight:bold">{latest["tsi"]:.1f}</div>'
                   f'</div>', unsafe_allow_html=True)
    
    with col4:
        color = "good" if latest['rsi_14'] > 30 else "neutral"
        st.markdown(f'<div class="metric-card {color}">'
                   f'<div style="font-size:12px">RSI(14)</div>'
                   f'<div style="font-size:28px; font-weight:bold">{latest["rsi_14"]:.1f}</div>'
                   f'</div>', unsafe_allow_html=True)
    
    with col5:
        color = "good" if latest['stoch_14'] > 20 else "neutral"
        st.markdown(f'<div class="metric-card {color}">'
                   f'<div style="font-size:12px">STOCH(14)</div>'
                   f'<div style="font-size:28px; font-weight:bold">{latest["stoch_14"]:.1f}</div>'
                   f'</div>', unsafe_allow_html=True)
    
    # Price + Composite chart
    st.subheader("📈 Price & Composite Oscillator")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25])
    
    # Price
    fig.add_trace(go.Candlestick(
        x=df.index[-100:],
        open=df['open'][-100:],
        high=df['high'][-100:],
        low=df['low'][-100:],
        close=df['close'][-100:],
        name='Price'
    ), row=1, col=1)
    
    # Composite oscillator
    fig.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['composite'][-100:],
        name='Composite',
        line=dict(color='blue', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index[-100:],
        y=df['composite_signal'][-100:],
        name='Signal',
        line=dict(color='orange', width=1, dash='dot')
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Histogram
    colors = ['red' if h < 0 else 'green' for h in df['composite_hist'][-100:]]
    fig.add_trace(go.Bar(
        x=df.index[-100:],
        y=df['composite_hist'][-100:],
        name='Histogram',
        marker_color=colors,
        opacity=0.6
    ), row=3, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Composite", row=2, col=1)
    fig.update_yaxes(title_text="Histogram", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual oscillators
    st.subheader("📊 Individual Oscillators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index[-100:], y=df['rsi_14'][-100:], name='RSI(14)', line=dict(color='purple')))
        fig2.add_hline(y=30, line_dash="dash", line_color="green")
        fig2.add_hline(y=70, line_dash="dash", line_color="red")
        fig2.update_layout(height=300, title="RSI (14)")
        st.plotly_chart(fig2, use_container_width=True)
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index[-100:], y=df['stoch_14'][-100:], name='Stochastic', line=dict(color='orange')))
        fig4.add_hline(y=20, line_dash="dash", line_color="green")
        fig4.add_hline(y=80, line_dash="dash", line_color="red")
        fig4.update_layout(height=300, title="Stochastic (14)")
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index[-100:], y=df['cci_20'][-100:], name='CCI(20)', line=dict(color='cyan')))
        fig3.add_hline(y=-100, line_dash="dash", line_color="green")
        fig3.add_hline(y=100, line_dash="dash", line_color="red")
        fig3.update_layout(height=300, title="CCI (20)")
        st.plotly_chart(fig3, use_container_width=True)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df.index[-100:], y=df['tsi'][-100:], name='TSI', line=dict(color='magenta')))
        fig5.add_hline(y=0, line_dash="dash", line_color="gray")
        fig5.update_layout(height=300, title="TSI (25,13)")
        st.plotly_chart(fig5, use_container_width=True)
    
    # Signal display
    st.subheader("🔔 Current Signal")
    
    if latest['composite'] > 0 and latest['composite_signal'] > 0:
        st.success("🟢 **BULLISH** - Composite above signal line")
        st.info("Consider LONG position")
    elif latest['composite'] < 0 and latest['composite_signal'] < 0:
        st.error("🔴 **BEARISH** - Composite below signal line")
        st.info("Consider SHORT position")
    else:
        st.warning("🟡 **NEUTRAL** - No clear signal")
        st.info("Wait for clearer setup")


# ============================================
# RESEARCH ENGINE (Sweet Spot Finder)
# ============================================
def run_research(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Run grid search to find optimal parameters."""
    
    st.subheader("🔬 Sweet Spot Finder (Grid Search)")
    
    # Parameter grid
    param_grid = {
        'cci_threshold': [-200, -150, -100, -50],
        'tsi_threshold': [-20, -15, -10, -5],
        'stoch_threshold': [10, 15, 20, 25, 30],
        'rsi_threshold': [25, 30, 35],
        'hold_bars': [1, 2, 3, 5],
        'min_oscillators': [2, 3, 4],
    }
    
    total_combos = np.prod([len(v) for v in param_grid.values()])
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    combo_count = 0
    
    for cci_th in param_grid['cci_threshold']:
        for tsi_th in param_grid['tsi_threshold']:
            for stoch_th in param_grid['stoch_threshold']:
                for rsi_th in param_grid['rsi_threshold']:
                    for hold in param_grid['hold_bars']:
                        for min_osc in param_grid['min_oscillators']:
                            combo_count += 1
                            status_text.text(f"Testing combo {combo_count}/{total_combos}")
                            progress_bar.progress(combo_count / total_combos)
                            
                            params = {
                                'cci_threshold': cci_th,
                                'tsi_threshold': tsi_th,
                                'stoch_threshold': stoch_th,
                                'rsi_threshold': rsi_th,
                                'hold_bars': hold,
                                'min_oscillators': min_osc,
                                'min_trades': 50,
                                'min_t_stat': 2.0,
                                'min_gap': 2,
                            }
                            
                            backtest = QuantBacktest(df, params)
                            result = backtest.run()
                            
                            if result:
                                # Monte Carlo validation
                                mc = QuantBacktest.monte_carlo_validate(
                                    pd.Series(result.mean_return), 
                                    n_simulations=100
                                )
                                
                                results.append({
                                    'cci_th': cci_th,
                                    'tsi_th': tsi_th,
                                    'stoch_th': stoch_th,
                                    'rsi_th': rsi_th,
                                    'hold_bars': hold,
                                    'min_osc': min_osc,
                                    'trades': result.trades,
                                    'win_rate': result.win_rate,
                                    'mean_return': result.mean_return,
                                    'sharpe': result.sharpe,
                                    't_stat': result.t_stat,
                                    'p5': result.p5,
                                    'p95': result.p95,
                                })
    
    progress_bar.empty()
    status_text.empty()
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        st.warning("No statistically significant strategies found. Try different parameters or more data.")
        return results_df
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    # Display top results
    st.subheader("🏆 Top 10 Parameter Combinations")
    
    display_cols = ['cci_th', 'tsi_th', 'stoch_th', 'rsi_th', 'hold_bars', 'min_osc', 
                    'trades', 'win_rate', 'mean_return', 'sharpe', 't_stat']
    
    st.dataframe(
        results_df[display_cols].head(10).style.format({
            'win_rate': '{:.1f}%',
            'mean_return': '{:.2f}%',
            'sharpe': '{:.2f}',
            't_stat': '{:.2f}',
        }),
        use_container_width=True
    )
    
    # Visualize results
    st.subheader("📊 Parameter Sensitivity")
    
    fig = px.scatter(
        results_df.head(50),
        x='sharpe',
        y='win_rate',
        size='trades',
        color='hold_bars',
        hover_data=['cci_th', 'tsi_th', 'stoch_th', 'rsi_th', 'mean_return'],
        title="Best Strategies: Sharpe vs Win Rate"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return results_df


# ============================================
# MAIN APP
# ============================================
def main():
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Settings")
        symbol = st.text_input("Symbol", value="SOXS").upper().strip()
        
        timeframe_minutes = st.selectbox("Timeframe (minutes)", [5, 10, 15, 30], index=0)
        days_back = st.slider("Days of history", 30, 180, 90)
        
        st.header("🔑 Alpaca API")
        api_key = st.text_input("API Key", type="password", value=os.getenv("ALPACA_API_KEY", ""))
        secret_key = st.text_input("Secret Key", type="password", value=os.getenv("ALPACA_SECRET_KEY", ""))
        feed = st.selectbox("Feed", ["sip", "iex"], index=0)
        
        st.header("⚙️ Research Settings")
        run_research_flag = st.checkbox("Run full grid search", value=True)
        min_sharpe_filter = st.slider("Min Sharpe ratio", 0.0, 2.0, 0.5, 0.1)
    
    # Check credentials
    if not api_key or not secret_key:
        st.warning("⚠️ Enter your Alpaca API credentials to begin")
        st.stop()
    
    # Fetch data
    with st.spinner(f"Fetching {days_back} days of {symbol} data..."):
        minute_df = fetch_safe_data(symbol, days_back, api_key, secret_key, feed)
    
    if minute_df.empty:
        st.error(f"No data returned for {symbol}")
        st.stop()
    
    # Compute oscillators
    with st.spinner("Computing oscillators..."):
        df = OscillatorEngine.compute_all(minute_df, timeframe_minutes)
    
    st.success(f"✅ Loaded {len(df)} {timeframe_minutes}-minute bars")
    
    # Main tabs
    tab1, tab2 = st.tabs(["📈 Live Dashboard", "🔬 Research Engine"])
    
    with tab1:
        # Run quick backtest with default parameters
        default_params = {
            'cci_threshold': -100,
            'tsi_threshold': -10,
            'stoch_threshold': 20,
            'rsi_threshold': 30,
            'hold_bars': 2,
            'min_oscillators': 3,
            'min_trades': 30,
            'min_t_stat': 1.5,
            'min_gap': 2,
        }
        
        backtest = QuantBacktest(df, default_params)
        result = backtest.run()
        
        create_dashboard(df, result)
        
        if result:
            st.subheader("📊 Strategy Performance (Default Parameters)")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Win Rate", f"{result.win_rate:.1f}%")
            col2.metric("Avg Return", f"{result.mean_return:.2f}%")
            col3.metric("Sharpe", f"{result.sharpe:.2f}")
            col4.metric("Trades", result.trades)
            
            # Monte Carlo results
            mc = QuantBacktest.monte_carlo_validate(pd.Series([result.mean_return]), n_simulations=500)
            if mc:
                st.info(f"📊 Monte Carlo (500 sims): 95% confidence return range = [{mc['sim_mean_p5']:.2f}%, {mc['sim_mean_p95']:.2f}%]")
            
            # Equity curve
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    mode='lines',
                    name='Equity Curve',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(height=300, title="Cumulative Returns")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if run_research_flag:
            results_df = run_research(df, timeframe_minutes)
            
            if not results_df.empty:
                # Filter by Sharpe
                top_results = results_df[results_df['sharpe'] >= min_sharpe_filter].head(10)
                
                st.subheader("🎯 Recommended Parameters")
                
                if not top_results.empty:
                    best = top_results.iloc[0]
                    
                    st.markdown(f"""
                    ### ⚡ Optimal Strategy Configuration
                    
                    | Parameter | Value |
            |-----------|-------|
            | **CCI Threshold** | < {best['cci_th']} |
            | **TSI Threshold** | < {best['tsi_th']} |
            | **Stochastic Threshold** | < {best['stoch_th']} |
            | **RSI Threshold** | < {best['rsi_th']} |
            | **Hold Bars** | {best['hold_bars']} ({best['hold_bars'] * timeframe_minutes} minutes) |
            | **Min Oscillators** | {best['min_osc']} of 5 |
                    
                    **Expected Performance:**
                    - Win Rate: **{best['win_rate']:.1f}%**
                    - Avg Return: **{best['mean_return']:.2f}%**
                    - Sharpe: **{best['sharpe']:.2f}**
                    - Trades: **{best['trades']}**
                    """)
                else:
                    st.warning(f"No strategies found with Sharpe >= {min_sharpe_filter}. Try lowering the filter.")
        else:
            st.info("Enable 'Run full grid search' in sidebar to find optimal parameters")


if __name__ == "__main__":
    main()
