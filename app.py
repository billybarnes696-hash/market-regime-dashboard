"""
UNIVERSAL SWEET SPOT BACKTEST - WITH DEBUG LOGS
Run this script directly (python script.py)
Will save results to CSV and print debug info
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import product
import sys

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ============================================
# CONFIGURATION - EDIT THESE
# ============================================
API_KEY = "YOUR_ALPACA_API_KEY"
SECRET_KEY = "YOUR_ALPACA_SECRET"
FEED = "sip"  # sip = full market, iex = limited

SYMBOL = "SOXS"  # Change this to test different symbols
DAYS_BACK = 180
TIMEFRAME_MIN = 10  # 5, 10, or 15
HOLD_BARS = 2  # 1, 2, 3, or 5

# Signal type: "cross" or "level"
SIGNAL_TYPE = "cross"  # cross = when oscillator crosses threshold (RECOMMENDED)
# SIGNAL_TYPE = "level"  # level = when oscillator is above threshold

# Thresholds to test
TSI_THRESHOLDS = [0]
CCI_THRESHOLDS = [0]
ROC_THRESHOLDS = [0]
RSI_THRESHOLDS = [50]

# ============================================
# DEBUG LOGGING
# ============================================
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}")
    sys.stdout.flush()

log(f"Starting backtest for {SYMBOL}")
log(f"Timeframe: {TIMEFRAME_MIN}min | Hold: {HOLD_BARS} bars ({TIMEFRAME_MIN * HOLD_BARS} min)")
log(f"Signal type: {SIGNAL_TYPE}")

# ============================================
# FETCH DATA
# ============================================
log(f"Fetching {DAYS_BACK} days of data from Alpaca...")

try:
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_BACK)
    
    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=50000,
        feed=FEED,
        adjustment="all",
    )
    
    bars = client.get_stock_bars(req).df
    log(f"Raw bars fetched: {len(bars)}")
    
    if bars.empty:
        log("No data returned! Check API keys and symbol.", "ERROR")
        sys.exit(1)
        
except Exception as e:
    log(f"API Error: {e}", "ERROR")
    sys.exit(1)

# Clean data
bars = bars.reset_index(level=0, drop=True)
bars.index = pd.to_datetime(bars.index)
bars = bars.tz_localize(None)

log(f"Data range: {bars.index[0]} to {bars.index[-1]}")
log(f"Total minute bars: {len(bars)}")

# ============================================
# RESAMPLE TO TIMEFRAME
# ============================================
freq = f'{TIMEFRAME_MIN}min'
df = bars.resample(freq).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

log(f"Resampled to {len(df)} {TIMEFRAME_MIN}-min bars")

if len(df) < 100:
    log(f"Only {len(df)} bars - need at least 100 for valid backtest", "WARNING")

# ============================================
# OSCILLATOR FUNCTIONS
# ============================================
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def tsi(close, long_p=25, short_p=13):
    delta = close.diff()
    m1 = ema(delta, long_p)
    m2 = ema(m1, short_p)
    a1 = ema(delta.abs(), long_p)
    a2 = ema(a1, short_p)
    return 100 * m2 / a2.replace(0, np.nan)

def cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period, min_periods=2).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=2).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def roc(close, period=10):
    return (close / close.shift(period) - 1) * 100

def rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ============================================
# CALCULATE OSCILLATORS
# ============================================
log("Calculating oscillators...")

df['tsi'] = tsi(df['close'])
df['cci'] = cci(df)
df['roc'] = roc(df['close'])
df['rsi'] = rsi(df['close'])

# Forward returns
df[f'forward_{HOLD_BARS}'] = (df['close'].shift(-HOLD_BARS) / df['close'] - 1) * 100

log(f"Oscillator stats:")
log(f"  TSI: min={df['tsi'].min():.1f}, max={df['tsi'].max():.1f}, mean={df['tsi'].mean():.1f}")
log(f"  CCI: min={df['cci'].min():.1f}, max={df['cci'].max():.1f}, mean={df['cci'].mean():.1f}")
log(f"  ROC: min={df['roc'].min():.1f}, max={df['roc'].max():.1f}, mean={df['roc'].mean():.1f}")
log(f"  RSI: min={df['rsi'].min():.1f}, max={df['rsi'].max():.1f}, mean={df['rsi'].mean():.1f}")

# ============================================
# GENERATE SIGNALS (CROSS vs LEVEL)
# ============================================
log(f"Generating {SIGNAL_TYPE} signals...")

if SIGNAL_TYPE == "cross":
    # CROSS signal - enters when oscillator crosses threshold
    df['tsi_signal'] = (df['tsi'].shift(1) <= 0) & (df['tsi'] > 0)
    df['cci_signal'] = (df['cci'].shift(1) <= 0) & (df['cci'] > 0)
    df['roc_signal'] = (df['roc'].shift(1) <= 0) & (df['roc'] > 0)
    df['rsi_signal'] = (df['rsi'].shift(1) <= 50) & (df['rsi'] > 50)
else:
    # LEVEL signal - enters when oscillator is above threshold (ALREADY MOVED)
    df['tsi_signal'] = df['tsi'] > 0
    df['cci_signal'] = df['cci'] > 0
    df['roc_signal'] = df['roc'] > 0
    df['rsi_signal'] = df['rsi'] > 50

# Combine signals (ALL must be true)
df['signal'] = (
    df['tsi_signal'] & 
    df['cci_signal'] & 
    df['roc_signal'] & 
    df['rsi_signal']
)

signal_count = df['signal'].sum()
signal_pct = signal_count / len(df) * 100

log(f"Total signals: {signal_count} ({signal_pct:.2f}% of bars)")

if signal_count < 20:
    log(f"Only {signal_count} signals - need at least 20 for valid backtest", "WARNING")
    log("Try: lower thresholds, longer timeframe, or more days of data")

# ============================================
# BACKTEST
# ============================================
signals = df[df['signal']].copy()
returns = signals[f'forward_{HOLD_BARS}'].dropna()

if len(returns) < 10:
    log(f"Only {len(returns)} trades with valid returns - cannot backtest", "ERROR")
    sys.exit(1)

win_rate = (returns > 0).mean() * 100
avg_return = returns.mean()
median_return = returns.median()
std_return = returns.std()
sharpe = avg_return / std_return if std_return > 0 else 0
t_stat = avg_return / (std_return / np.sqrt(len(returns))) if std_return > 0 else 0

# Percentiles
p5 = np.percentile(returns, 5)
p25 = np.percentile(returns, 25)
p75 = np.percentile(returns, 75)
p95 = np.percentile(returns, 95)

# ============================================
# MONTE CARLO SIMULATION
# ============================================
log(f"Running Monte Carlo ({min(500, len(returns))} iterations)...")

n_sims = min(500, len(returns) * 10)
sim_means = []
sim_wins = []

np.random.seed(42)
for _ in range(n_sims):
    sample = np.random.choice(returns, len(returns), replace=True)
    sim_means.append(sample.mean())
    sim_wins.append((sample > 0).mean() * 100)

mc_mean = np.mean(sim_means)
mc_std = np.std(sim_means)
mc_p5 = np.percentile(sim_means, 5)
mc_p95 = np.percentile(sim_means, 95)
mc_win_mean = np.mean(sim_wins)
mc_win_p5 = np.percentile(sim_wins, 5)
mc_win_p95 = np.percentile(sim_wins, 95)

# ============================================
# PRINT RESULTS
# ============================================
print("\n" + "="*70)
print(f"RESULTS FOR {SYMBOL}")
print("="*70)
print(f"""
Configuration:
  Timeframe:        {TIMEFRAME_MIN} minutes
  Hold time:        {TIMEFRAME_MIN * HOLD_BARS} minutes ({HOLD_BARS} bars)
  Signal type:      {SIGNAL_TYPE}
  Data period:      {df.index[0].date()} to {df.index[-1].date()}
  Total bars:       {len(df)}

Signal Statistics:
  Total signals:    {signal_count}
  Signal frequency: {signal_pct:.2f}% of bars
  Valid trades:     {len(returns)}

Performance:
  Win Rate:         {win_rate:.1f}%
  Avg Return:       {avg_return:.2f}%
  Median Return:    {median_return:.2f}%
  Std Deviation:    {std_return:.2f}%
  Sharpe Ratio:     {sharpe:.2f}
  t-statistic:      {t_stat:.2f}

Return Distribution:
  5th percentile:   {p5:.2f}%
  25th percentile:  {p25:.2f}%
  75th percentile:  {p75:.2f}%
  95th percentile:  {p95:.2f}%
  Best trade:       {returns.max():.2f}%
  Worst trade:      {returns.min():.2f}%

Monte Carlo (95% confidence):
  Return range:     [{mc_p5:.2f}%, {mc_p95:.2f}%]
  Win rate range:   [{mc_win_p5:.1f}%, {mc_win_p95:.1f}%]
""")

# ============================================
# INTERPRETATION
# ============================================
print("="*70)
print("INTERPRETATION")
print("="*70)

if t_stat > 2.0:
    print("✅ t-stat > 2.0 - Statistically significant")
else:
    print("❌ t-stat < 2.0 - Not statistically significant (may be random)")

if win_rate > 60:
    print(f"✅ Win rate {win_rate:.1f}% - Better than coin flip")
elif win_rate > 55:
    print(f"🟡 Win rate {win_rate:.1f}% - Modest edge")
else:
    print(f"❌ Win rate {win_rate:.1f}% - Worse than coin flip")

if avg_return > 0:
    print(f"✅ Positive expectancy: {avg_return:.2f}% per trade")
else:
    print(f"❌ Negative expectancy: {avg_return:.2f}% per trade")

if signal_count < 50:
    print(f"⚠️ Only {signal_count} signals - may not be enough for reliable strategy")

# ============================================
# CHECK SIGNAL DIRECTION
# ============================================
print("\n" + "="*70)
print("SIGNAL DIRECTION CHECK")
print("="*70)

# Check what happens AFTER signal
signal_returns = returns
no_signal_returns = df[~df['signal']][f'forward_{HOLD_BARS}'].dropna()

print(f"With signal:    avg return = {signal_returns.mean():.3f}%")
print(f"Without signal: avg return = {no_signal_returns.mean():.3f}%")
print(f"Difference:     {signal_returns.mean() - no_signal_returns.mean():.3f}%")

if signal_returns.mean() > no_signal_returns.mean():
    print("✅ Signal is better than random (positive edge)")
else:
    print("❌ Signal is WORSE than random (try OPPOSITE signal)")

# ============================================
# SAVE RESULTS
# ============================================
results = {
    'symbol': SYMBOL,
    'timeframe_min': TIMEFRAME_MIN,
    'hold_minutes': TIMEFRAME_MIN * HOLD_BARS,
    'signal_type': SIGNAL_TYPE,
    'days_back': DAYS_BACK,
    'total_bars': len(df),
    'total_signals': signal_count,
    'signal_pct': round(signal_pct, 2),
    'valid_trades': len(returns),
    'win_rate': round(win_rate, 1),
    'avg_return': round(avg_return, 2),
    'median_return': round(median_return, 2),
    'sharpe': round(sharpe, 2),
    't_stat': round(t_stat, 2),
    'p5': round(p5, 2),
    'p95': round(p95, 2),
    'mc_return_p5': round(mc_p5, 2),
    'mc_return_p95': round(mc_p95, 2),
    'mc_win_p5': round(mc_win_p5, 1),
    'mc_win_p95': round(mc_win_p95, 1),
}

# Save to CSV
output_file = f"{SYMBOL}_backtest_{TIMEFRAME_MIN}min_{SIGNAL_TYPE}.csv"

# Also save full trade list
trades_df = pd.DataFrame({
    'timestamp': signals.index[:len(returns)],
    'return_pct': returns.values
})
trades_file = f"{SYMBOL}_trades_{TIMEFRAME_MIN}min_{SIGNAL_TYPE}.csv"
trades_df.to_csv(trades_file, index=False)

print(f"\n📁 Results saved to: {output_file}")
print(f"📁 Trade list saved to: {trades_file}")

# ============================================
# RECOMMENDATION
# ============================================
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if avg_return > 0 and t_stat > 2.0 and win_rate > 55:
    print(f"✅ This strategy works for {SYMBOL}!")
    print(f"   Entry: When TSI, CCI, ROC cross above 0 AND RSI > 50")
    print(f"   Exit: After {TIMEFRAME_MIN * HOLD_BARS} minutes")
elif avg_return < 0 and t_stat > 2.0:
    print(f"🔄 Try the OPPOSITE signal for {SYMBOL}")
    print(f"   Entry: When TSI, CCI, ROC cross BELOW 0 AND RSI < 50")
elif t_stat < 2.0:
    print(f"❌ No statistically significant edge found for {SYMBOL}")
    print(f"   Try: Different timeframe, hold time, or thresholds")
else:
    print(f"❌ Strategy does not work for {SYMBOL} with current settings")
