#!/usr/bin/env python3
"""
UNIVERSAL OSCILLATOR BACKTEST - INSTITUTIONAL v2.0
Run: python soxs_backtest_debug.py
Features: Score-based signals, Regime filters, Slippage-aware Monte Carlo, Full debugging
"""

import os, sys, time, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ============================================
# CONFIGURATION - EDIT THESE
# ============================================
API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET")
FEED = "sip"  # sip = full market, iex = limited

SYMBOL = "SOXS"
DAYS_BACK = 180
TIMEFRAME_MIN = 10
HOLD_BARS = 3  # 3 bars × 10 min = 30 min hold
MIN_TRADES = 20
SLIPPAGE_PCT = 0.008  # 0.8% realistic for leveraged ETFs
MC_ITERATIONS = 500
CONFIDENCE_LEVEL = 95

# Score engine & regime filters
SCORE_THRESHOLD = 0.6
MIN_VOL_PCT = 0.4  # 0.4% ATR floor (filters dead/chop markets)

# ============================================
# DEBUGGING ENGINE
# ============================================
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {level}: {msg}")
    sys.stdout.flush()

# ============================================
# OSCILLATOR CALCULATIONS
# ============================================
def calc_oscillators(df):
    c = df['close']
    h, l = df['high'], df['low']
    
    # RSI
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # CCI
    tp = (h + l + c) / 3
    sma = tp.rolling(20).mean()
    mad = (tp - sma).abs().rolling(20).mean()
    df['cci'] = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    
    # ROC
    df['roc'] = (c / c.shift(10) - 1) * 100
    
    # TSI
    mom = c.diff()
    m1 = mom.ewm(span=25).mean()
    m2 = m1.ewm(span=13).mean()
    a1 = mom.abs().ewm(span=25).mean()
    a2 = a1.ewm(span=13).mean()
    df['tsi'] = 100 * m2 / a2.replace(0, np.nan)
    
    # Stoch
    lo, hi = c.rolling(14).min(), c.rolling(14).max()
    df['stoch'] = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    
    return df

# ============================================
# SIGNAL & REGIME ENGINE
# ============================================
def generate_signals(df):
    # 1. Component signals (crossing above threshold)
    df['sig_rsi']   = (df['rsi'].shift(1) <= 50) & (df['rsi'] > 50)
    df['sig_cci']   = (df['cci'].shift(1) <= 0)  & (df['cci'] > 0)
    df['sig_roc']   = (df['roc'].shift(1) <= 0)  & (df['roc'] > 0)
    df['sig_tsi']   = (df['tsi'].shift(1) <= 0)  & (df['tsi'] > 0)
    df['sig_stoch'] = (df['stoch'].shift(1) <= 50) & (df['stoch'] > 50)
    
    # 2. Score engine (0.0 to 1.0)
    df['score'] = (df['sig_rsi'].astype(int) + df['sig_cci'].astype(int) + 
                   df['sig_roc'].astype(int) + df['sig_tsi'].astype(int) + 
                   df['sig_stoch'].astype(int)) / 5.0
    
    # 3. Regime filters
    df['atr'] = np.maximum(df['high'] - df['low'], 
                  np.maximum((df['high'] - df['close'].shift(1)).abs(),
                             (df['low'] - df['close'].shift(1)).abs())).rolling(14).mean()
    df['vol_pct'] = (df['atr'] / df['close']) * 100
    
    df['trend_ok'] = df['close'] > df['close'].rolling(50).mean()  # Above 50-bar trend
    df['vol_ok']   = df['vol_pct'] >= MIN_VOL_PCT
    
    # 4. Final signal: Score threshold + regime filters
    df['signal'] = (df['score'] >= SCORE_THRESHOLD) & df['trend_ok'] & df['vol_ok']
    return df

# ============================================
# MONTE CARLO VALIDATION
# ============================================
def monte_carlo(returns, n_iter=MC_ITERATIONS, slip=SLIPPAGE_PCT, conf=CONFIDENCE_LEVEL):
    if len(returns) < 5: return None
    boots_mean, boots_wr = [], []
    for _ in range(n_iter):
        samp = np.random.choice(returns, len(returns), replace=True)
        net = samp - (slip * 100)  # Apply slippage per trade
        boots_mean.append(net.mean())
        boots_wr.append((net > 0).mean() * 100)
    
    alpha = (100 - conf) / 200
    return {
        'mean_ci': (np.percentile(boots_mean, alpha*100), np.percentile(boots_mean, (1-alpha)*100)),
        'wr_ci':   (np.percentile(boots_wr, alpha*100), np.percentile(boots_wr, (1-alpha)*100)),
        'mean': np.mean(boots_mean),
        'std': np.std(boots_mean)
    }

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    log("="*70)
    log(f"STARTING BACKTEST: {SYMBOL}")
    log(f"Timeframe: {TIMEFRAME_MIN}min | Hold: {HOLD_BARS} bars ({TIMEFRAME_MIN*HOLD_BARS}min)")
    log(f"Score Threshold: {SCORE_THRESHOLD} | Min Vol: {MIN_VOL_PCT}% | Slippage: {SLIPPAGE_PCT*100:.1f}%")
    
    # 1. FETCH DATA
    if API_KEY == "YOUR_ALPACA_KEY":
        log("⚠️ Replace API keys at the top of the script", "ERROR")
        sys.exit(1)
        
    log(f"Fetching {DAYS_BACK} days from Alpaca ({FEED} feed)...")
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_BACK)
    
    req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Minute, 
                           start=start, end=end, limit=50000, feed=FEED, adjustment="all")
    bars = client.get_stock_bars(req).df
    
    if bars.empty:
        log("No data returned. Check symbol/API keys.", "ERROR")
        sys.exit(1)
        
    bars = bars.reset_index(level=0, drop=True)
    log(f"✅ Raw minute bars: {len(bars)}")
    
    # 2. RESAMPLE
    freq = f'{TIMEFRAME_MIN}min'
    df = bars.resample(freq).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    log(f"✅ Resampled to {len(df)} {TIMEFRAME_MIN}-min bars | Range: {df.index[0].date()} → {df.index[-1].date()}")
    
    # 3. OSCILLATORS
    log("Calculating oscillators...")
    df = calc_oscillators(df)
    log(f"📊 Oscillator Stats:")
    for col in ['rsi','cci','roc','tsi','stoch']:
        log(f"   {col.upper():5} | Min: {df[col].min():.1f} | Max: {df[col].max():.1f} | Mean: {df[col].mean():.1f}")
        
    # 4. SIGNALS
    log("Generating signals (score engine + regime filters)...")
    df = generate_signals(df)
    total_signals = df['signal'].sum()
    log(f"📈 Total signals: {total_signals} ({total_signals/len(df)*100:.2f}% of bars)")
    
    if total_signals < MIN_TRADES:
        log(f"⚠️ Only {total_signals} signals (min: {MIN_TRADES}). Try lower threshold or longer timeframe.", "WARNING")
        
    # 5. BACKTEST
    log("Running backtest...")
    signals = df[df['signal']].copy()
    ret_col = f'forward_{HOLD_BARS}'
    df[ret_col] = (df['close'].shift(-HOLD_BARS) / df['close'] - 1) * 100
    gross_ret = signals[ret_col].dropna()
    net_ret = gross_ret - (SLIPPAGE_PCT * 100)
    
    if len(net_ret) < MIN_TRADES:
        log(f"⚠️ Insufficient valid returns: {len(net_ret)}. Aborting.", "WARNING")
        sys.exit(1)
        
    win_rate = (net_ret > 0).mean() * 100
    avg_ret = net_ret.mean()
    std_ret = net_ret.std()
    sharpe = avg_ret / std_ret if std_ret > 0 else 0
    t_stat = avg_ret / (std_ret / np.sqrt(len(net_ret))) if std_ret > 0 else 0
    
    # 6. MONTE CARLO
    log(f"Running Monte Carlo ({MC_ITERATIONS} iterations)...")
    mc = monte_carlo(gross_ret)
    mc_mean, mc_std, mc_ci = mc['mean'], mc['std'], mc['mean_ci']
    mc_wr_ci = mc['wr_ci']
    log(f"✅ MC Mean: {mc_mean:.2f}% | 95% CI: [{mc_ci[0]:.2f}%, {mc_ci[1]:.2f}%]")
    log(f"✅ MC WinRate 95% CI: [{mc_wr_ci[0]:.1f}%, {mc_wr_ci[1]:.1f}%]")
    
    # 7. PRINT RESULTS
    print("\n" + "="*70)
    print(f"RESULTS FOR {SYMBOL}")
    print("="*70)
    print(f"""Configuration:
  Timeframe:        {TIMEFRAME_MIN} minutes
  Hold time:        {TIMEFRAME_MIN * HOLD_BARS} minutes ({HOLD_BARS} bars)
  Data period:      {df.index[0].date()} to {df.index[-1].date()}
  Total bars:       {len(df)}

Signal Statistics:
  Total signals:    {total_signals}
  Signal frequency: {total_signals/len(df)*100:.2f}% of bars
  Valid trades:     {len(net_ret)}

Performance (Net of {SLIPPAGE_PCT*100:.1f}% slippage):
  Win Rate:         {win_rate:.1f}%
  Avg Return:       {avg_ret:.2f}%
  Std Deviation:    {std_ret:.2f}%
  Sharpe Ratio:     {sharpe:.2f}
  t-statistic:      {t_stat:.2f}

Return Distribution:
  5th percentile:   {np.percentile(net_ret, 5):.2f}%
  25th percentile:  {np.percentile(net_ret, 25):.2f}%
  75th percentile:  {np.percentile(net_ret, 75):.2f}%
  95th percentile:  {np.percentile(net_ret, 95):.2f}%
  Best trade:       {net_ret.max():.2f}%
  Worst trade:      {net_ret.min():.2f}%

Monte Carlo Validation ({CONFIDENCE_LEVEL}% confidence):
  Avg Return CI:    [{mc_ci[0]:.2f}%, {mc_ci[1]:.2f}%]
  Win Rate CI:      [{mc_wr_ci[0]:.1f}%, {mc_wr_ci[1]:.1f}%]
""")
    
    # 8. INTERPRETATION
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    if t_stat > 2.0:
        print("✅ t-stat > 2.0 - Statistically significant edge")
    else:
        print("❌ t-stat < 2.0 - Not statistically significant (likely noise)")
        
    if win_rate > 60:
        print(f"✅ Win rate {win_rate:.1f}% - Strong edge")
    elif win_rate > 55:
        print(f"🟡 Win rate {win_rate:.1f}% - Modest edge")
    else:
        print(f"❌ Win rate {win_rate:.1f}% - Below random")
        
    if avg_ret > 0.5:
        print(f"✅ Positive expectancy: {avg_ret:.2f}% per trade (survives slippage)")
    elif avg_ret > 0:
        print(f"🟡 Positive but thin: {avg_ret:.2f}% (vulnerable to execution drag)")
    else:
        print(f"❌ Negative expectancy: {avg_ret:.2f}% per trade")
        
    if mc_ci[0] > 0:
        print(f"✅ Monte Carlo confirms positive lower bound")
    else:
        print(f"⚠️ Monte Carlo lower bound dips into negative territory")
        
    # 9. EXPORT
    out_file = f"{SYMBOL}_backtest_{TIMEFRAME_MIN}min_{HOLD_BARS}bars.csv"
    trade_log = pd.DataFrame({
        'timestamp': signals.index[:len(net_ret)],
        'score': signals['score'].values[:len(net_ret)],
        'vol_pct': signals['vol_pct'].values[:len(net_ret)],
        'net_return_pct': net_ret.values,
        'win': (net_ret > 0).astype(int)
    })
    trade_log.to_csv(out_file, index=False)
    log(f"\n📁 Trade log saved to: {out_file}")
    log("="*70)
    log("BACKTEST COMPLETE")

if __name__ == "__main__":
    main()
