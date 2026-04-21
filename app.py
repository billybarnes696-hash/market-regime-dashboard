import pandas as pd
import numpy as np

# ============================================
# SIMULATE REAL 2-HOUR DATA (since you have XLF.csv)
# ============================================

# Load your XLF daily data
df = pd.read_csv("XLF.csv", skiprows=1)
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# Convert daily to simulated 2-hour bars (for testing)
# Each day -> 3.25 bars, so we create ~3 bars per day
def daily_to_2hour(df):
    bars = []
    for i in range(len(df)):
        date = df.index[i]
        o, h, l, c = df.iloc[i][["Open", "High", "Low", "Close"]]
        # Simulate 3 intraday bars per day
        for j in range(3):
            bars.append({
                "Date": date + pd.Timedelta(hours=9.5 + j*2),
                "Open": o * (1 + np.random.randn()*0.002),
                "High": h * (1 + np.random.rand()*0.005),
                "Low": l * (1 - np.random.rand()*0.005),
                "Close": c * (1 + np.random.randn()*0.003),
                "Volume": df.iloc[i]["Volume"] / 3
            })
    return pd.DataFrame(bars).set_index("Date")

two_hour = daily_to_2hour(df.tail(200))  # Last 200 days

# ============================================
# SMOOTHING FUNCTIONS TO TEST
# ============================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# Your current 2-hour smoothing (jaggy)
current_smooth = ema(two_hour["Close"], 8)

# Recommended smoothing (smooth but responsive)
recommended_smooth = ema(two_hour["Close"], 21)

# Over-smoothed (what I wrongly suggested - too slow)
oversmoothed = ema(two_hour["Close"], 160)

# ============================================
# CALCULATE RESPONSIVENESS METRICS
# ============================================

def responsiveness(original, smoothed):
    """How many bars does smoothed lag behind original?"""
    # Find crossovers
    original_dir = original.diff(3) > 0
    smoothed_dir = smoothed.diff(3) > 0
    
    # Average lag in bars
    lag = 0
    count = 0
    for i in range(50, len(original)):
        if original_dir.iloc[i] != original_dir.iloc[i-1]:  # Direction change
            # Find when smoothed changed
            for j in range(i, min(i+20, len(smoothed))):
                if smoothed_dir.iloc[j] != smoothed_dir.iloc[j-1]:
                    lag += (j - i)
                    count += 1
                    break
    return lag / count if count > 0 else 0

# ============================================
# RESULTS
# ============================================

print("=" * 60)
print("2-HOUR SMOOTHING COMPARISON")
print("=" * 60)

print(f"\nCurrent (EMA 8):")
print(f"  - Smoothness: {current_smooth.diff().std():.6f} (lower = smoother)")
print(f"  - Lag: {responsiveness(two_hour['Close'], current_smooth):.1f} bars")

print(f"\nRecommended (EMA 21):")
print(f"  - Smoothness: {recommended_smooth.diff().std():.6f} (lower = smoother)")
print(f"  - Lag: {responsiveness(two_hour['Close'], recommended_smooth):.1f} bars")

print(f"\nOver-smoothed (EMA 160):")
print(f"  - Smoothness: {oversmoothed.diff().std():.6f} (lower = smoother)")
print(f"  - Lag: {responsiveness(two_hour['Close'], oversmoothed):.1f} bars")

print("\n" + "=" * 60)
print("VISUAL COMPARISON (last 50 bars)")
print("=" * 60)

# Print a simple ASCII chart of the last 30 bars
import sys
def ascii_chart(original, smoothed, name, width=50):
    recent_o = original.iloc[-30:].values
    recent_s = smoothed.iloc[-30:].values
    
    # Normalize
    min_val = min(recent_o.min(), recent_s.min())
    max_val = max(recent_o.max(), recent_s.max())
    range_val = max_val - min_val
    
    print(f"\n{name}:")
    print("Original: ", "".join(["█" if x > np.median(recent_o) else "░" for x in recent_o]))
    print("Smoothed: ", "".join(["█" if x > np.median(recent_s) else "░" for x in recent_s]))
    print(f"Std Dev: {original.std():.4f} → {smoothed.std():.4f} (reduction: {(1 - smoothed.std()/original.std())*100:.0f}%)")

ascii_chart(two_hour["Close"], current_smooth, "Current (EMA 8)")
ascii_chart(two_hour["Close"], recommended_smooth, "Recommended (EMA 21)")
ascii_chart(two_hour["Close"], oversmoothed, "Over-smoothed (EMA 160)")
