# options_liquidity_scanner.py
import streamlit as st
import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Options Liquidity Scanner")

st.title("🎯 Options Liquidity Scanner")
st.caption("Find symbols with liquid options for trading - focuses on options volume, open interest, and bid-ask spreads")

# ============================================================================
# OPTIONS LIQUIDITY CONFIGURATION
# ============================================================================

OPTIONS_CONFIG = {
    # Minimum options volume (sum of puts + calls)
    "min_options_volume": 5000,      # 5,000 contracts/day minimum
    "min_open_interest": 10000,      # 10,000 open interest minimum
    "max_bid_ask_spread": 0.10,      # Max 10% spread (e.g., $0.10 on $1.00 option)
    "min_near_strike_options": 5,    # At least 5 strikes near the money
    
    # Stock liquidity (secondary - affects options liquidity)
    "min_stock_volume": 500000,      # 500k shares/day (liquid underlying)
    "min_stock_price": 5.00,         # $5 minimum stock price
    
    # Expiration preference
    "prefer_weekly": True,           # Prefer weekly options over monthly
    "min_days_to_expiry": 3,         # Minimum 3 days to expiry
    "max_days_to_expiry": 60,        # Maximum 60 days to expiry (near-term)
}

# ============================================================================
# OPTIONS LIQUIDITY CHECK
# ============================================================================

def get_options_liquidity(symbol: str) -> dict:
    """Get comprehensive options liquidity metrics"""
    
    clean_sym = symbol.strip().upper().replace('/', '-')
    
    try:
        ticker = yf.Ticker(clean_sym)
        
        # Get stock info
        info = ticker.info
        stock_price = info.get('regularMarketPrice', 0)
        stock_volume = info.get('averageVolume', 0)
        
        # Get available expirations
        try:
            expirations = list(ticker.options)
        except:
            expirations = []
        
        if not expirations:
            return {
                'symbol': symbol,
                'has_options': False,
                'error': 'No options chain available'
            }
        
        today = datetime.now()
        
        # Analyze each expiration
        expiration_data = []
        total_options_volume = 0
        total_open_interest = 0
        best_expiration = None
        best_liquidity_score = 0
        
        for exp_date in expirations[:5]:  # Check first 5 expirations
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            days_to_expiry = (exp_dt - today).days
            
            # Skip too far out or already expired
            if days_to_expiry < OPTIONS_CONFIG['min_days_to_expiry']:
                continue
            if days_to_expiry > OPTIONS_CONFIG['max_days_to_expiry']:
                continue
            
            try:
                chain = ticker.option_chain(exp_date)
                
                # Analyze calls
                calls = chain.calls
                puts = chain.puts
                
                if calls is None or puts is None or calls.empty or puts.empty:
                    continue
                
                # Calculate metrics for this expiration
                calls_volume = calls['volume'].fillna(0).sum()
                puts_volume = puts['volume'].fillna(0).sum()
                calls_oi = calls['openInterest'].fillna(0).sum()
                puts_oi = puts['openInterest'].fillna(0).sum()
                
                exp_volume = calls_volume + puts_volume
                exp_oi = calls_oi + puts_oi
                
                # Find strikes near the money
                near_strikes = calls[
                    (calls['strike'] >= stock_price * 0.95) & 
                    (calls['strike'] <= stock_price * 1.05)
                ]
                near_strike_count = len(near_strikes)
                
                # Calculate average bid-ask spread for ATM options
                if not near_strikes.empty:
                    avg_spread = ((near_strikes['ask'] - near_strikes['bid']) / 
                                  ((near_strikes['ask'] + near_strikes['bid']) / 2)).mean()
                else:
                    avg_spread = 1.0
                
                # Calculate liquidity score for this expiration
                liquidity_score = 0
                liquidity_score += min(40, exp_volume / 10000)  # Volume up to 40 pts
                liquidity_score += min(30, exp_oi / 20000)      # OI up to 30 pts
                liquidity_score += min(20, near_strike_count)   # Strikes up to 20 pts
                liquidity_score += max(0, 10 - (avg_spread * 100))  # Spread up to 10 pts
                
                expiration_data.append({
                    'expiration': exp_date,
                    'days_to_expiry': days_to_expiry,
                    'total_volume': exp_volume,
                    'total_oi': exp_oi,
                    'near_strike_count': near_strike_count,
                    'avg_spread': avg_spread,
                    'liquidity_score': liquidity_score,
                })
                
                total_options_volume += exp_volume
                total_open_interest += exp_oi
                
                if liquidity_score > best_liquidity_score:
                    best_liquidity_score = liquidity_score
                    best_expiration = exp_date
                    
            except Exception as e:
                continue
        
        if not expiration_data:
            return {
                'symbol': symbol,
                'has_options': True,
                'error': 'No liquid options found'
            }
        
        # Get best expiration details
        best_data = next((e for e in expiration_data if e['expiration'] == best_expiration), expiration_data[0])
        
        # Calculate overall liquidity score
        overall_score = min(100, (
            min(30, total_options_volume / 20000) +
            min(25, total_open_interest / 50000) +
            min(25, best_data['liquidity_score']) +
            min(20, stock_volume / 2000000)
        ))
        
        return {
            'symbol': symbol,
            'has_options': True,
            'stock_price': stock_price,
            'stock_volume': stock_volume,
            'total_options_volume': total_options_volume,
            'total_open_interest': total_open_interest,
            'best_expiration': best_expiration,
            'best_days_to_expiry': best_data['days_to_expiry'],
            'best_volume': best_data['total_volume'],
            'best_oi': best_data['total_oi'],
            'best_near_strikes': best_data['near_strike_count'],
            'best_spread': best_data['avg_spread'],
            'liquidity_score': overall_score,
            'expiration_summary': expiration_data,
            'error': None,
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'has_options': False,
            'error': str(e)[:100],
        }


def is_tradeable_options(metrics: dict) -> tuple:
    """Check if options are tradeable based on liquidity"""
    
    if not metrics.get('has_options', False):
        return False, "No options available"
    
    # Check options volume
    if metrics.get('total_options_volume', 0) < OPTIONS_CONFIG['min_options_volume']:
        return False, f"Options volume too low ({metrics['total_options_volume']:,} < {OPTIONS_CONFIG['min_options_volume']:,})"
    
    # Check open interest
    if metrics.get('total_open_interest', 0) < OPTIONS_CONFIG['min_open_interest']:
        return False, f"Open interest too low ({metrics['total_open_interest']:,} < {OPTIONS_CONFIG['min_open_interest']:,})"
    
    # Check stock volume (liquid underlying)
    if metrics.get('stock_volume', 0) < OPTIONS_CONFIG['min_stock_volume']:
        return False, f"Stock volume too low ({metrics['stock_volume']:,} < {OPTIONS_CONFIG['min_stock_volume']:,})"
    
    # Check stock price
    if metrics.get('stock_price', 0) < OPTIONS_CONFIG['min_stock_price']:
        return False, f"Stock price too low (${metrics['stock_price']:.2f})"
    
    # Check bid-ask spread
    if metrics.get('best_spread', 1.0) > OPTIONS_CONFIG['max_bid_ask_spread']:
        return False, f"Bid-ask spread too wide ({metrics['best_spread']:.1%})"
    
    # Check near-strike options
    if metrics.get('best_near_strikes', 0) < OPTIONS_CONFIG['min_near_strike_options']:
        return False, f"Not enough near-strike options ({metrics['best_near_strikes']} < {OPTIONS_CONFIG['min_near_strike_options']})"
    
    return True, "TRADEABLE"


# ============================================================================
# STREAMLIT UI
# ============================================================================

uploaded_file = st.file_uploader("Upload CSV file with 'Symbol' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'Symbol' not in df.columns:
        st.error("CSV must contain a 'Symbol' column")
        st.stop()
    
    symbols = df['Symbol'].tolist()
    total_symbols = len(symbols)
    
    st.write(f"Loaded **{total_symbols}** symbols")
    
    # Filter options
    with st.expander("⚙️ Options Liquidity Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_opt_vol = st.number_input("Min Options Volume", value=OPTIONS_CONFIG['min_options_volume'], step=1000)
            min_oi = st.number_input("Min Open Interest", value=OPTIONS_CONFIG['min_open_interest'], step=5000)
        with col2:
            max_spread = st.number_input("Max Bid-Ask Spread %", value=OPTIONS_CONFIG['max_bid_ask_spread'] * 100, step=1.0) / 100
            min_strikes = st.number_input("Min Near-Strike Options", value=OPTIONS_CONFIG['min_near_strike_options'], step=1)
        with col3:
            min_stock_vol = st.number_input("Min Stock Volume", value=OPTIONS_CONFIG['min_stock_volume'], step=250000)
            min_stock_price = st.number_input("Min Stock Price", value=OPTIONS_CONFIG['min_stock_price'], step=1.0)
        
        OPTIONS_CONFIG['min_options_volume'] = int(min_opt_vol)
        OPTIONS_CONFIG['min_open_interest'] = int(min_oi)
        OPTIONS_CONFIG['max_bid_ask_spread'] = max_spread
        OPTIONS_CONFIG['min_near_strike_options'] = int(min_strikes)
        OPTIONS_CONFIG['min_stock_volume'] = int(min_stock_vol)
        OPTIONS_CONFIG['min_stock_price'] = float(min_stock_price)
    
    batch_size = st.slider("Batch size", 10, 100, 30, help="Smaller batches = slower but fewer errors")
    
    if st.button("🚀 Scan Options Liquidity", type="primary"):
        
        results = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            progress_bar.progress((i + 1) / total_symbols)
            
            metrics = get_options_liquidity(symbol)
            tradeable, reason = is_tradeable_options(metrics)
            
            results.append({
                'symbol': symbol,
                'tradeable': tradeable,
                'reason': reason if not tradeable else 'TRADEABLE',
                'stock_price': metrics.get('stock_price', 0),
                'stock_volume': metrics.get('stock_volume', 0),
                'options_volume': metrics.get('total_options_volume', 0),
                'open_interest': metrics.get('total_open_interest', 0),
                'best_expiry': metrics.get('best_expiration', 'N/A'),
                'days_to_expiry': metrics.get('best_days_to_expiry', 0),
                'near_strikes': metrics.get('best_near_strikes', 0),
                'bid_ask_spread': metrics.get('best_spread', 0),
                'liquidity_score': metrics.get('liquidity_score', 0),
            })
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        progress_bar.empty()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Filter tradeable options
        tradeable_df = results_df[results_df['tradeable'] == True].copy()
        untradeable_df = results_df[results_df['tradeable'] == False].copy()
        
        # Display summary
        st.subheader("📊 Options Liquidity Scan Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Symbols Scanned", len(results_df))
        col2.metric("✅ Tradeable Options", len(tradeable_df), delta=f"{len(tradeable_df)/len(results_df)*100:.0f}%")
        col3.metric("❌ Not Tradeable", len(untradeable_df))
        
        # Sort by liquidity score
        tradeable_df = tradeable_df.sort_values('liquidity_score', ascending=False)
        
        # Format display
        display_df = tradeable_df.copy()
        display_df['stock_price'] = display_df['stock_price'].apply(lambda x: f"${x:.2f}")
        display_df['stock_volume'] = display_df['stock_volume'].apply(lambda x: f"{x:,.0f}")
        display_df['options_volume'] = display_df['options_volume'].apply(lambda x: f"{x:,.0f}")
        display_df['open_interest'] = display_df['open_interest'].apply(lambda x: f"{x:,.0f}")
        display_df['bid_ask_spread'] = display_df['bid_ask_spread'].apply(lambda x: f"{x:.2%}")
        display_df['liquidity_score'] = display_df['liquidity_score'].apply(lambda x: f"{x:.0f}")
        
        st.subheader(f"🏆 Tradeable Options ({len(tradeable_df)})")
        st.dataframe(display_df, width='stretch', use_container_width=True)
        
        # Download results
        if not tradeable_df.empty:
            st.download_button(
                "📥 Download Tradeable Options List",
                "\n".join(tradeable_df['symbol'].tolist()),
                f"tradeable_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )
        
        # Show untradeable
        with st.expander(f"❌ Not Tradeable ({len(untradeable_df)})"):
            st.dataframe(untradeable_df[['symbol', 'reason', 'options_volume', 'open_interest']].head(30), width='stretch')

else:
    st.info("👈 Upload a CSV file with a 'Symbol' column")
    
    st.markdown("""
    ### 🎯 Options Liquidity Metrics
    
    This scanner focuses on what matters for options trading:
    
    | Metric | Why It Matters |
    |--------|----------------|
    | **Options Volume** | Can you get in and out? |
    | **Open Interest** | Is there liquidity at your strike? |
    | **Bid-Ask Spread** | What's your actual trading cost? |
    | **Near-Strike Options** | Can you trade ATM/OTM? |
    | **Stock Volume** | Liquid underlying = better options |
    | **Days to Expiry** | Time decay considerations |
    
    **Good options liquidity means:**
    - Tight bid-ask spreads (< 10%)
    - High volume (> 5,000 contracts/day)
    - High open interest (> 10,000)
    - Multiple strikes near the money
    
    **Examples of highly liquid options:**
    - SPY, QQQ, IWM (index ETFs)
    - AAPL, MSFT, NVDA (mega-cap tech)
    - TSLA, AMD, META (high volatility names)
    """)
