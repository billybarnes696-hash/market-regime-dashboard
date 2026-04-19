# liquidity_profiler.py
import streamlit as st
import pandas as pd
import yfinance as yf
from pathlib import Path
import time

st.set_page_config(layout="wide", page_title="Liquidity Profiler - Filter Stocks & ETFs")

st.title("💧 Liquidity Profiler")
st.caption("Upload your stock/ETF list, filter for liquid instruments, and save the clean list for scanning")

# ============================================================================
# LIQUIDITY CONFIGURATION
# ============================================================================

LIQUIDITY_CONFIG = {
    "min_avg_volume": 500000,       # Minimum 500k average daily volume
    "min_price": 5.00,              # Minimum $5 stock price
    "min_market_cap_billions": 1.0, # Minimum $1B market cap
    "max_spread_pct": 0.02,         # Maximum 2% bid-ask spread
    "include_etfs": True,           # Include ETFs (if True, uses ETF volume filter)
    "etf_min_volume": 1000000,      # If including ETFs, require 1M volume
    "etf_min_price": 10.00,         # ETFs minimum price (higher threshold)
}

st.sidebar.header("⚙️ Liquidity Filters")

# Allow user to adjust filters
min_volume = st.sidebar.number_input("Min Avg Volume", value=LIQUIDITY_CONFIG["min_avg_volume"], step=100000)
min_price = st.sidebar.number_input("Min Price ($)", value=LIQUIDITY_CONFIG["min_price"], step=1.0)
min_mcap = st.sidebar.number_input("Min Market Cap ($B)", value=LIQUIDITY_CONFIG["min_market_cap_billions"], step=0.5)
include_etfs = st.sidebar.checkbox("Include ETFs", value=LIQUIDITY_CONFIG["include_etfs"])
etf_min_volume = st.sidebar.number_input("ETF Min Volume", value=LIQUIDITY_CONFIG["etf_min_volume"], step=250000)

# Update config
LIQUIDITY_CONFIG["min_avg_volume"] = int(min_volume)
LIQUIDITY_CONFIG["min_price"] = float(min_price)
LIQUIDITY_CONFIG["min_market_cap_billions"] = float(min_mcap)
LIQUIDITY_CONFIG["include_etfs"] = include_etfs
LIQUIDITY_CONFIG["etf_min_volume"] = int(etf_min_volume)

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_info(symbol: str) -> dict:
    """Fetch stock/ETF info from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get metrics
        avg_volume = info.get('averageVolume', 0)
        current_price = info.get('regularMarketPrice', 0)
        market_cap = info.get('marketCap', 0)
        quote_type = info.get('quoteType', '')
        
        # Calculate spread (if available)
        bid = info.get('bid', 0)
        ask = info.get('ask', 0)
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2)
        else:
            spread_pct = 0.05  # Default high spread if unknown
        
        # Determine instrument type
        is_etf = quote_type == 'ETF'
        
        return {
            'symbol': symbol,
            'price': current_price,
            'avg_volume': avg_volume,
            'market_cap_b': market_cap / 1e9,
            'spread_pct': spread_pct,
            'is_etf': is_etf,
            'quote_type': quote_type,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'price': 0,
            'avg_volume': 0,
            'market_cap_b': 0,
            'spread_pct': 1.0,
            'is_etf': False,
            'quote_type': 'ERROR',
            'sector': 'N/A',
            'industry': 'N/A',
            'error': str(e)[:50],
        }


def check_liquidity(info: dict) -> tuple:
    """Check if symbol passes liquidity filters"""
    
    # Skip if no data
    if info.get('price', 0) == 0:
        return False, "No price data"
    
    # ETF handling
    if info['is_etf']:
        if not LIQUIDITY_CONFIG['include_etfs']:
            return False, "ETF excluded by filter"
        
        # ETF-specific filters
        if info['avg_volume'] < LIQUIDITY_CONFIG['etf_min_volume']:
            return False, f"ETF volume too low ({info['avg_volume']:,} < {LIQUIDITY_CONFIG['etf_min_volume']:,})"
        if info['price'] < LIQUIDITY_CONFIG['etf_min_price']:
            return False, f"ETF price too low (${info['price']:.2f} < ${LIQUIDITY_CONFIG['etf_min_price']})"
    else:
        # Stock filters
        if info['avg_volume'] < LIQUIDITY_CONFIG['min_avg_volume']:
            return False, f"Volume too low ({info['avg_volume']:,} < {LIQUIDITY_CONFIG['min_avg_volume']:,})"
        if info['price'] < LIQUIDITY_CONFIG['min_price']:
            return False, f"Price too low (${info['price']:.2f} < ${LIQUIDITY_CONFIG['min_price']})"
        if info['market_cap_b'] < LIQUIDITY_CONFIG['min_market_cap_billions']:
            return False, f"Market cap too low (${info['market_cap_b']:.1f}B < ${LIQUIDITY_CONFIG['min_market_cap_billions']}B)"
        if info['spread_pct'] > LIQUIDITY_CONFIG['max_spread_pct']:
            return False, f"Spread too wide ({info['spread_pct']:.1%} > {LIQUIDITY_CONFIG['max_spread_pct']:.1%})"
    
    return True, "PASS"


# ============================================================================
# MAIN UI
# ============================================================================

uploaded_file = st.file_uploader("Upload CSV file with 'Symbol' column", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)
    
    if 'Symbol' not in df.columns:
        st.error("CSV must contain a 'Symbol' column")
        st.stop()
    
    symbols = df['Symbol'].tolist()
    st.write(f"Loaded **{len(symbols)}** symbols from file")
    
    # Display preview
    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(20), width='stretch')
    
    # Run liquidity check
    st.subheader("🔍 Running Liquidity Check...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    for i, symbol in enumerate(symbols):
        status_text.text(f"Checking {symbol} ({i+1}/{len(symbols)})")
        info = get_stock_info(symbol)
        passed, reason = check_liquidity(info)
        
        results.append({
            'symbol': symbol,
            'pass': passed,
            'reason': reason if not passed else 'PASS',
            'price': info['price'],
            'avg_volume': info['avg_volume'],
            'market_cap_b': info['market_cap_b'],
            'spread_pct': info['spread_pct'],
            'is_etf': info['is_etf'],
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
        })
        
        progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.1)  # Rate limiting
    
    status_text.text("Done!")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary
    st.subheader("📊 Liquidity Filter Results")
    
    passed_count = results_df['pass'].sum()
    failed_count = len(results_df) - passed_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Symbols", len(results_df))
    col2.metric("✅ Liquid (Pass)", passed_count, delta=f"{passed_count/len(results_df)*100:.0f}%")
    col3.metric("❌ Filtered Out", failed_count, delta=f"-{failed_count/len(results_df)*100:.0f}%")
    
    # Show passed symbols
    passed_symbols = results_df[results_df['pass'] == True]['symbol'].tolist()
    st.success(f"**Liquid Symbols ({len(passed_symbols)}):** {', '.join(passed_symbols[:20])}{'...' if len(passed_symbols) > 20 else ''}")
    
    # Detailed results table
    with st.expander("View detailed results for all symbols"):
        display_df = results_df.copy()
        display_df['avg_volume'] = display_df['avg_volume'].apply(lambda x: f"{x:,.0f}")
        display_df['market_cap_b'] = display_df['market_cap_b'].apply(lambda x: f"${x:.1f}B" if x > 0 else "N/A")
        display_df['spread_pct'] = display_df['spread_pct'].apply(lambda x: f"{x:.2%}")
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        
        st.dataframe(display_df, width='stretch', use_container_width=True)
    
    # Save liquid symbols to file
    if passed_symbols:
        st.subheader("💾 Save Liquid Symbols")
        
        # Create filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"liquid_symbols_{timestamp}.txt"
        
        # Save as text file (one symbol per line)
        file_content = "\n".join(passed_symbols)
        
        st.download_button(
            label="📥 Download Liquid Symbols (TXT)",
            data=file_content,
            file_name=filename,
            mime="text/plain"
        )
        
        # Also show as comma-separated for easy copying
        st.text_area("Copy these symbols for scanner:", ", ".join(passed_symbols), height=100)
        
        # Option to update scanner
        st.info(f"Next: Use these {len(passed_symbols)} liquid symbols in your Diamond Scanner")
        
        # Show filtered out symbols with reasons
        failed_df = results_df[results_df['pass'] == False][['symbol', 'reason', 'price', 'avg_volume']]
        if not failed_df.empty:
            with st.expander(f"❌ Filtered out symbols ({len(failed_df)})"):
                st.dataframe(failed_df, width='stretch', use_container_width=True)

else:
    st.info("👈 Upload a CSV file with a 'Symbol' column to begin")
    
    # Show expected format
    with st.expander("📄 Expected CSV Format"):
        st.code("""
Symbol,Name,Exchange,Sector,Industry,SCTR,Universe,Close,Volume
AAPL,Apple Inc.,NASD,Technology,Computer Hardware,85.2,lrg,175.50,50000000
MSFT,Microsoft Corp.,NASD,Technology,Software,92.1,lrg,420.75,22000000
NVDA,NVIDIA Corp.,NASD,Technology,Semiconductors,98.5,lrg,950.25,45000000
        """)
        st.caption("Only the 'Symbol' column is required. Other columns are optional.")
