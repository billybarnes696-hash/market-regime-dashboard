# liquidity_profiler_batched.py
import streamlit as st
import pandas as pd
import yfinance as yf
import time
from pathlib import Path
from datetime import datetime

st.set_page_config(layout="wide", page_title="Liquidity Profiler - Batched (No Rate Limits)")

st.title("💧 Liquidity Profiler - Batched Version")
st.caption("Processes symbols in small batches to avoid Yahoo Finance rate limits")

# ============================================================================
# LIQUIDITY CONFIGURATION
# ============================================================================

LIQUIDITY_CONFIG = {
    "min_avg_volume": 500000,
    "min_price": 5.00,
    "min_market_cap_billions": 1.0,
    "max_spread_pct": 0.02,
    "include_etfs": True,
    "etf_min_volume": 1000000,
    "etf_min_price": 10.00,
}

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    batch_size = st.slider("Batch size", 25, 200, 100, help="Smaller batches = slower but fewer rate limit errors")
    delay_seconds = st.slider("Delay between batches (seconds)", 1, 10, 3, help="Longer delay = safer but slower")
    resume_from = st.number_input("Resume from row #", 0, 5000, 0, help="Start from a specific row if previous run stopped")
    
    st.header("Liquidity Filters")
    st.caption(f"Min Volume: {LIQUIDITY_CONFIG['min_avg_volume']:,}")
    st.caption(f"Min Price: ${LIQUIDITY_CONFIG['min_price']}")
    st.caption(f"Min Market Cap: ${LIQUIDITY_CONFIG['min_market_cap_billions']}B")
    st.caption(f"Include ETFs: {LIQUIDITY_CONFIG['include_etfs']}")

# ============================================================================
# FUNCTIONS
# ============================================================================

def get_stock_info_batch(symbols: list, batch_num: int, total_batches: int) -> list:
    """Fetch info for a batch of symbols with rate limit handling"""
    results = []
    
    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            avg_volume = info.get('averageVolume', 0)
            current_price = info.get('regularMarketPrice', 0)
            market_cap = info.get('marketCap', 0)
            quote_type = info.get('quoteType', '')
            
            bid = info.get('bid', 0)
            ask = info.get('ask', 0)
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / ((ask + bid) / 2)
            else:
                spread_pct = 0.05
            
            is_etf = quote_type == 'ETF'
            
            results.append({
                'symbol': symbol,
                'price': current_price,
                'avg_volume': avg_volume,
                'market_cap_b': market_cap / 1e9 if market_cap else 0,
                'spread_pct': spread_pct,
                'is_etf': is_etf,
                'quote_type': quote_type,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
            })
            
            # Small delay between symbols within batch
            time.sleep(0.05)
            
        except Exception as e:
            results.append({
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
            })
    
    return results


def check_liquidity(info: dict) -> tuple:
    """Check if symbol passes liquidity filters"""
    
    if info.get('price', 0) == 0:
        return False, "No price data"
    
    if info['is_etf']:
        if not LIQUIDITY_CONFIG['include_etfs']:
            return False, "ETF excluded"
        if info['avg_volume'] < LIQUIDITY_CONFIG['etf_min_volume']:
            return False, f"ETF volume too low ({info['avg_volume']:,})"
        if info['price'] < LIQUIDITY_CONFIG['etf_min_price']:
            return False, f"ETF price too low (${info['price']:.2f})"
    else:
        if info['avg_volume'] < LIQUIDITY_CONFIG['min_avg_volume']:
            return False, f"Volume too low ({info['avg_volume']:,})"
        if info['price'] < LIQUIDITY_CONFIG['min_price']:
            return False, f"Price too low (${info['price']:.2f})"
        if info['market_cap_b'] < LIQUIDITY_CONFIG['min_market_cap_billions']:
            return False, f"Market cap too low (${info['market_cap_b']:.1f}B)"
        if info['spread_pct'] > LIQUIDITY_CONFIG['max_spread_pct']:
            return False, f"Spread too wide ({info['spread_pct']:.1%})"
    
    return True, "PASS"

# ============================================================================
# MAIN UI
# ============================================================================

uploaded_file = st.file_uploader("Upload CSV file with 'Symbol' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'Symbol' not in df.columns:
        st.error("CSV must contain a 'Symbol' column")
        st.stop()
    
    all_symbols = df['Symbol'].tolist()
    total_symbols = len(all_symbols)
    
    st.write(f"Loaded **{total_symbols}** symbols")
    
    # Start from resume point
    start_idx = resume_from
    symbols_to_process = all_symbols[start_idx:]
    
    st.info(f"Starting from row {start_idx} ({len(symbols_to_process)} symbols remaining)")
    
    # Calculate batches
    batches = [symbols_to_process[i:i+batch_size] for i in range(0, len(symbols_to_process), batch_size)]
    total_batches = len(batches)
    
    st.write(f"Will process **{total_batches} batches** of ~{batch_size} symbols each")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    
    # Load existing results if resuming
    if start_idx > 0 and Path("liquidity_results_partial.csv").exists():
        existing = pd.read_csv("liquidity_results_partial.csv")
        all_results = existing.to_dict('records')
        st.info(f"Loaded {len(all_results)} existing results")
    
    for batch_num, batch in enumerate(batches):
        batch_start = start_idx + batch_num * batch_size
        status_text.text(f"Batch {batch_num+1}/{total_batches} - Symbols {batch_start+1}-{batch_start+len(batch)}")
        
        # Process batch
        batch_results = get_stock_info_batch(batch, batch_num+1, total_batches)
        
        # Check liquidity for each
        for result in batch_results:
            passed, reason = check_liquidity(result)
            result['pass'] = passed
            result['reason'] = reason if not passed else 'PASS'
        
        all_results.extend(batch_results)
        
        # Save progress after each batch
        temp_df = pd.DataFrame(all_results)
        temp_df.to_csv("liquidity_results_partial.csv", index=False)
        
        # Update progress
        progress_bar.progress((batch_num + 1) / total_batches)
        
        # Delay between batches to avoid rate limits
        if batch_num < total_batches - 1:
            time.sleep(delay_seconds)
    
    status_text.text("Done!")
    progress_bar.empty()
    
    # Create final results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Display summary
    st.subheader("📊 Liquidity Filter Results")
    
    passed_count = results_df['pass'].sum()
    failed_count = len(results_df) - passed_count
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processed", len(results_df))
    col2.metric("✅ Liquid (Pass)", passed_count, delta=f"{passed_count/len(results_df)*100:.0f}%")
    col3.metric("❌ Filtered Out", failed_count)
    
    # Get liquid symbols
    liquid_symbols = results_df[results_df['pass'] == True]['symbol'].tolist()
    
    st.success(f"**Liquid Symbols ({len(liquid_symbols)}):**")
    
    # Display in columns for easy copying
    cols = st.columns(4)
    for i, sym in enumerate(liquid_symbols):
        cols[i % 4].markdown(f"`{sym}`")
    
    # Save final liquid symbols file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as text file (one per line)
    txt_content = "\n".join(liquid_symbols)
    st.download_button(
        label="📥 Download Liquid Symbols (TXT)",
        data=txt_content,
        file_name=f"liquid_symbols_{timestamp}.txt",
        mime="text/plain"
    )
    
    # Also save as CSV for reference
    liquid_df = results_df[results_df['pass'] == True].copy()
    csv_data = liquid_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Detailed Results (CSV)",
        data=csv_data,
        file_name=f"liquidity_results_{timestamp}.csv",
        mime="text/csv"
    )
    
    # Show filtered out symbols
    with st.expander(f"❌ Filtered out symbols ({len(results_df[results_df['pass'] == False])})"):
        failed_df = results_df[results_df['pass'] == False][['symbol', 'reason', 'price', 'avg_volume']]
        st.dataframe(failed_df, width='stretch', use_container_width=True)
    
    # Clean up partial file
    Path("liquidity_results_partial.csv").unlink(missing_ok=True)
    
else:
    st.info("👈 Upload a CSV file with a 'Symbol' column")
    st.markdown("""
    ### How to use:
    1. Upload your CSV with stock/ETF symbols
    2. Adjust batch size (100 is safe, 50 is safer)
    3. Click 'Run' and wait
    4. Download the liquid symbols list
    
    **The batched approach avoids Yahoo Finance rate limits**
    """)
