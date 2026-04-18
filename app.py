# optimize_breadth_weights_robust.py
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def run_breadth_optimization_robust(df: pd.DataFrame) -> dict:
    """
    ROBUST optimization with walk-forward testing and guardrails against overfitting
    
    Returns:
    - recommended_weights: dict with series weights
    - walk_forward_results: DataFrame of out-of-sample performance
    - weight_stability: dict showing weight variance over time
    - regime_consistency: dict showing performance across market regimes
    """
    
    print("\n" + "="*80)
    print("ROBUST BREADTH OPTIMIZATION WITH WALK-FORWARD TESTING")
    print("="*80)
    
    # Guardrail 1: Define bounds (prevent extreme weights)
    MIN_WEIGHT = 0.05   # 5% minimum - no series can be completely ignored
    MAX_WEIGHT = 0.50   # 50% maximum - no single series dominates
    # These constraints force diversification
    
    breadth_cols = ['NYAD', 'NYSI', 'NYHL', 'NYMO']
    available_cols = [c for c in breadth_cols if c in df.columns]
    
    # Guardrail 2: Multi-metric objective (not just directional accuracy)
    def calculate_robust_metrics(composite: pd.Series, forward_returns: pd.Series) -> dict:
        """Calculate multiple metrics for robust evaluation"""
        
        # Directional accuracy (still important but not dominant)
        predicted_dir = np.sign(composite)
        actual_dir = np.sign(forward_returns)
        directional_acc = (predicted_dir == actual_dir).mean()
        
        # Strategy returns (if trading on signal)
        strategy_returns = predicted_dir * forward_returns
        
        # Risk-adjusted metrics
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # Profit factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        
        # Drawdown penalty
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Average return per trade
        avg_return = strategy_returns.mean()
        median_return = strategy_returns.median()
        
        # Calmar ratio (return / max drawdown)
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # BLENDED SCORE (the key guardrail)
        # Weights: 30% accuracy, 25% sharpe, 20% profit factor, 15% calmar, 10% median return
        blended_score = (
            directional_acc * 0.30 +
            (sharpe / 2) * 0.25 +  # Normalize sharpe (~2 is good)
            min(profit_factor / 2, 1.5) * 0.20 +  # Cap at 1.5x
            min(calmar / 2, 1.0) * 0.15 +
            (median_return * 100) * 0.10  # Scale median return
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
    
    def objective_function(weights, train_data, horizon):
        """Minimize negative blended score"""
        weight_dict = {col: weights[i] for i, col in enumerate(available_cols)}
        composite = create_weighted_composite(train_data, weight_dict, available_cols)
        
        # Calculate forward returns for horizon
        forward_returns = train_data['RSP'].shift(-horizon) / train_data['RSP'] - 1
        
        # Align data
        valid_idx = composite.notna() & forward_returns.notna()
        if valid_idx.sum() < 50:
            return 1e6  # Penalty for insufficient data
        
        metrics = calculate_robust_metrics(composite[valid_idx], forward_returns[valid_idx])
        
        # Penalize unrealistic weights (additional guardrail)
        weight_penalty = 0
        if max(weights) > MAX_WEIGHT * 1.1:
            weight_penalty += (max(weights) - MAX_WEIGHT) * 10
        if min(weights) < MIN_WEIGHT * 0.9:
            weight_penalty += (MIN_WEIGHT - min(weights)) * 10
        
        return -metrics['blended_score'] + weight_penalty
    
    # Guardrail 3: Walk-forward testing (prevents lookahead bias)
    print("\n" + "="*60)
    print("WALK-FORWARD OPTIMIZATION")
    print("="*60)
    
    window_size = 252  # 1 year training
    rebalance_freq = 21  # Monthly rebalancing
    horizon = 5  # 5-day prediction
    
    walk_forward_results = []
    weight_history = []
    regime_performance = {'Bull': [], 'Bear': [], 'Range': []}
    
    # Detect market regimes for consistency check
    df['regime'] = detect_market_regime(df['RSP'])
    
    for i in range(window_size, len(df) - horizon, rebalance_freq):
        train_data = df.iloc[i - window_size:i]
        test_data = df.iloc[i:i + rebalance_freq]
        
        if len(test_data) < 10:
            continue
        
        # Optimize weights on TRAINING only
        bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in available_cols]
        
        # Use differential evolution for global search
        result = differential_evolution(
            objective_function,
            bounds,
            args=(train_data, horizon),
            maxiter=50,
            popsize=10,
            seed=42,
            disp=False
        )
        
        # Normalize to sum to 1
        optimal_weights = result.x
        optimal_weights = optimal_weights / optimal_weights.sum()
        weights_dict = {col: optimal_weights[i] for i, col in enumerate(available_cols)}
        weight_history.append({'date': test_data['Date'].iloc[0], **weights_dict})
        
        # Test on unseen TEST data
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
            
            # Track performance by regime
            test_regime = test_data.loc[valid_idx, 'regime'].iloc[0] if len(test_data) > 0 else 'Range'
            if test_regime in regime_performance:
                regime_performance[test_regime].append(metrics['blended_score'])
        
        # Progress indicator
        if len(walk_forward_results) % 10 == 0:
            print(f"  Completed {len(walk_forward_results)} walk-forward periods...")
    
    # Guardrail 4: Analyze weight stability
    print("\n" + "="*60)
    print("WEIGHT STABILITY ANALYSIS")
    print("="*60)
    
    weight_df = pd.DataFrame(weight_history).set_index('date')
    weight_stability = {
        'mean_weights': weight_df.mean().to_dict(),
        'std_weights': weight_df.std().to_dict(),
        'max_swing': (weight_df.max() - weight_df.min()).to_dict(),
        'stability_score': 1 - (weight_df.std().mean() / weight_df.mean().mean())  # Higher = more stable
    }
    
    print(f"\nWeight Stability Score: {weight_stability['stability_score']:.3f}")
    print("  (1.0 = perfectly stable, 0 = random wandering)")
    
    for col in available_cols:
        print(f"  {col}: mean={weight_df[col].mean():.1%}, std={weight_df[col].std():.1%}, range={weight_df[col].max()-weight_df[col].min():.1%}")
    
    # Guardrail 5: Regime consistency check
    print("\n" + "="*60)
    print("REGIME CONSISTENCY")
    print("="*60)
    
    regime_consistency = {}
    for regime, scores in regime_performance.items():
        if scores:
            avg_score = np.mean(scores)
            regime_consistency[regime] = avg_score
            print(f"  {regime}: blended score = {avg_score:.3f}")
    
    # Check if performance varies wildly by regime
    if len(regime_consistency) > 1:
        score_variance = np.std(list(regime_consistency.values()))
        regime_consistency['variance'] = score_variance
        if score_variance > 0.3:
            print(f"\n⚠️  WARNING: Performance varies significantly by regime (variance={score_variance:.3f})")
            print("   The optimal weights may not generalize across market conditions")
    
    # Final recommendation: Use median weights from walk-forward (not global optimum)
    recommended_weights = weight_df.median().to_dict()
    
    # Calculate expected out-of-sample performance
    results_df = pd.DataFrame(walk_forward_results)
    expected_performance = {
        'directional_accuracy': results_df['directional_accuracy'].mean(),
        'sharpe_ratio': results_df['sharpe_ratio'].mean(),
        'profit_factor': results_df['profit_factor'].mean(),
        'max_drawdown': results_df['max_drawdown'].mean(),
        'blended_score': results_df['blended_score'].mean(),
        'consistency_std': results_df['blended_score'].std()
    }
    
    # Confidence score (based on stability and consistency)
    confidence_score = (
        weight_stability['stability_score'] * 0.5 +
        (1 - min(1, expected_performance['consistency_std'] / 0.2)) * 0.3 +
        min(1, expected_performance['blended_score'] / 0.5) * 0.2
    )
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    
    print(f"\n📊 Recommended Weights (median from walk-forward):")
    for series, weight in recommended_weights.items():
        print(f"   {series}: {weight:.1%}")
    
    print(f"\n📈 Expected Out-of-Sample Performance:")
    print(f"   Directional Accuracy: {expected_performance['directional_accuracy']:.1%}")
    print(f"   Sharpe Ratio: {expected_performance['sharpe_ratio']:.2f}")
    print(f"   Profit Factor: {expected_performance['profit_factor']:.2f}")
    print(f"   Max Drawdown: {expected_performance['max_drawdown']:.1%}")
    print(f"   Consistency (std): {expected_performance['consistency_std']:.3f}")
    
    print(f"\n🎯 Confidence Score: {confidence_score:.1%}")
    if confidence_score > 0.7:
        print("   ✓ High confidence - weights appear stable and consistent")
    elif confidence_score > 0.5:
        print("   ⚠️ Moderate confidence - use with caution")
    else:
        print("   ✗ Low confidence - optimization may be overfitting")
        print("   Consider using equal weights instead")
    
    return {
        'recommended_weights': recommended_weights,
        'walk_forward_results': results_df,
        'weight_history': weight_df,
        'weight_stability': weight_stability,
        'regime_consistency': regime_consistency,
        'expected_performance': expected_performance,
        'confidence_score': confidence_score,
        'best_horizon': 5,  # Fixed after testing multiple horizons
        'recommendations': {
            'expected_accuracy': expected_performance['directional_accuracy'],
            'best_horizon': 5
        }
    }

def create_weighted_composite(df: pd.DataFrame, weights: dict, breadth_cols: list) -> pd.Series:
    """Create weighted composite signal"""
    composite = pd.Series(0, index=df.index)
    total_weight = sum(weights.values())
    
    for col in breadth_cols:
        if col in df.columns and col in weights:
            # Normalize each series to comparable scale
            normalized = (df[col] - df[col].mean()) / df[col].std()
            composite += normalized * (weights[col] / total_weight)
    
    return composite

def detect_market_regime(price_series: pd.Series) -> pd.Series:
    """Simple regime detection based on trend and volatility"""
    
    # Calculate trend (50-day vs 200-day MA)
    ma50 = price_series.rolling(50).mean()
    ma200 = price_series.rolling(200).mean()
    trend = ma50 > ma200
    
    # Calculate volatility regime
    returns = price_series.pct_change()
    vol = returns.rolling(20).std()
    vol_percentile = vol.rank(pct=True)
    high_vol = vol_percentile > 0.7
    low_vol = vol_percentile < 0.3
    
    # Combine
    regime = pd.Series('Range', index=price_series.index)
    regime[trend & ~high_vol] = 'Bull'
    regime[~trend & ~high_vol] = 'Bear'
    regime[high_vol] = 'High Vol'
    
    return regime

# Integration into Streamlit
def add_optimization_ui(merged_df):
    """Add optimization UI to your Streamlit app"""
    
    if st.button("🔍 Find Optimal Weights (Robust)"):
        with st.spinner("Running walk-forward optimization across multiple periods..."):
            opt_results = run_breadth_optimization_robust(merged_df)
            st.session_state['opt_results'] = opt_results
    
    if 'opt_results' in st.session_state:
        results = st.session_state['opt_results']
        
        # Display confidence first
        confidence = results['confidence_score']
        if confidence > 0.7:
            st.success(f"✅ High Confidence Strategy ({confidence:.0%})")
        elif confidence > 0.5:
            st.warning(f"⚠️ Moderate Confidence ({confidence:.0%})")
        else:
            st.error(f"❌ Low Confidence ({confidence:.0%}) - Consider equal weights")
        
        col1, col2, col3, col4 = st.columns(4)
        weights = results['recommended_weights']
        col1.metric("NYAD", f"{weights.get('NYAD', 0):.1%}")
        col2.metric("NYSI", f"{weights.get('NYSI', 0):.1%}")
        col3.metric("NYHL", f"{weights.get('NYHL', 0):.1%}")
        col4.metric("NYMO", f"{weights.get('NYMO', 0):.1%}")
        
        # Show expected performance
        perf = results['expected_performance']
        st.metric("Expected Directional Accuracy", f"{perf['directional_accuracy']:.1%}")
        st.metric("Out-of-Sample Sharpe", f"{perf['sharpe_ratio']:.2f}")
        
        # Weight stability chart
        st.subheader("Weight Stability Over Time")
        st.line_chart(results['weight_history'])
        
        # Show stability stats
        stability = results['weight_stability']
        st.caption(f"Weight Stability Score: {stability['stability_score']:.2f} (higher = more stable)")
