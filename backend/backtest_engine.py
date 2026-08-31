"""
Backtest Integration Module: Walk-forward validation with vol overlay strategies
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class BacktestEngine:
    """
    Walk-forward backtesting engine with position sizing and risk controls.
    """
    
    def __init__(self, returns: pd.Series, initial_capital: float = 100000.0):
        """
        Initialize backtest engine.
        
        Args:
            returns: Log returns series
            initial_capital: Starting portfolio value
        """
        self.returns = returns
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.dates = returns.index
        self.portfolio_values = [initial_capital]
        self.trades = []
        
    def inverse_vol_position_sizing(self, forecast_vol: pd.Series, vol_target: float = 0.15) -> pd.Series:
        """
        Compute position sizes inversely proportional to forecasted volatility.
        
        Args:
            forecast_vol: Forecasted volatility series
            vol_target: Target portfolio volatility
        
        Returns:
            Position sizing weights (0 to 1 for long, -1 to 0 for short)
        """
        # Inverse relationship: lower vol -> higher exposure
        inv_vol = 1.0 / (forecast_vol + 1e-6)
        
        # Normalize to sum to 1
        normalized = inv_vol / inv_vol.mean()
        
        # Scale to target volatility
        position_size = vol_target / (forecast_vol + 1e-6)
        
        # Cap at 1.0 (full allocation)
        position_size = np.minimum(position_size, 1.0)
        position_size = np.maximum(position_size, 0.0)
        
        return pd.Series(position_size, index=forecast_vol.index)
    
    def regime_based_exposure(self, regime_probs: np.ndarray, regime_threshold: float = 0.5) -> pd.Series:
        """
        Adjust exposure based on regime probabilities.
        
        Args:
            regime_probs: Smoothed regime probabilities (T x 2)
            regime_threshold: Probability threshold for "Turbulent" regime
        
        Returns:
            Exposure adjustment factor (0 to 1)
        """
        # Assume regime 1 is "turbulent"
        turbulent_prob = regime_probs[:, 1]
        
        # Linear adjustment: full exposure in calm, reduce in turbulent
        exposure = 1.0 - turbulent_prob
        
        return pd.Series(exposure, index=range(len(exposure)))
    
    def run_backtest(
        self,
        base_returns: pd.Series,
        vol_forecast: pd.Series = None,
        regime_probs: np.ndarray = None,
        use_vol_sizing: bool = True,
        use_regime_control: bool = True,
        vol_target: float = 0.15,
        regime_threshold: float = 0.5
    ) -> Dict:
        """
        Run backtest with optional vol overlay strategies.
        
        Args:
            base_returns: Base strategy returns (e.g., buy-and-hold)
            vol_forecast: Forecasted volatility for position sizing
            regime_probs: Regime probabilities for exposure control
            use_vol_sizing: Apply inverse-vol position sizing
            use_regime_control: Apply regime-based exposure control
            vol_target: Target portfolio volatility
            regime_threshold: Threshold for regime switch
        
        Returns:
            Dictionary with backtest results
        """
        n = len(base_returns)
        portfolio_returns = pd.Series(index=base_returns.index, dtype=float)
        positions = pd.Series(index=base_returns.index, dtype=float, data=1.0)  # Start fully invested
        
        # Apply vol position sizing
        if use_vol_sizing and vol_forecast is not None:
            vol_sizes = self.inverse_vol_position_sizing(vol_forecast, vol_target=vol_target)
            positions = positions * vol_sizes
        
        # Apply regime-based exposure control
        if use_regime_control and regime_probs is not None:
            regime_exposure = self.regime_based_exposure(regime_probs, regime_threshold)
            positions = positions * regime_exposure.values
        
        # Compute portfolio returns
        portfolio_returns = base_returns * positions
        
        # Compute cumulative portfolio value
        cum_returns = (1 + portfolio_returns).cumprod()
        portfolio_values = self.initial_capital * cum_returns
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'positions': positions,
            'dates': base_returns.index
        }


def compute_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
    """
    Compute comprehensive performance metrics.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns (optional)
    
    Returns:
        Dictionary with performance metrics
    """
    # Annualization factor
    annual_factor = 252
    
    # Basic statistics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns.mean()) ** annual_factor - 1
    annual_vol = returns.std() * np.sqrt(annual_factor)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino ratio (only negative returns)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(annual_factor)
    sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Metrics
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'n_positive_days': (returns > 0).sum(),
        'n_negative_days': (returns < 0).sum(),
        'mean_return': returns.mean(),
        'std_return': returns.std()
    }
    
    # Benchmark comparison
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns[:len(returns)]
        excess_returns = returns - benchmark_returns
        
        metrics['excess_return'] = excess_returns.sum()
        metrics['tracking_error'] = excess_returns.std() * np.sqrt(annual_factor)
        metrics['information_ratio'] = (excess_returns.mean() * annual_factor) / metrics['tracking_error'] \
            if metrics['tracking_error'] > 0 else 0
    
    return metrics


def run_walk_forward_backtest(
    returns: pd.Series,
    vol_forecast: pd.Series = None,
    regime_probs: np.ndarray = None,
    window_size: int = 252,
    step_size: int = 63,  # Quarterly rebalance
    vol_target: float = 0.15
) -> Dict:
    """
    Run walk-forward backtest with rolling model refit.
    
    Args:
        returns: Daily returns
        vol_forecast: Volatility forecasts
        regime_probs: Regime probabilities
        window_size: Training window size (days)
        step_size: Step size for rolling window (days)
        vol_target: Target portfolio volatility
    
    Returns:
        Dictionary with backtest results and metrics
    """
    n = len(returns)
    portfolio_values = []
    positions_series = []
    
    engine = BacktestEngine(returns)
    
    # Walk forward
    for start_idx in range(0, n - window_size, step_size):
        end_idx = start_idx + window_size
        
        if end_idx + step_size > n:
            end_idx = n
        
        # Training period returns
        train_returns = returns.iloc[start_idx:end_idx]
        
        # Test period
        test_start = end_idx
        test_end = min(end_idx + step_size, n)
        
        if test_start >= n:
            break
        
        test_returns = returns.iloc[test_start:test_end]
        
        # Get vol forecasts and regime probs for test period
        test_vol = None
        test_regime = None
        
        if vol_forecast is not None:
            test_vol = vol_forecast.iloc[test_start:test_end]
        
        if regime_probs is not None:
            test_regime = regime_probs[test_start:test_end]
        
        # Run backtest on test period
        result = engine.run_backtest(
            test_returns,
            vol_forecast=test_vol,
            regime_probs=test_regime,
            use_vol_sizing=True,
            use_regime_control=True,
            vol_target=vol_target
        )
        
        portfolio_values.extend(result['portfolio_values'].values)
        positions_series.extend(result['positions'].values)
    
    # Compute final metrics
    if len(portfolio_values) > 0:
        portfolio_values = pd.Series(portfolio_values, index=returns.iloc[-len(portfolio_values):].index)
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        metrics = compute_performance_metrics(portfolio_returns, benchmark_returns=returns.iloc[-len(portfolio_returns):])
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'positions': positions_series,
            'metrics': metrics,
            'n_periods': len(portfolio_returns)
        }
    
    return None


def compare_strategies(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple strategy backtests.
    
    Args:
        results_dict: Dictionary {strategy_name: backtest_result}
    
    Returns:
        DataFrame with comparison metrics
    """
    comparison = {}
    
    for strategy_name, result in results_dict.items():
        if 'metrics' in result:
            comparison[strategy_name] = result['metrics']
    
    comparison_df = pd.DataFrame(comparison).T
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    from data_pull import fetch_and_process_assets
    from vol_models import fit_all_volatility_models
    from regime_models import fit_and_analyze_markov_model
    
    # Fetch data
    data = fetch_and_process_assets(['SPY'])
    if 'SPY' in data:
        spy_data = data['SPY']
        returns = spy_data['logReturn']
        
        # Fit models
        vol_results = fit_all_volatility_models(returns)
        regime_results = fit_and_analyze_markov_model(returns, symbol='SPY')
        
        # Get forecasts
        garch_vol = vol_results['garch']['conditional_vol'] if 'garch' in vol_results else returns.rolling(20).std()
        regime_probs = regime_results['model_result']['smoothed_probs']
        
        # Run backtest
        backtest_result = run_walk_forward_backtest(
            returns,
            vol_forecast=garch_vol,
            regime_probs=regime_probs,
            vol_target=0.15
        )
        
        print("\nBacktest Metrics:")
        for metric, value in backtest_result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
