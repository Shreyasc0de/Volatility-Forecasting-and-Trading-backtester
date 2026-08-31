"""
Volatility Modeling Module: GARCH, EGARCH, GJR-GARCH using arch library
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


def test_arch_effects(returns: pd.Series, lags: int = 4) -> dict:
    """
    Perform ARCH-LM test for heteroskedasticity.
    
    Args:
        returns: Log returns series
        lags: Number of lags for test
    
    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    # Square returns to test for autocorrelation (ARCH effect)
    squared_returns = returns ** 2
    
    # Ljung-Box test on squared returns
    lb_test = acorr_ljungbox(squared_returns, lags=lags, return_df=True)
    lb_stat = lb_test['lb_stat'].iloc[-1]
    lb_pval = lb_test['lb_pvalue'].iloc[-1]
    
    return {
        'test': 'ARCH-LM (Ljung-Box)',
        'statistic': lb_stat,
        'p_value': lb_pval,
        'reject_null': lb_pval < 0.05,  # Null: no ARCH effects
        'interpretation': 'ARCH effects present' if lb_pval < 0.05 else 'No significant ARCH effects'
    }


def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    Fit GARCH(p, q) model using MLE.
    
    Args:
        returns: Log returns series (in decimal form, e.g., 0.01 for 1%)
        p, q: GARCH order
    
    Returns:
        Dictionary with model results and diagnostics
    """
    # Scale returns to percentage for numerical stability
    returns_pct = returns * 100
    
    model = arch_model(returns_pct, vol='Garch', p=p, q=q)
    
    try:
        result = model.fit(disp='off', show_warning=False)
        
        return {
            'model_name': f'GARCH({p},{q})',
            'result': result,
            'aic': result.aic,
            'bic': result.bic,
            'loglikelihood': result.loglikelihood,
            'params': result.params.to_dict(),
            'conditional_vol': result.conditional_volatility / 100,  # Convert back to decimal
            'standardized_resid': result.std_resid,
            'convergence': result.convergence_flag == 0
        }
    except Exception as e:
        print(f"✗ GARCH fit failed: {e}")
        return None


def fit_egarch_model(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    Fit EGARCH(p, q) model (captures leverage effect / asymmetric vol).
    
    Args:
        returns: Log returns series
        p, q: EGARCH order
    
    Returns:
        Dictionary with model results
    """
    returns_pct = returns * 100
    
    model = arch_model(returns_pct, vol='EGarch', p=p, q=q)
    
    try:
        result = model.fit(disp='off', show_warning=False)
        
        return {
            'model_name': f'EGARCH({p},{q})',
            'result': result,
            'aic': result.aic,
            'bic': result.bic,
            'loglikelihood': result.loglikelihood,
            'params': result.params.to_dict(),
            'conditional_vol': result.conditional_volatility / 100,
            'standardized_resid': result.std_resid,
            'convergence': result.convergence_flag == 0
        }
    except Exception as e:
        print(f"✗ EGARCH fit failed: {e}")
        return None


def fit_gjr_model(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    Fit GJR-GARCH (Glosten-Jagannathan-Runkle) model.
    Captures leverage effect through asymmetric response to negative shocks.
    
    Args:
        returns: Log returns series
        p, q: Model order
    
    Returns:
        Dictionary with model results
    """
    returns_pct = returns * 100
    
    model = arch_model(returns_pct, vol='Garch', p=p, o=1, q=q)  # o=1 for asymmetric term
    
    try:
        result = model.fit(disp='off', show_warning=False)
        
        return {
            'model_name': f'GJR-GARCH({p},1,{q})',
            'result': result,
            'aic': result.aic,
            'bic': result.bic,
            'loglikelihood': result.loglikelihood,
            'params': result.params.to_dict(),
            'conditional_vol': result.conditional_volatility / 100,
            'standardized_resid': result.std_resid,
            'convergence': result.convergence_flag == 0
        }
    except Exception as e:
        print(f"✗ GJR-GARCH fit failed: {e}")
        return None


def compute_baseline_forecasts(returns: pd.Series) -> dict:
    """
    Compute baseline volatility forecasts:
    - Naive (yesterday's realized vol)
    - Historical average
    - EWMA
    
    Args:
        returns: Log returns series
    
    Returns:
        Dictionary with baseline forecast series
    """
    squared_returns = returns ** 2
    
    # Naive: yesterday's realized vol
    naive_vol = np.sqrt(squared_returns).shift(1)
    
    # Historical average (20-day)
    hist_vol = returns.rolling(window=20).std() * np.sqrt(252)
    
    # EWMA
    ewma_var = pd.Series(index=returns.index, dtype=float)
    ewma_var.iloc[0] = squared_returns.iloc[0]
    
    lambda_param = 0.94
    for i in range(1, len(returns)):
        ewma_var.iloc[i] = lambda_param * ewma_var.iloc[i-1] + (1 - lambda_param) * squared_returns.iloc[i]
    
    ewma_vol = np.sqrt(ewma_var)
    
    return {
        'naive': naive_vol,
        'historical_20d': hist_vol,
        'ewma': ewma_vol
    }


def evaluate_vol_forecast(realized_vol: pd.Series, forecast_vol: pd.Series) -> dict:
    """
    Evaluate volatility forecast accuracy using RMSE and QLIKE loss.
    
    Args:
        realized_vol: Realized/actual volatility
        forecast_vol: Forecasted volatility
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Align series, drop NaNs
    aligned = pd.DataFrame({
        'realized': realized_vol,
        'forecast': forecast_vol
    }).dropna()
    
    if len(aligned) == 0:
        return None
    
    realized = aligned['realized'].values
    forecast = aligned['forecast'].values
    
    # RMSE
    rmse = np.sqrt(np.mean((realized - forecast) ** 2))
    
    # QLIKE: mean of log(forecast) + realized^2 / forecast^2
    qlike = np.mean(np.log(forecast) + (realized ** 2) / (forecast ** 2))
    
    # Mean absolute error
    mae = np.mean(np.abs(realized - forecast))
    
    # Direction accuracy (did forecast correctly predict vol increase/decrease)
    realized_direction = np.sign(np.diff(realized))
    forecast_direction = np.sign(np.diff(forecast))
    direction_accuracy = np.mean(realized_direction == forecast_direction)
    
    return {
        'rmse': rmse,
        'qlike': qlike,
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'n_obs': len(aligned)
    }


def fit_all_volatility_models(returns: pd.Series, train_ratio: float = 0.8) -> dict:
    """
    Fit all volatility models and compare on out-of-sample data.
    
    Args:
        returns: Log returns series
        train_ratio: Fraction of data for training
    
    Returns:
        Dictionary with all model results and comparison metrics
    """
    print(f"Fitting volatility models ({len(returns)} observations)...")
    
    n_train = int(len(returns) * train_ratio)
    train_returns = returns.iloc[:n_train]
    test_returns = returns.iloc[n_train:]
    
    print(f"  Train set: {len(train_returns)} obs, Test set: {len(test_returns)} obs")
    
    results = {}
    
    # Test ARCH effects
    arch_test = test_arch_effects(train_returns)
    results['arch_test'] = arch_test
    print(f"  ARCH-LM test: {arch_test['interpretation']} (p={arch_test['p_value']:.4f})")
    
    # Baseline forecasts
    baseline = compute_baseline_forecasts(train_returns)
    results['baseline'] = baseline
    
    # GARCH(1,1)
    print(f"  Fitting GARCH(1,1)...")
    garch_result = fit_garch_model(train_returns)
    if garch_result:
        results['garch'] = garch_result
        print(f"    AIC: {garch_result['aic']:.2f}, Converged: {garch_result['convergence']}")
    
    # EGARCH(1,1)
    print(f"  Fitting EGARCH(1,1)...")
    egarch_result = fit_egarch_model(train_returns)
    if egarch_result:
        results['egarch'] = egarch_result
        print(f"    AIC: {egarch_result['aic']:.2f}, Converged: {egarch_result['convergence']}")
    
    # GJR-GARCH(1,1,1)
    print(f"  Fitting GJR-GARCH(1,1,1)...")
    gjr_result = fit_gjr_model(train_returns)
    if gjr_result:
        results['gjr'] = gjr_result
        print(f"    AIC: {gjr_result['aic']:.2f}, Converged: {gjr_result['convergence']}")
    
    # Out-of-sample evaluation
    # Realized volatility on test set (rolling 20-day)
    test_realized_vol = test_returns.rolling(window=20).std() * np.sqrt(252)
    
    results['out_of_sample_evals'] = {}
    
    # Evaluate baselines
    for baseline_name, baseline_series in baseline.items():
        test_baseline = baseline_series.iloc[n_train:]
        eval_result = evaluate_vol_forecast(test_realized_vol, test_baseline)
        if eval_result:
            results['out_of_sample_evals'][f'baseline_{baseline_name}'] = eval_result
            print(f"  Baseline ({baseline_name}) RMSE: {eval_result['rmse']:.6f}")
    
    # Evaluate GARCH models
    if 'garch' in results:
        eval_result = evaluate_vol_forecast(test_realized_vol, results['garch']['conditional_vol'].iloc[n_train:])
        if eval_result:
            results['out_of_sample_evals']['garch'] = eval_result
            print(f"  GARCH RMSE: {eval_result['rmse']:.6f}")
    
    if 'egarch' in results:
        eval_result = evaluate_vol_forecast(test_realized_vol, results['egarch']['conditional_vol'].iloc[n_train:])
        if eval_result:
            results['out_of_sample_evals']['egarch'] = eval_result
            print(f"  EGARCH RMSE: {eval_result['rmse']:.6f}")
    
    if 'gjr' in results:
        eval_result = evaluate_vol_forecast(test_realized_vol, results['gjr']['conditional_vol'].iloc[n_train:])
        if eval_result:
            results['out_of_sample_evals']['gjr'] = eval_result
            print(f"  GJR-GARCH RMSE: {eval_result['rmse']:.6f}")
    
    print(f"✓ Volatility modeling complete!")
    
    return results


if __name__ == "__main__":
    from data_pull import fetch_and_process_assets
    
    # Example usage
    data = fetch_and_process_assets(['SPY'])
    if 'SPY' in data:
        results = fit_all_volatility_models(data['SPY']['logReturn'])
