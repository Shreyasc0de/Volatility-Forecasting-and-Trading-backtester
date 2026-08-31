# Backend Python Analysis Guide

## Quick Start

### 1. Fetch Data
```python
from data_pull import fetch_and_process_assets

data = fetch_and_process_assets(['SPY', 'AAPL', 'BTC-USD'])
spy_df = data['SPY']  # DataFrame with OHLCV + returns + volatilities
```

### 2. Fit Volatility Models
```python
from vol_models import fit_all_volatility_models

results = fit_all_volatility_models(spy_df['logReturn'], train_ratio=0.8)
# results contains: GARCH, EGARCH, GJR-GARCH + baseline forecasts + OOS evaluation
```

### 3. Markov Regime-Switching
```python
from regime_models import fit_and_analyze_markov_model

regime_analysis = fit_and_analyze_markov_model(spy_df['logReturn'], symbol='SPY')
# regime_analysis contains: smoothed probs, persistence, stats by regime
```

### 4. Walk-Forward Backtest
```python
from backtest_engine import run_walk_forward_backtest

backtest = run_walk_forward_backtest(
    returns=spy_df['logReturn'],
    vol_forecast=results['garch']['conditional_vol'],
    regime_probs=regime_analysis['model_result']['smoothed_probs'],
    vol_target=0.15
)
# backtest metrics: Sharpe, max drawdown, Calmar ratio, etc.
```

## Module Details

### data_pull.py
**Functions:**
- `pull_asset_data(symbol, period='5y', interval='1d')` → yfinance DataFrame
- `compute_log_returns(prices)` → log returns series
- `compute_rolling_volatility(returns, window=20)` → rolling std (annualized)
- `compute_ewma_volatility(returns, lambda_param=0.94)` → RiskMetrics vol
- `preprocess_asset_data(symbol, raw_df)` → cleaned + features
- `save_raw_data(symbol, df)` → timestamp-based save
- `fetch_and_process_assets(symbols=None)` → main entry point

### vol_models.py
**Functions:**
- `test_arch_effects(returns, lags=4)` → ARCH-LM test results
- `fit_garch_model(returns, p=1, q=1)` → GARCH(p,q) fit
- `fit_egarch_model(returns, p=1, q=1)` → EGARCH(p,q) fit (asymmetric)
- `fit_gjr_model(returns, p=1, q=1)` → GJR-GARCH fit (leverage effect)
- `compute_baseline_forecasts(returns)` → naive, hist avg, EWMA
- `evaluate_vol_forecast(realized, forecast)` → RMSE, QLIKE, MAE
- `fit_all_volatility_models(returns, train_ratio=0.8)` → complete pipeline

### regime_models.py
**Functions:**
- `fit_markov_switching_model(data, k_regimes=2, order=0)` → 2-state MS
- `analyze_regime_persistence(smoothed_probs)` → duration, transitions
- `detect_regime_shifts(smoothed_probs)` → crossover indices
- `regime_conditioned_statistics(data, smoothed_probs)` → stats by regime
- `fit_and_analyze_markov_model(returns, symbol)` → complete analysis

### backtest_engine.py
**Classes:**
- `BacktestEngine` - orchestrates backtesting
  - `inverse_vol_position_sizing(forecast_vol, vol_target=0.15)`
  - `regime_based_exposure(regime_probs, threshold=0.5)`
  - `run_backtest(...)`

**Functions:**
- `compute_performance_metrics(returns, benchmark=None)` → Sharpe, Calmar, etc.
- `run_walk_forward_backtest(returns, vol_forecast, regime_probs, window_size=252, step_size=63)`
- `compare_strategies(results_dict)` → comparison table

## Output Files

Processed data saved to `data/processed/{SYMBOL}_processed.csv`:
- Index: Date
- Columns: Open, High, Low, Close, Volume, logReturn, pctReturn, rollingVol_20d, rollingVol_60d, ewmaVol

Raw data timestamped in `data/raw/{SYMBOL}_{YYYYMMDD_HHMMSS}.csv`

## Jupyter Workflow (Optional)

For exploratory analysis:
```python
import pandas as pd
import matplotlib.pyplot as plt

from data_pull import fetch_and_process_assets
from vol_models import fit_all_volatility_models
from regime_models import fit_and_analyze_markov_model

# Fetch
data = fetch_and_process_assets(['SPY'])
spy = data['SPY']

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
ax1.plot(spy.index, spy['Close'], label='SPY Close')
ax1.set_ylabel('Price')
ax1.legend()
ax2.plot(spy.index, spy['ewmaVol'], label='EWMA Vol')
ax2.set_ylabel('Volatility')
ax2.legend()
plt.tight_layout()
plt.show()

# Model
vol_results = fit_all_volatility_models(spy['logReturn'])
regime_results = fit_and_analyze_markov_model(spy['logReturn'])
```

## Troubleshooting

**ModuleNotFoundError:** Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

**yfinance connection error:** Check internet connection, yfinance server status

**GARCH convergence issues:** Data may have extreme values; try log-differencing or scaling

**Regime model fitting slow:** Reduce training set size or increase tolerance

## Performance Notes

- Data fetch (5 years): ~5-10 seconds per symbol
- GARCH fitting (1000 obs): ~2-5 seconds
- Markov model fitting: ~10-20 seconds
- Walk-forward backtest: ~30-60 seconds depending on window size
- API server startup: ~3-5 seconds (compiling statsmodels/arch models)

## Key References

- Engle (1982): ARCH models
- Bollerslev (1986): GARCH models
- Nelson (1991): EGARCH (exponential GARCH)
- Glosten, Jagannathan, Runkle (1993): GJR-GARCH
- Hamilton (1989): Regime-switching models
