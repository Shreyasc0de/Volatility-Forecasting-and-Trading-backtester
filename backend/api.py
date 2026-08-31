"""
FastAPI server for volatility forecasting and backtesting
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from data_pull import fetch_and_process_assets, load_processed_data
from vol_models import fit_all_volatility_models
from regime_models import fit_and_analyze_markov_model
from backtest_engine import run_walk_forward_backtest

app = FastAPI(title="Volatility Forecasting & Trading Backtest API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE = {
    'data': {},
    'vol_models': {},
    'regime_models': {},
    'backtest_results': {}
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
    return value


class DataFetchRequest(BaseModel):
    symbols: List[str] = ["SPY", "AAPL", "BTC-USD"]
    period: str = "5y"
    force_refresh: bool = False


class BacktestRequest(BaseModel):
    symbol: str = "SPY"
    vol_target: float = 0.15
    window_size: int = 252
    step_size: int = 63


@app.get("/health")
async def health():
    return {"status": "OK", "timestamp": datetime.now().isoformat()}


@app.post("/data/fetch")
async def fetch_data(request: DataFetchRequest):
    try:
        data = fetch_and_process_assets(
            symbols=request.symbols,
            period=request.period,
            force_refresh=request.force_refresh
        )
        for symbol, df in data.items():
            CACHE['data'][symbol] = df

        response = {}
        for symbol, df in data.items():
            response[symbol] = {
                'n_obs': len(df),
                'date_range': {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat(),
                },
                'summary': {
                    'mean_return': float(df['logReturn'].mean()),
                    'std_dev': float(df['logReturn'].std()),
                    'skewness': float(df['logReturn'].skew()),
                    'kurtosis': float(df['logReturn'].kurtosis()),
                }
            }

        return JSONResponse({"status": "success", "data": to_jsonable(response), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/{symbol}/series")
async def get_symbol_series(symbol: str):
    try:
        if symbol not in CACHE['data']:
            df = load_processed_data(symbol)
            if df is None:
                raise ValueError(f"Data not found for {symbol}")
            CACHE['data'][symbol] = df
        else:
            df = CACHE['data'][symbol]

        df = df.rename(columns={
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume',
            'logReturn': 'logReturn',
            'rollingVol_20d': 'rollingVol20',
            'rollingVol_60d': 'rollingVol60',
            'ewmaVol': 'ewmaVol',
        })

        records = []
        for row in df.reset_index().itertuples(index=False):
            records.append({
                'date': str(row.Index),
                'timestamp': pd.Timestamp(row.Index).value // 1_000_000,
                'open': float(row.open),
                'high': float(row.high),
                'low': float(row.low),
                'close': float(row.close),
                'volume': int(row.volume),
                'logReturn': float(row.logReturn),
                'rollingVol20': float(row.rollingVol20),
                'rollingVol60': float(row.rollingVol60),
                'ewmaVol': float(row.ewmaVol),
                'intradayRealizedVol': float(row.rollingVol20) if pd.notna(row.rollingVol20) else 0.0,
            })

        return JSONResponse({"status": "success", "data": records, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/volatility/fit")
async def fit_volatility_models(symbol: str = "SPY"):
    try:
        if symbol not in CACHE['data']:
            df = load_processed_data(symbol)
            if df is None:
                raise ValueError(f"Data not found for {symbol}")
            CACHE['data'][symbol] = df
        else:
            df = CACHE['data'][symbol]

        results = fit_all_volatility_models(df['logReturn'], train_ratio=0.8)
        CACHE['vol_models'][symbol] = results

        response = {
            'symbol': symbol,
            'arch_test': {
                'test': results['arch_test']['test'],
                'statistic': float(results['arch_test']['statistic']),
                'p_value': float(results['arch_test']['p_value']),
                'reject_null': bool(results['arch_test']['reject_null']),
                'interpretation': results['arch_test']['interpretation'],
            },
            'models': {},
        }

        for model_name in ['garch', 'egarch', 'gjr']:
            if model_name in results:
                model_result = results[model_name]
                response['models'][model_name] = {
                    'aic': float(model_result['aic']),
                    'bic': float(model_result['bic']),
                    'loglikelihood': float(model_result['loglikelihood']),
                    'converged': bool(model_result['convergence']),
                }

        response['out_of_sample_eval'] = {}
        for eval_name, eval_result in results.get('out_of_sample_evals', {}).items():
            if eval_result:
                response['out_of_sample_eval'][eval_name] = {
                    'rmse': float(eval_result['rmse']),
                    'qlike': float(eval_result['qlike']),
                    'mae': float(eval_result['mae']),
                    'direction_accuracy': float(eval_result['direction_accuracy']),
                }

        return JSONResponse({"status": "success", "data": to_jsonable(response), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regime/fit")
async def fit_regime_model(symbol: str = "SPY"):
    try:
        if symbol not in CACHE['data']:
            df = load_processed_data(symbol)
            if df is None:
                raise ValueError(f"Data not found for {symbol}")
            CACHE['data'][symbol] = df
        else:
            df = CACHE['data'][symbol]

        regime_result = fit_and_analyze_markov_model(df['logReturn'], symbol=symbol)
        CACHE['regime_models'][symbol] = regime_result

        response = {
            'symbol': symbol,
            'persistence': regime_result['persistence'],
            'conditioned_stats': {
                k: {kk: float(vv) if isinstance(vv, (int, np.integer, float, np.floating)) else vv for kk, vv in v.items()}
                for k, v in regime_result['conditioned_stats'].items()}
            ,
            'n_transitions': int(len(regime_result['shifts'])),
            'regimes': list(regime_result['persistence'].keys()),
            'smoothed_probs': regime_result['model_result']['smoothed_probs'].tolist(),
            'filtered_probs': regime_result['model_result']['smoothed_probs'].tolist(),
            'state_probs': regime_result['model_result']['smoothed_probs'].tolist(),
            'transition_matrix': regime_result['model_result']['transition_matrix'].tolist(),
        }

        return JSONResponse({"status": "success", "data": to_jsonable(response), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest/run")
async def run_backtest(request: BacktestRequest):
    try:
        symbol = request.symbol

        if symbol not in CACHE['data']:
            df = load_processed_data(symbol)
            if df is None:
                raise ValueError(f"Data not found for {symbol}")
            CACHE['data'][symbol] = df
        else:
            df = CACHE['data'][symbol]

        vol_forecast = None
        if symbol in CACHE['vol_models'] and 'garch' in CACHE['vol_models'][symbol]:
            vol_forecast = CACHE['vol_models'][symbol]['garch']['conditional_vol']

        regime_probs = None
        if symbol in CACHE['regime_models']:
            regime_probs = CACHE['regime_models'][symbol]['model_result']['smoothed_probs']

        backtest_result = run_walk_forward_backtest(
            df['logReturn'],
            vol_forecast=vol_forecast,
            regime_probs=regime_probs,
            window_size=request.window_size,
            step_size=request.step_size,
            vol_target=request.vol_target
        )

        if not backtest_result:
            raise ValueError("Backtest failed to produce results")

        CACHE['backtest_results'][symbol] = backtest_result

        metrics = backtest_result['metrics']
        response = {
            'symbol': symbol,
            'metrics': {k: float(v) if isinstance(v, (int, np.integer, float, np.floating)) else v for k, v in metrics.items()},
            'n_periods': int(backtest_result['n_periods']),
            'portfolio_values': backtest_result['portfolio_values'].values.tolist(),
            'portfolio_returns': backtest_result['portfolio_returns'].values.tolist(),
            'dates': [d.isoformat() for d in backtest_result['portfolio_returns'].index],
        }

        return JSONResponse({"status": "success", "data": to_jsonable(response), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/{symbol}/summary")
async def get_data_summary(symbol: str):
    try:
        if symbol not in CACHE['data']:
            df = load_processed_data(symbol)
            if df is None:
                raise ValueError(f"Data not found for {symbol}")
            CACHE['data'][symbol] = df
        else:
            df = CACHE['data'][symbol]

        summary = {
            'symbol': symbol,
            'n_obs': len(df),
            'date_range': {
                'start': df.index[0].isoformat(),
                'end': df.index[-1].isoformat(),
            },
            'returns': {
                'mean': float(df['logReturn'].mean()),
                'std': float(df['logReturn'].std()),
                'skewness': float(df['logReturn'].skew()),
                'kurtosis': float(df['logReturn'].kurtosis()),
                'min': float(df['logReturn'].min()),
                'max': float(df['logReturn'].max()),
            },
            'prices': {
                'start': float(df['Close'].iloc[0]),
                'end': float(df['Close'].iloc[-1]),
                'min': float(df['Close'].min()),
                'max': float(df['Close'].max()),
            },
        }

        return JSONResponse({"status": "success", "data": to_jsonable(summary), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/status")
async def get_cache_status():
    return JSONResponse({
        "status": "success",
        "cache": {
            'data_symbols': list(CACHE['data'].keys()),
            'vol_models_symbols': list(CACHE['vol_models'].keys()),
            'regime_models_symbols': list(CACHE['regime_models'].keys()),
            'backtest_results_symbols': list(CACHE['backtest_results'].keys()),
        },
        "timestamp": datetime.now().isoformat(),
    })


@app.post("/cache/clear")
async def clear_cache(symbol: Optional[str] = None):
    if symbol:
        for cache_dict in [CACHE['data'], CACHE['vol_models'], CACHE['regime_models'], CACHE['backtest_results']]:
            cache_dict.pop(symbol, None)
        return JSONResponse({"status": "success", "cleared": symbol})
    CACHE['data'].clear(); CACHE['vol_models'].clear(); CACHE['regime_models'].clear(); CACHE['backtest_results'].clear()
    return JSONResponse({"status": "success", "cleared": "all"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
