"""
Data Acquisition Module: Pull OHLCV data from yfinance and preprocess
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path


def ensure_data_dirs():
    """Create data directories if they don't exist"""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def pull_asset_data(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Pull OHLCV data from yfinance for a given symbol.
    
    Args:
        symbol: Asset ticker (e.g., 'SPY', 'AAPL', 'BTC-USD')
        period: Data period (default '5y')
        interval: Frequency ('1d' for daily, '1h' for hourly, etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Pulling {symbol} data for {period}...")
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        print(f"✓ Downloaded {len(data)} rows for {symbol}")
        return data
    except Exception as e:
        print(f"✗ Error downloading {symbol}: {e}")
        return pd.DataFrame()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from price series"""
    return np.log(prices / prices.shift(1))


def compute_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling historical volatility (annualized)"""
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_vol


def compute_ewma_volatility(returns: pd.Series, lambda_param: float = 0.94) -> pd.Series:
    """
    Compute EWMA volatility (RiskMetrics style).
    
    Args:
        returns: Log returns series
        lambda_param: Decay parameter (default 0.94 for daily data)
    
    Returns:
        EWMA volatility series (annualized)
    """
    variance = returns.var()
    ewma_var = pd.Series(index=returns.index, dtype=float)
    ewma_var.iloc[0] = variance
    
    for i in range(1, len(returns)):
        ewma_var.iloc[i] = lambda_param * ewma_var.iloc[i-1] + (1 - lambda_param) * returns.iloc[i]**2
    
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
    return ewma_vol


def preprocess_asset_data(symbol: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw OHLCV data: compute returns, volatilities, and clean NaNs.
    
    Args:
        symbol: Asset ticker
        raw_df: Raw OHLCV DataFrame from yfinance
    
    Returns:
        Processed DataFrame with returns and volatilities
    """
    df = raw_df.copy()
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Normalize column names (lowercase to title case)
    df.columns = df.columns.str.title()
    
    df = df.dropna(subset=['Close'])
    
    # Compute log returns
    df['logReturn'] = compute_log_returns(df['Close'])
    df['pctReturn'] = df['Close'].pct_change()
    
    # Historical volatilities
    df['rollingVol_20d'] = compute_rolling_volatility(df['logReturn'], window=20)
    df['rollingVol_60d'] = compute_rolling_volatility(df['logReturn'], window=60)
    
    # EWMA volatility
    df['ewmaVol'] = compute_ewma_volatility(df['logReturn'], lambda_param=0.94)
    
    # Drop first row (NaN from log return calculation)
    df = df.dropna()
    
    return df


def save_raw_data(symbol: str, df: pd.DataFrame):
    """Save raw OHLCV data with timestamp in filename"""
    ensure_data_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/{symbol}_{timestamp}.csv"
    df.to_csv(filename)
    print(f"✓ Saved raw data to {filename}")
    return filename


def save_processed_data(symbol: str, df: pd.DataFrame):
    """Save processed data"""
    ensure_data_dirs()
    filename = f"data/processed/{symbol}_processed.csv"
    df.to_csv(filename, index=True)
    print(f"✓ Saved processed data to {filename}")
    return filename


def load_processed_data(symbol: str) -> pd.DataFrame:
    """Load processed data for a symbol"""
    filename = f"data/processed/{symbol}_processed.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    return None


def fetch_and_process_assets(symbols: list = None, period: str = "5y", force_refresh: bool = False) -> dict:
    """
    Fetch and process multiple assets, returning processed DataFrames.
    
    Args:
        symbols: List of asset tickers (default: ['SPY', 'AAPL', 'BTC-USD'])
        period: Data period
        force_refresh: Re-fetch data even if processed files exist
    
    Returns:
        Dictionary {symbol: processed_df}
    """
    if symbols is None:
        symbols = ['SPY', 'AAPL', 'BTC-USD']
    
    processed_data = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")
        
        # Try loading existing processed data first
        if not force_refresh:
            existing = load_processed_data(symbol)
            if existing is not None and len(existing) > 0:
                print(f"✓ Using existing processed data for {symbol}")
                processed_data[symbol] = existing
                continue
        
        # Pull raw data
        raw_df = pull_asset_data(symbol, period=period)
        if raw_df.empty:
            print(f"✗ Failed to fetch {symbol}, skipping")
            continue
        
        # Save raw data (with timestamp)
        save_raw_data(symbol, raw_df)
        
        # Preprocess
        processed_df = preprocess_asset_data(symbol, raw_df)
        
        # Save processed data
        save_processed_data(symbol, processed_df)
        
        processed_data[symbol] = processed_df
        
        # Print summary statistics
        print(f"\nSummary Statistics for {symbol}:")
        print(f"  Date range: {processed_df.index[0]} to {processed_df.index[-1]}")
        print(f"  Observations: {len(processed_df)}")
        print(f"  Mean daily return: {processed_df['logReturn'].mean():.6f}")
        print(f"  Std dev (daily): {processed_df['logReturn'].std():.6f}")
        print(f"  Skewness: {processed_df['logReturn'].skew():.4f}")
        print(f"  Excess kurtosis: {processed_df['logReturn'].kurtosis():.4f}")
    
    return processed_data


if __name__ == "__main__":
    # Example: Fetch SPY, AAPL, BTC-USD
    data = fetch_and_process_assets(force_refresh=False)
    print("\n✓ Data acquisition complete!")
