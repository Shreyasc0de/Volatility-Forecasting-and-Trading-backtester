"""
Regime-Switching Layer: 2-State Markov-switching model using hmmlearn
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import warnings

warnings.filterwarnings('ignore')


def fit_markov_switching_model(data: pd.Series, k_regimes: int = 2, order: int = 0) -> dict:
    """
    Fit a k-state Markov-switching model using Gaussian HMM.
    
    Args:
        data: Time series (returns, log returns, or volatility)
        k_regimes: Number of regimes (default 2: calm/turbulent)
        order: Number of AR lags (not used for basic HMM)
    
    Returns:
        Dictionary with model results and smoothed regime probabilities
    """
    print(f"Fitting {k_regimes}-state Markov-switching model ({order} AR lags)...")
    
    # Prepare data - reshape for hmmlearn (must be 2D)
    data_clean = data.dropna().values
    X = data_clean.reshape(-1, 1)
    
    try:
        # Fit Gaussian HMM
        model = hmm.GaussianHMM(n_components=k_regimes, covariance_type="full", n_iter=100)
        model.fit(X)
        
        # Get smoothed probabilities
        smoothed_probs, _ = model.decode(X, algorithm='viterbi')
        # Convert to probabilities
        hidden_states = model.predict(X)
        
        # Compute posterior probabilities
        posteriors = model.predict_proba(X)
        
        # Get transition matrix
        transition_matrix = model.transmat_
        
        return {
            'model': model,
            'result': model,
            'smoothed_probs': posteriors,  # Shape: (T, k_regimes)
            'predicted_state': hidden_states,
            'k_regimes': k_regimes,
            'transition_matrix': transition_matrix,
            'regime_means': model.means_,
            'regime_covars': model.covars_,
            'llf': model.score(X)
        }
    except Exception as e:
        print(f"✗ Markov-switching fit failed: {e}")
        return None


def analyze_regime_persistence(smoothed_probs: np.ndarray, regime_names: list = None) -> dict:
    """
    Analyze regime persistence: duration in each state, transition probabilities.
    
    Args:
        smoothed_probs: Smoothed regime probabilities (T x k_regimes)
        regime_names: Names for regimes (e.g., ['Calm', 'Turbulent'])
    
    Returns:
        Dictionary with regime statistics
    """
    if regime_names is None:
        regime_names = [f'Regime {i}' for i in range(smoothed_probs.shape[1])]
    
    predicted_state = np.argmax(smoothed_probs, axis=1)
    
    stats = {}
    
    for regime in range(smoothed_probs.shape[1]):
        regime_mask = predicted_state == regime
        regime_duration = []
        current_duration = 0
        
        for i, is_regime in enumerate(regime_mask):
            if is_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    regime_duration.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            regime_duration.append(current_duration)
        
        if len(regime_duration) > 0:
            avg_duration = np.mean(regime_duration)
            max_duration = np.max(regime_duration)
        else:
            avg_duration = 0
            max_duration = 0
        
        n_transitions = np.sum(np.diff(predicted_state) != 0)
        freq = np.sum(regime_mask) / len(predicted_state)
        
        stats[regime_names[regime]] = {
            'frequency': freq,
            'avg_duration_days': avg_duration,
            'max_duration_days': max_duration,
            'n_transitions': n_transitions,
            'obs_count': np.sum(regime_mask)
        }
    
    return stats


def detect_regime_shifts(smoothed_probs: np.ndarray, threshold: float = 0.5) -> pd.Series:
    """
    Detect significant regime shifts (crossovers).
    
    Args:
        smoothed_probs: Smoothed regime probabilities
        threshold: Probability threshold for regime assignment
    
    Returns:
        Series with regime labels
    """
    regime_index = np.argmax(smoothed_probs, axis=1)
    
    # Detect transitions
    transitions = np.where(np.diff(regime_index) != 0)[0]
    
    return pd.Series(regime_index, name='regime'), transitions


def regime_conditioned_statistics(data: pd.Series, smoothed_probs: np.ndarray) -> dict:
    """
    Compute regime-conditioned statistics (mean, vol, skew, kurt by regime).
    
    Args:
        data: Time series data
        smoothed_probs: Smoothed regime probabilities
    
    Returns:
        Dictionary with statistics by regime
    """
    predicted_state = np.argmax(smoothed_probs, axis=1)
    data_values = data.values[:len(predicted_state)]
    
    stats = {}
    
    for regime in range(smoothed_probs.shape[1]):
        regime_data = data_values[predicted_state == regime]
        
        if len(regime_data) > 0:
            stats[f'Regime {regime}'] = {
                'mean': np.mean(regime_data),
                'std': np.std(regime_data),
                'skew': pd.Series(regime_data).skew(),
                'kurtosis': pd.Series(regime_data).kurtosis(),
                'min': np.min(regime_data),
                'max': np.max(regime_data),
                'obs_count': len(regime_data)
            }
    
    return stats


def fit_and_analyze_markov_model(returns: pd.Series, symbol: str = "Asset") -> dict:
    """
    Complete Markov-switching analysis pipeline.
    
    Args:
        returns: Log returns or volatility series
        symbol: Asset symbol for labeling
    
    Returns:
        Dictionary with all analysis results
    """
    print(f"\n{'='*60}")
    print(f"Markov Regime Analysis for {symbol}")
    print(f"{'='*60}")
    
    # Fit 2-state model (Calm / Turbulent)
    ms_result = fit_markov_switching_model(returns, k_regimes=2, order=0)
    
    if not ms_result:
        return None
    
    smoothed_probs = ms_result['smoothed_probs']
    predicted_state = np.argmax(smoothed_probs, axis=1)
    
    # Analyze persistence
    regime_names = ['Calm (Low Vol)', 'Turbulent (High Vol)']
    persistence = analyze_regime_persistence(smoothed_probs, regime_names=regime_names)
    
    print(f"\nRegime Persistence:")
    for regime_name, stats in persistence.items():
        print(f"  {regime_name}:")
        print(f"    Frequency: {stats['frequency']:.1%}")
        print(f"    Avg Duration: {stats['avg_duration_days']:.1f} days")
        print(f"    Max Duration: {stats['max_duration_days']} days")
        print(f"    Transitions: {stats['n_transitions']}")
    
    # Regime-conditioned statistics
    cond_stats = regime_conditioned_statistics(returns, smoothed_probs)
    
    print(f"\nRegime-Conditioned Statistics:")
    for regime_name, stats in cond_stats.items():
        print(f"  {regime_name}:")
        print(f"    Mean: {stats['mean']:.6f}")
        print(f"    Std: {stats['std']:.6f}")
        print(f"    Skew: {stats['skew']:.4f}")
        print(f"    Kurtosis: {stats['kurtosis']:.4f}")
    
    # Detect shifts
    regime_series, shifts = detect_regime_shifts(smoothed_probs)
    
    print(f"\nKey Regime Shifts Detected: {len(shifts)} transitions")
    if len(shifts) > 0:
        print(f"  First 5 shift indices: {shifts[:5]}")
    
    return {
        'model_result': ms_result,
        'regime_series': regime_series,
        'shifts': shifts,
        'persistence': persistence,
        'conditioned_stats': cond_stats,
        'symbol': symbol
    }


if __name__ == "__main__":
    from data_pull import fetch_and_process_assets
    
    # Example usage
    data = fetch_and_process_assets(['SPY'])
    if 'SPY' in data:
        # Analyze regime switching on returns
        analysis = fit_and_analyze_markov_model(data['SPY']['logReturn'], symbol='SPY')
