"""Two-state Gaussian hidden Markov model, with filtered and smoothed states.

This module exists because of a specific bug. The previous version fitted a
Markov-switching model and then fed the SMOOTHED state probabilities into the
backtest as a trading signal. Smoothed probabilities are produced by the
forward-backward pass, so the probability assigned to bar t uses every
observation in the sample, including everything after t. Trading on them means
the strategy knows the turbulent period is coming, which is lookahead in its
purest form, and it can manufacture most of a Sharpe improvement on its own.
The README meanwhile claimed "no lookahead bias".

Both quantities are offered here, clearly named, and the backtest driver takes
``filtered`` only. ``test_regime.py`` asserts the difference that matters: a
later observation can change a smoothed probability and cannot change a
filtered one.

Implemented directly in numpy rather than via statsmodels, for two reasons.
The filtered pass is the object the backtest needs and is a few lines once the
recursion is explicit, and a dependency whose default output is the wrong one
for this purpose is a bad thing to build a trading signal on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_VAR = 1e-12


@dataclass
class RegimeModel:
    """A fitted 2-state model. State 1 is always the higher-variance state.

    Ordering the states by variance matters: an EM fit can land on either
    labelling, and a strategy that reduces exposure in "state 1" would do the
    opposite of what was intended half the time it was refit.
    """

    means: np.ndarray            # shape (2,)
    variances: np.ndarray        # shape (2,)
    transition: np.ndarray       # shape (2, 2), row i = from state i
    initial: np.ndarray          # shape (2,)
    log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def calm(self) -> int:
        return 0

    @property
    def turbulent(self) -> int:
        return 1

    def persistence(self) -> dict:
        """Expected duration of each state, in bars, from its self-transition."""
        stay = np.diag(self.transition)
        with np.errstate(divide="ignore"):
            duration = np.where(stay < 1.0, 1.0 / (1.0 - stay), np.inf)
        return {
            "p_stay_calm": float(stay[0]),
            "p_stay_turbulent": float(stay[1]),
            "expected_duration_calm": float(duration[0]),
            "expected_duration_turbulent": float(duration[1]),
            "vol_ratio": float(np.sqrt(self.variances[1] / self.variances[0])),
        }


def _emission_density(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Gaussian densities, shape (T, 2)."""
    var = np.maximum(variances, MIN_VAR)
    z = (x[:, None] - means[None, :]) ** 2 / var[None, :]
    return np.exp(-0.5 * z) / np.sqrt(2.0 * np.pi * var[None, :])


def _forward(density: np.ndarray, transition: np.ndarray, initial: np.ndarray):
    """Scaled forward pass. Returns (alpha, scaling, log_likelihood).

    ``alpha[t]`` is P(state at t | observations up to and including t), which is
    the filtered probability. Nothing after t enters it.
    """
    T = len(density)
    alpha = np.zeros((T, 2))
    scale = np.zeros(T)

    a = initial * density[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0] if scale[0] > 0 else np.array([0.5, 0.5])
    for t in range(1, T):
        a = (alpha[t - 1] @ transition) * density[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t] if scale[t] > 0 else np.array([0.5, 0.5])
    return alpha, scale, float(np.sum(np.log(np.maximum(scale, 1e-300))))


def _backward(density: np.ndarray, transition: np.ndarray, scale: np.ndarray) -> np.ndarray:
    T = len(density)
    beta = np.zeros((T, 2))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = transition @ (density[t + 1] * beta[t + 1])
        if scale[t + 1] > 0:
            beta[t] /= scale[t + 1]
    return beta


def fit(returns: pd.Series, max_iter: int = 200, tol: float = 1e-8,
        seed: int = 7) -> RegimeModel:
    """Baum-Welch EM for a 2-state Gaussian HMM on the return series."""
    x = returns.dropna().to_numpy(dtype=float)
    if len(x) < 50:
        raise ValueError(f"need at least 50 observations to fit two states, got {len(x)}")

    # Seed the two states from the low and high halves of squared returns, which
    # is a far better starting point than random and makes the fit reproducible.
    order = np.argsort(np.abs(x - x.mean()))
    quiet, loud = x[order[: len(x) // 2]], x[order[len(x) // 2:]]
    means = np.array([quiet.mean(), loud.mean()])
    variances = np.array([max(quiet.var(ddof=1), MIN_VAR), max(loud.var(ddof=1), MIN_VAR)])
    transition = np.array([[0.95, 0.05], [0.10, 0.90]])
    initial = np.array([0.5, 0.5])

    previous_ll, converged = -np.inf, False
    for iteration in range(1, max_iter + 1):
        density = _emission_density(x, means, variances)
        alpha, scale, ll = _forward(density, transition, initial)
        beta = _backward(density, transition, scale)

        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

        xi = np.zeros((2, 2))
        for t in range(len(x) - 1):
            num = (alpha[t][:, None] * transition
                   * (density[t + 1] * beta[t + 1])[None, :])
            total = num.sum()
            if total > 0:
                xi += num / total

        initial = gamma[0]
        transition = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        weight = gamma.sum(axis=0)
        means = (gamma * x[:, None]).sum(axis=0) / np.maximum(weight, 1e-300)
        variances = np.maximum(
            (gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / np.maximum(weight, 1e-300),
            MIN_VAR,
        )

        if abs(ll - previous_ll) < tol * max(1.0, abs(previous_ll)):
            converged = True
            previous_ll = ll
            break
        previous_ll = ll

    # Relabel so state 1 is always the turbulent one.
    if variances[0] > variances[1]:
        means, variances = means[::-1], variances[::-1]
        transition = transition[::-1, ::-1]
        initial = initial[::-1]

    return RegimeModel(means, variances, transition, initial,
                       float(previous_ll), iteration, converged)


def filtered_probabilities(model: RegimeModel, returns: pd.Series) -> pd.DataFrame:
    """P(state at t | data up to t). The only version safe to trade on.

    Column ``turbulent`` is what the backtest reduces exposure against.
    """
    r = returns.dropna()
    density = _emission_density(r.to_numpy(dtype=float), model.means, model.variances)
    alpha, _, _ = _forward(density, model.transition, model.initial)
    return pd.DataFrame(alpha, index=r.index, columns=["calm", "turbulent"])


def smoothed_probabilities(model: RegimeModel, returns: pd.Series) -> pd.DataFrame:
    """P(state at t | ALL data, including bars after t).

    Correct for describing history and wrong for trading it. Provided so the
    README can show the difference, and named so that using it by accident
    requires typing the word "smoothed".
    """
    r = returns.dropna()
    density = _emission_density(r.to_numpy(dtype=float), model.means, model.variances)
    alpha, scale, _ = _forward(density, model.transition, model.initial)
    beta = _backward(density, model.transition, scale)
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    return pd.DataFrame(gamma, index=r.index, columns=["calm", "turbulent"])


def walk_forward_filtered(
    returns: pd.Series, start: int, refit_every: int = 63
) -> pd.DataFrame:
    """Filtered turbulent probability for each bar, refitting as the window grows.

    For bar i the model parameters come from ``returns[:i]`` and the filter is
    run over ``returns[:i]``, so the probability attached to bar i uses nothing
    from bar i onward. That is stricter than the filtered pass alone, which
    would still let bar i's own return inform bar i's state.
    """
    r = returns.dropna()
    if len(r) <= start:
        raise ValueError(f"need more than {start} observations, got {len(r)}")

    out = pd.DataFrame(index=r.index, columns=["calm", "turbulent"], dtype=float)
    model = None
    for i in range(start, len(r)):
        if model is None or (i - start) % refit_every == 0:
            model = fit(r.iloc[:i])
        probs = filtered_probabilities(model, r.iloc[:i])
        out.iloc[i] = probs.iloc[-1].to_numpy()
    return out


def exposure_from_regime(turbulent_prob: pd.Series, floor: float = 0.0) -> pd.Series:
    """Linear exposure: full in calm, ``floor`` when certainly turbulent."""
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must lie in [0, 1]")
    return (1.0 - turbulent_prob).clip(lower=0.0, upper=1.0) * (1.0 - floor) + floor
