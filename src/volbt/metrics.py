"""Performance metrics, with the return convention stated rather than assumed.

The previous version of this project mixed conventions: it produced log returns
upstream and then compounded them as if they were simple returns, and it
computed an annualised return by raising ``1 + mean(log return)`` to the 252nd
power. Both are wrong, and the error grows with volatility, so it mattered most
for the asset with the highest volatility.

Everything here takes SIMPLE returns. ``from_log`` converts once, at the edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def from_log(log_returns: pd.Series) -> pd.Series:
    """Simple returns from log returns: r = exp(l) - 1."""
    return np.expm1(log_returns)


def to_log(simple_returns: pd.Series) -> pd.Series:
    return np.log1p(simple_returns)


def equity_curve(simple_returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Cumulative wealth from simple returns, starting at ``initial``.

    One continuous curve. The old engine multiplied each walk-forward window by
    a fixed starting capital, so the stitched series reset every window and
    produced a fictitious jump at each boundary.
    """
    return initial * (1.0 + simple_returns).cumprod()


def max_drawdown(simple_returns: pd.Series) -> float:
    curve = equity_curve(simple_returns)
    return float((curve / curve.cummax() - 1.0).min())


def annualised_return(simple_returns: pd.Series) -> float:
    """Geometric (CAGR-style) annualised return, not a compounded arithmetic mean."""
    n = len(simple_returns)
    if n == 0:
        return float("nan")
    total_growth = float((1.0 + simple_returns).prod())
    if total_growth <= 0:
        return -1.0
    return total_growth ** (TRADING_DAYS / n) - 1.0


def annualised_vol(simple_returns: pd.Series) -> float:
    return float(simple_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _degenerate_spread(excess: pd.Series) -> bool:
    """True when the dispersion is numerically indistinguishable from zero.

    An exact ``sd == 0`` test is not enough: numpy's standard deviation of 100
    identical values comes back as 2.2e-19 rather than 0.0, which turned a
    constant return series into a Sharpe of 7.3e16. A strategy that is flat
    because its position is zero everywhere is exactly the case that reaches
    this, so the guard is relative to the mean's own magnitude.
    """
    sd = float(excess.std(ddof=1))
    if not np.isfinite(sd):
        return True
    scale = max(abs(float(excess.mean())), 1e-12)
    return sd <= 1e-15 * scale


def sharpe(simple_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe on excess returns.

    The risk-free rate is an explicit argument because a Sharpe quoted without
    one is a different statistic, and over 2020 to 2025 the difference is not
    cosmetic: cash paid roughly nothing in 2021 and over 5% in 2023.
    """
    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = simple_returns - daily_rf
    if _degenerate_spread(excess):
        return float("nan")
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(simple_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = simple_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) < 2 or _degenerate_spread(downside):
        return float("nan")
    return float(excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def summarise(simple_returns: pd.Series, risk_free_annual: float = 0.0) -> dict:
    """The metric block every strategy is reported with."""
    returns = simple_returns.dropna()
    mdd = max_drawdown(returns)
    ann_ret = annualised_return(returns)
    return {
        "n_days": int(len(returns)),
        "total_return": float((1.0 + returns).prod() - 1.0),
        "annualised_return": ann_ret,
        "annualised_vol": annualised_vol(returns),
        "sharpe": sharpe(returns, risk_free_annual),
        "sortino": sortino(returns, risk_free_annual),
        "max_drawdown": mdd,
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else float("nan"),
        "hit_rate": float((returns > 0).mean()),
        "risk_free_annual": risk_free_annual,
    }
