"""Backtest engine: one equity curve, lagged positions, exposure-matched comparison.

Three defects in the previous engine are fixed here by construction rather than
by care, and each has a test named after it in ``tests/test_backtest.py``.

1. Every walk-forward window restarted the portfolio at the initial capital, so
   the stitched series had a fabricated jump at each boundary. That alone made
   the engine report a better Sharpe and a shallower drawdown than buy-and-hold
   for a strategy that was provably identical to buy-and-hold. Here the segments
   are concatenated as RETURNS and the curve is computed once, at the end.
2. Positions were applied on the same bar as the return they scaled. A position
   is now explicitly shifted by ``lag`` bars, defaulting to 1, so the weight
   used on day t is one built from information available at the close of t-1.
3. A vol-targeted book that is capped at full investment spends most of its life
   under-invested, so it has a lower drawdown than the benchmark for free.
   ``match_exposure`` rescales a strategy to a target realised volatility so the
   comparison is between shapes rather than between sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, annualised_vol, equity_curve, summarise


@dataclass
class BacktestResult:
    returns: pd.Series          # strategy simple returns, after costs
    positions: pd.Series        # the weight actually applied to each bar
    gross_returns: pd.Series    # before costs, for attributing the cost drag
    turnover: pd.Series
    curve: pd.Series

    def metrics(self, risk_free_annual: float = 0.0) -> dict:
        out = summarise(self.returns, risk_free_annual)
        out["mean_exposure"] = float(self.positions.mean())
        out["annual_turnover"] = float(self.turnover.sum() / len(self.turnover) * TRADING_DAYS)
        out["cost_drag_annual"] = float(
            (self.gross_returns.mean() - self.returns.mean()) * TRADING_DAYS
        )
        return out


def target_vol_position(
    forecast_vol: pd.Series,
    vol_target: float = 0.15,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Weight that targets ``vol_target`` given a forecast of annualised vol.

    ``max_leverage`` is explicit and defaults to 1.0 (long only, no borrowing),
    which is the old behaviour. Set it above 1 to let the overlay lever up in
    calm periods, which is what makes a vol-target strategy comparable to the
    benchmark rather than simply smaller than it.
    """
    if vol_target <= 0:
        raise ValueError("vol_target must be positive")
    weight = vol_target / forecast_vol.replace(0.0, np.nan)
    return weight.clip(lower=0.0, upper=max_leverage)


def run(
    asset_returns: pd.Series,
    positions: pd.Series,
    lag: int = 1,
    cost_per_unit_turnover: float = 0.0,
) -> BacktestResult:
    """Apply ``positions`` to ``asset_returns``, lagged, with linear costs.

    ``asset_returns`` and ``positions`` must be simple returns and weights on
    the same index. The weight is shifted forward by ``lag`` bars before it
    multiplies anything: setting ``lag=0`` is only ever correct for a signal
    that is genuinely known at the open of the bar it trades, and the tests
    treat lag=0 as the thing to guard against.
    """
    if not asset_returns.index.equals(positions.index):
        raise ValueError("returns and positions must share an index")
    if lag < 0:
        raise ValueError("lag cannot be negative")

    applied = positions.shift(lag)
    # An unknown weight means no position, not a forward-filled stale one.
    applied = applied.fillna(0.0)

    gross = asset_returns * applied
    turnover = applied.diff().abs().fillna(applied.abs())
    net = gross - cost_per_unit_turnover * turnover

    keep = asset_returns.notna()
    return BacktestResult(
        returns=net[keep],
        positions=applied[keep],
        gross_returns=gross[keep],
        turnover=turnover[keep],
        curve=equity_curve(net[keep]),
    )


def match_exposure(
    result: BacktestResult,
    target_vol: float,
) -> tuple[pd.Series, float]:
    """Rescale a strategy's returns to a target realised volatility.

    Returns the rescaled series and the constant multiplier used. This is the
    volatility analogue of comparing two classifiers at the same recall: until
    the two books carry the same risk, a drawdown difference says more about
    position size than about the signal.

    The multiplier is a single number applied to the whole series, computed
    in-sample on the realised vol. That is a deliberate simplification and it
    favours neither side; it is stated in the README rather than hidden.
    """
    realised = annualised_vol(result.returns)
    if not np.isfinite(realised) or realised == 0:
        raise ValueError("cannot rescale a series with zero or undefined volatility")
    scale = target_vol / realised
    return result.returns * scale, float(scale)


def buy_and_hold(asset_returns: pd.Series) -> BacktestResult:
    """The benchmark: fully invested, every day, no trading."""
    ones = pd.Series(1.0, index=asset_returns.index)
    # lag=0 is correct here and only here: holding the asset from the first bar
    # requires no forecast, so there is no information to be early about.
    return run(asset_returns, ones, lag=0, cost_per_unit_turnover=0.0)
