"""Volatility forecasters, each producing genuine one-step-ahead numbers.

The previous version fit each GARCH variant on the training slice and then
evaluated ``conditional_vol.iloc[n_train:]`` against the test set. A series fit
to 800 observations has 800 entries, so that slice was empty and every
out-of-sample comparison the README reported ran on nothing at all.

Patching the index would not have been enough, because the underlying object
was also wrong: the in-sample conditional volatility of a model fitted to the
whole training set is not a forecast. So the interface here is different. A
forecaster is handed the return history and an index, and must return, for each
bar from that index onward, a number built only from strictly earlier bars.
``test_vol.py`` asserts that property directly by perturbing a future return
and checking that earlier forecasts do not move.

Parameter estimation for the GARCH family needs the ``arch`` package. The
variance recursions that turn those parameters into one-step-ahead forecasts
are implemented and tested here, so the part that can silently corrupt a
backtest does not depend on an untested library path.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

TRADING_DAYS = 252
#: E|z| for a standard normal, used by the EGARCH recursion.
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


class Forecaster(Protocol):
    name: str

    def forecast_span(self, returns: pd.Series, start: int, refit_every: int) -> pd.Series: ...


def arch_available() -> bool:
    try:
        import arch  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------- window statistics


class _WindowStat:
    """Forecasters that are a statistic of the recent past, with nothing to fit."""

    name = "window"

    def _value(self, history: np.ndarray) -> float:
        raise NotImplementedError

    def forecast_span(self, returns: pd.Series, start: int, refit_every: int = 1) -> pd.Series:
        r = returns.dropna()
        _check_span(r, start)
        out = pd.Series(index=r.index, dtype=float)
        values = r.to_numpy()
        for i in range(start, len(r)):
            out.iloc[i] = self._value(values[:i])  # strictly before bar i
        return out


class ConstantVol(_WindowStat):
    """Standard deviation of everything seen so far. The floor for any forecast."""

    name = "constant"

    def _value(self, history: np.ndarray) -> float:
        if len(history) < 2:
            return float("nan")
        return float(np.std(history, ddof=1) * np.sqrt(TRADING_DAYS))


class RollingWindowVol(_WindowStat):
    def __init__(self, window: int = 20):
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.name = f"rolling_{window}d"

    def _value(self, history: np.ndarray) -> float:
        tail = history[-self.window:]
        if len(tail) < 2:
            return float("nan")
        return float(np.std(tail, ddof=1) * np.sqrt(TRADING_DAYS))


class EwmaVol(_WindowStat):
    """RiskMetrics EWMA at lambda 0.94. The baseline a GARCH has to beat."""

    def __init__(self, lam: float = 0.94, seed_window: int = 60):
        if not 0.0 < lam < 1.0:
            raise ValueError("lam must lie in (0, 1)")
        self.lam = lam
        self.seed_window = seed_window
        self.name = f"ewma_{lam}"

    def _value(self, history: np.ndarray) -> float:
        if len(history) < 2:
            return float("nan")
        var = float(np.var(history[: min(len(history), self.seed_window)], ddof=1))
        for x in history:
            var = self.lam * var + (1.0 - self.lam) * x * x
        return float(np.sqrt(var * TRADING_DAYS))


# ------------------------------------------------------------- GARCH family


def garch_filter(resid: np.ndarray, omega: float, alpha: float, beta: float,
                 gamma: float = 0.0) -> np.ndarray:
    """GARCH(1,1) and GJR-GARCH(1,1,1) conditional variance recursion.

    sigma2[t] = omega + alpha*e[t-1]^2 + gamma*e[t-1]^2*1{e[t-1]<0} + beta*sigma2[t-1]

    Returns an array of length ``len(resid) + 1``: the conditional variance of
    each observed bar, and in the final slot the one-step-ahead forecast for
    the bar after the data ends. That last element is the only thing a position
    sized at the close may use.
    """
    denom = 1.0 - alpha - 0.5 * gamma - beta
    sigma2 = np.empty(len(resid) + 1)
    sigma2[0] = omega / denom if denom > 1e-8 else float(np.var(resid))
    for t in range(len(resid)):
        e = resid[t]
        shock = alpha * e * e + (gamma * e * e if e < 0 else 0.0)
        sigma2[t + 1] = omega + shock + beta * sigma2[t]
    return sigma2


def egarch_filter(resid: np.ndarray, omega: float, alpha: float, beta: float,
                  gamma: float = 0.0) -> np.ndarray:
    """EGARCH(1,1) recursion in arch's parameterisation.

    ln sigma2[t] = omega + alpha*(|z[t-1]| - E|z|) + gamma*z[t-1] + beta*ln sigma2[t-1]

    Same return contract as ``garch_filter``: one extra slot on the end holding
    the one-step-ahead forecast.
    """
    log_var = np.empty(len(resid) + 1)
    log_var[0] = np.log(max(float(np.var(resid)), 1e-12)) if len(resid) else 0.0
    for t in range(len(resid)):
        sigma = np.sqrt(np.exp(log_var[t]))
        z = resid[t] / sigma if sigma > 0 else 0.0
        log_var[t + 1] = (omega + alpha * (abs(z) - SQRT_2_OVER_PI)
                          + gamma * z + beta * log_var[t])
    return np.exp(np.clip(log_var, -50.0, 50.0))


class GarchVol:
    """GARCH(1,1), EGARCH(1,1) or GJR-GARCH(1,1,1).

    Parameters are re-estimated every ``refit_every`` bars using ``arch``.
    Between re-estimations the variance is filtered forward with those fixed
    parameters over the actual history, which is what makes each day's number a
    fresh forecast rather than a repeat of the last fitted value.
    """

    SPECS = {
        "garch": dict(vol="GARCH", p=1, q=1, o=0),
        "egarch": dict(vol="EGARCH", p=1, q=1, o=1),
        "gjr": dict(vol="GARCH", p=1, q=1, o=1),
    }

    def __init__(self, kind: str = "garch", scale: float = 100.0):
        if kind not in self.SPECS:
            raise ValueError(f"kind must be one of {sorted(self.SPECS)}")
        self.kind = kind
        self.name = kind
        # arch's optimiser behaves far better on percentage returns.
        self.scale = scale

    def _estimate(self, train: np.ndarray) -> dict:
        from arch import arch_model

        res = arch_model(train, mean="Constant", dist="normal", **self.SPECS[self.kind]).fit(
            disp="off", show_warning=False
        )
        p = res.params
        return {
            "mu": float(p.get("mu", 0.0)),
            "omega": float(p["omega"]),
            "alpha": float(p.get("alpha[1]", 0.0)),
            "beta": float(p.get("beta[1]", 0.0)),
            "gamma": float(p.get("gamma[1]", 0.0)),
        }

    def _forecast_from(self, params: dict, history: np.ndarray) -> float:
        resid = history - params["mu"]
        filt = egarch_filter if self.kind == "egarch" else garch_filter
        sigma2 = filt(resid, params["omega"], params["alpha"], params["beta"], params["gamma"])
        daily_var = sigma2[-1] / (self.scale**2)
        return float(np.sqrt(max(daily_var, 0.0) * TRADING_DAYS))

    def forecast_span(self, returns: pd.Series, start: int, refit_every: int = 21) -> pd.Series:
        r = returns.dropna()
        _check_span(r, start)
        if refit_every < 1:
            raise ValueError("refit_every must be at least 1")

        scaled = r.to_numpy() * self.scale
        out = pd.Series(index=r.index, dtype=float)
        params = None
        for i in range(start, len(r)):
            if params is None or (i - start) % refit_every == 0:
                params = self._estimate(scaled[:i])
            out.iloc[i] = self._forecast_from(params, scaled[:i])
        return out


# ------------------------------------------------------------------ plumbing


def _check_span(r: pd.Series, start: int) -> None:
    if start < 2:
        raise ValueError("start must be at least 2")
    if len(r) <= start:
        raise ValueError(f"need more than {start} observations, got {len(r)}")


def default_forecasters(include_garch: bool | None = None) -> list[Forecaster]:
    """Every forecaster the report compares, baselines first."""
    models: list[Forecaster] = [ConstantVol(), RollingWindowVol(20), RollingWindowVol(60), EwmaVol(0.94)]
    use_garch = arch_available() if include_garch is None else include_garch
    if use_garch:
        models += [GarchVol("garch"), GarchVol("egarch"), GarchVol("gjr")]
    return models


def realised_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Backward-looking realised volatility over the ``window`` bars ending at t.

    Useful for plotting and for describing what volatility did. NOT a fair
    scoring target for a one-step-ahead forecast: at bar t it covers t-window+1
    through t, which overlaps almost entirely with the window a rolling
    forecaster used to produce its number for bar t. Scoring a rolling-20d
    forecast against a rolling-20d realised measure compares a statistic with
    itself shifted by one day, and the resulting correlation near 0.97 says
    nothing about skill. ``squared_return_proxy`` is the scoring target.
    """
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def squared_return_proxy(returns: pd.Series) -> pd.Series:
    """|r_t| annualised: the unbiased one-bar proxy for bar t's volatility.

    Noisy by construction, since a single squared return is a one-observation
    estimate of a variance. That noise is exactly what QLIKE is robust to: it
    is minimised in expectation by the true conditional variance even when the
    proxy is this rough, which is why it is the standard loss for comparing
    one-step-ahead variance forecasts. Bar t's proxy uses bar t only, so it
    shares nothing with the history any forecaster conditioned on.
    """
    return returns.abs() * np.sqrt(TRADING_DAYS)


def forward_realised_vol(returns: pd.Series, horizon: int = 21) -> pd.Series:
    """Realised volatility over the ``horizon`` bars STARTING at t.

    A second, less noisy target, for the question "how volatile will the next
    month be". It uses only bars t and later, so it too shares nothing with a
    forecaster's conditioning set. Overlapping windows make consecutive values
    highly autocorrelated, so treat differences between close scores here with
    more caution than the one-bar proxy.
    """
    reversed_roll = returns[::-1].rolling(horizon).std(ddof=1)[::-1]
    return reversed_roll * np.sqrt(TRADING_DAYS)


def qlike(realised_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss, lower is better.

    Preferred to RMSE on volatility because it is minimised in expectation by
    the true conditional variance even when the realised proxy is noisy.
    """
    keep = (forecast_var > 0) & np.isfinite(forecast_var) & np.isfinite(realised_var)
    if keep.sum() == 0:
        return float("nan")
    rv, fv = realised_var[keep], forecast_var[keep]
    return float(np.mean(np.log(fv) + rv / fv))


def _safe_corr(a: np.ndarray, f: np.ndarray) -> float:
    """Correlation that returns NaN instead of warning when either side is flat."""
    if len(a) < 3 or np.std(a) == 0 or np.std(f) == 0:
        return float("nan")
    return float(np.corrcoef(a, f)[0, 1])


def score_forecasts(realised: pd.Series, forecast: pd.Series) -> dict:
    """Score a forecast against a realised proxy on their overlapping dates."""
    joined = pd.DataFrame({"realised": realised, "forecast": forecast}).dropna()
    if joined.empty:
        raise ValueError(
            "no overlapping dates between the realised and forecast series. This is "
            "the exact failure the previous version hid: it scored an empty slice and "
            "published the result as an out-of-sample comparison."
        )
    a, f = joined["realised"].to_numpy(), joined["forecast"].to_numpy()
    return {
        "n": int(len(joined)),
        "rmse": float(np.sqrt(np.mean((a - f) ** 2))),
        "mae": float(np.mean(np.abs(a - f))),
        "qlike": qlike(a**2, f**2),
        "bias": float(np.mean(f - a)),
        "corr": _safe_corr(a, f),
    }
