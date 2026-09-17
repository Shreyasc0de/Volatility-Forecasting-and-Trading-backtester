"""Run the whole study and write the reports the README quotes.

    python scripts/backtest.py --symbol SPY

Everything is out-of-sample and time-ordered. The forecast for a bar is built
from strictly earlier bars, the regime probability for a bar is filtered and
refit on strictly earlier bars, and positions are lagged before they scale a
return. One strategy is deliberately built the wrong way, on smoothed regime
probabilities, so the report can put a number on what the original version's
lookahead was worth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from volbt import backtest as B, data as D, metrics as M, regime as R, vol as V  # noqa: E402


def evaluate_forecasters(returns: pd.Series, start: int, refit_every: int,
                         include_garch: bool | None) -> tuple[dict, dict]:
    """Score every forecaster against three targets and return (scores, forecasts).

    ``one_bar`` is the primary ranking: bar t's own squared return, which shares
    no observation with any forecaster's conditioning set. ``forward_21d`` asks
    the same question over a month. ``overlapping_20d`` is the backward-looking
    measure, reported only to show how badly it flatters a rolling forecaster
    whose window it almost duplicates.
    """
    targets = {
        "one_bar": V.squared_return_proxy(returns),
        "forward_21d": V.forward_realised_vol(returns, horizon=21),
        "overlapping_20d": V.realised_vol(returns, window=20),
    }
    scores, forecasts = {}, {}
    for model in V.default_forecasters(include_garch):
        print(f"[vol] {model.name} ...", flush=True)
        forecast = model.forecast_span(returns, start=start, refit_every=refit_every)
        forecasts[model.name] = forecast
        scores[model.name] = {t: V.score_forecasts(target, forecast) for t, target in targets.items()}
        one = scores[model.name]["one_bar"]
        print(f"      QLIKE {one['qlike']:+.4f}  bias {one['bias']:+.4f}  "
              f"corr {one['corr']:+.3f}  n={one['n']}")
    return scores, forecasts


def build_strategies(returns: pd.Series, forecast: pd.Series, filtered: pd.Series,
                     smoothed: pd.Series, vol_target: float, max_leverage: float,
                     cost: float, lag: int) -> dict[str, B.BacktestResult]:
    """Every book the report compares, including one built wrongly on purpose."""
    ones = pd.Series(1.0, index=returns.index)
    vol_weight = B.target_vol_position(forecast, vol_target, max_leverage)
    calm_exposure = R.exposure_from_regime(filtered)
    peek_exposure = R.exposure_from_regime(smoothed)

    return {
        "buy_and_hold": B.buy_and_hold(returns),
        "vol_target": B.run(returns, vol_weight, lag=lag, cost_per_unit_turnover=cost),
        "regime_only": B.run(returns, ones * calm_exposure, lag=lag, cost_per_unit_turnover=cost),
        "vol_and_regime": B.run(returns, vol_weight * calm_exposure, lag=lag,
                                cost_per_unit_turnover=cost),
        # Built on smoothed probabilities, which use the whole sample. Reported
        # so the README can quantify the original version's lookahead rather
        # than merely describe it. Never treat this row as a result.
        "vol_and_regime_LOOKAHEAD": B.run(returns, vol_weight * peek_exposure, lag=lag,
                                          cost_per_unit_turnover=cost),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--initial", type=int, default=504, help="bars before the first forecast")
    p.add_argument("--refit-every", type=int, default=21, help="bars between parameter refits")
    p.add_argument("--regime-refit-every", type=int, default=63)
    p.add_argument("--vol-target", type=float, default=0.15)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--cost", type=float, default=0.0005, help="cost per unit of turnover")
    p.add_argument("--risk-free", type=float, default=0.0, help="annual risk-free rate")
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--no-garch", action="store_true")
    p.add_argument("--no-fetch", action="store_true")
    args = p.parse_args()

    prices = D.load(args.symbol, args.data_dir, args.start, args.end,
                    allow_fetch=not args.no_fetch)
    returns = D.simple_returns(prices)
    facts = D.summarise(returns)
    print(f"[data] {args.symbol}: {facts['n_days']} days, "
          f"{facts['first_day']} to {facts['last_day']}, "
          f"annualised vol {facts['annualised_vol']:.1%}")

    include_garch = False if args.no_garch else None
    scores, forecasts = evaluate_forecasters(returns, args.initial, args.refit_every, include_garch)
    best_vol = min(scores, key=lambda k: scores[k]["one_bar"]["qlike"])
    print(f"[vol] best by QLIKE: {best_vol}")

    print("[regime] walk-forward filtered fit ...", flush=True)
    filtered = R.walk_forward_filtered(returns, args.initial, args.regime_refit_every)["turbulent"]
    full_model = R.fit(returns)
    smoothed = R.smoothed_probabilities(full_model, returns)["turbulent"]

    # Restrict every strategy to the span where all signals exist.
    span = returns.index[args.initial:]
    strategies = build_strategies(
        returns.loc[span], forecasts[best_vol].loc[span], filtered.loc[span],
        smoothed.loc[span], args.vol_target, args.max_leverage, args.cost, args.lag,
    )

    bench_vol = M.annualised_vol(strategies["buy_and_hold"].returns)
    table = {}
    for name, result in strategies.items():
        raw = result.metrics(args.risk_free)
        matched_returns, scale = B.match_exposure(result, bench_vol)
        raw["matched_scale"] = scale
        raw["matched_max_drawdown"] = M.max_drawdown(matched_returns)
        # Sharpe is scale-invariant, so matching cannot change it and there is
        # no "matched Sharpe" worth reporting. Drawdown is what matching fixes:
        # an under-invested book has a shallower one for free, and the matched
        # column removes that free lunch from the comparison.
        raw["matched_sharpe_equals_sharpe"] = bool(
            abs(M.sharpe(matched_returns, args.risk_free) - raw["sharpe"]) < 1e-9
        )
        table[name] = raw
        print(f"[bt] {name:26} Sharpe {raw['sharpe']:+.3f}  "
              f"MDD {raw['max_drawdown']:+7.1%}  "
              f"MDD at matched risk {raw['matched_max_drawdown']:+7.1%}  "
              f"exposure {raw['mean_exposure']:.2f}")

    honest = table["vol_and_regime"]["sharpe"]
    peeking = table["vol_and_regime_LOOKAHEAD"]["sharpe"]
    print(f"[bt] lookahead is worth {peeking - honest:+.3f} Sharpe "
          f"({100 * (peeking - honest) / abs(honest):+.0f}%)")

    # One directory per symbol: a single shared metrics.json meant each run
    # silently overwrote the last, so only the final symbol survived.
    reports = ROOT / "reports" / args.symbol
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": args.symbol,
        "config": vars(args),
        "series": facts,
        "vol_forecast_scores": scores,
        "best_vol_forecaster": best_vol,
        "regime": full_model.persistence()
        | {"log_likelihood": full_model.log_likelihood, "converged": full_model.converged},
        "strategies": table,
        "lookahead_sharpe_gain": float(peeking - honest),
        "lookahead_sharpe_gain_pct": float(100 * (peeking - honest) / abs(honest)),
        "benchmark_annualised_vol": float(bench_vol),
    }
    (reports / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    curves = pd.DataFrame({name: r.curve for name, r in strategies.items()})
    curves.to_csv(reports / "equity_curves.csv")
    pd.DataFrame({"filtered": filtered, "smoothed": smoothed}).dropna().to_csv(
        reports / "regime_probabilities.csv"
    )
    pd.DataFrame(forecasts).to_csv(reports / "vol_forecasts.csv")

    print(f"[done] wrote {reports}/metrics.json and 3 CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
