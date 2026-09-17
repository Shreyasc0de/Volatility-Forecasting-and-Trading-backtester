"""Re-derive every figure quoted in README.md from reports/ and fail on drift.

    python scripts/verify_readme.py

Each quoted value is passed as the STRING the README prints, so trailing zeros
survive and the tolerance is half a unit of the last printed digit. Passing
0.880 as a number would arrive as "0.88" and silently claim a tolerance ten
times looser.

Four numbers in the README's correction section describe the behaviour of the
previous implementation, which no longer exists in the repository. They cannot
be re-derived from anything here and this script says so explicitly rather than
letting the gap pass unnoticed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS = ("SPY", "AAPL", "BTC-USD")
MODELS = ("constant", "rolling_20d", "rolling_60d", "ewma_0.94", "garch", "egarch", "gjr")
GARCH = ("garch", "egarch", "gjr")

checks = 0
failures: list[str] = []


def near(label: str, actual: float, quoted: str) -> None:
    global checks
    checks += 1
    text = quoted.replace(",", "").replace("%", "").lstrip("+")
    dot = text.find(".")
    decimals = 0 if dot == -1 else len(text) - dot - 1
    tolerance = 0.5 * 10**-decimals
    expected = float(text)
    if not abs(actual - expected) <= tolerance:
        failures.append(f"{label}: README says {quoted}, reports give {actual:,.6g} "
                        f"(tolerance {tolerance:g})")


def exact(label: str, actual, quoted) -> None:
    global checks
    checks += 1
    if actual != quoted:
        failures.append(f"{label}: README says {quoted!r}, reports give {actual!r}")


def load() -> dict:
    out = {}
    for symbol in SYMBOLS:
        path = ROOT / "reports" / symbol / "metrics.json"
        if not path.exists():
            sys.exit(f"missing {path}. Run scripts/backtest.py --symbol {symbol} first.")
        out[symbol] = json.loads(path.read_text())
    return out


data = load()


def qlike(symbol: str, model: str, target: str) -> float:
    return data[symbol]["vol_forecast_scores"][model][target]["qlike"]


def ranking(symbol: str, target: str) -> list[str]:
    return sorted(MODELS, key=lambda m: qlike(symbol, m, target))


def strat(symbol: str, name: str, field: str) -> float:
    return data[symbol]["strategies"][name][field]


# --------------------------------------------------------------- the data set

for symbol in SYMBOLS:
    config = data[symbol]["config"]
    exact(f"{symbol} initial training window", config["initial"], 504)
    exact(f"{symbol} volatility refit interval", config["refit_every"], 21)
    exact(f"{symbol} regime refit interval", config["regime_refit_every"], 63)
    near(f"{symbol} cost per unit turnover", config["cost"] * 10000, "5")
    exact(f"{symbol} positions are lagged one bar", config["lag"], 1)
    exact(f"{symbol} leverage cap", config["max_leverage"], 1.0)
    exact(f"{symbol} all three GARCH arms ran",
          all(m in data[symbol]["vol_forecast_scores"] for m in GARCH), True)

exact("first day", data["SPY"]["series"]["first_day"], "2021-08-31")
exact("SPY last day", data["SPY"]["series"]["last_day"], "2026-08-28")
exact("BTC last day", data["BTC-USD"]["series"]["last_day"], "2026-08-30")
near("SPY out-of-sample bars", strat("SPY", "buy_and_hold", "n_days"), "750")
near("AAPL out-of-sample bars", strat("AAPL", "buy_and_hold", "n_days"), "750")
near("BTC out-of-sample bars", strat("BTC-USD", "buy_and_hold", "n_days"), "1321")
near("SPY annualised vol", data["SPY"]["series"]["annualised_vol"] * 100, "17.2")
near("AAPL annualised vol", data["AAPL"]["series"]["annualised_vol"] * 100, "28.0")
near("BTC annualised vol", data["BTC-USD"]["series"]["annualised_vol"] * 100, "43.3")
near("AAPL excess kurtosis", data["AAPL"]["series"]["excess_kurtosis"], "6.7")
near("BTC excess kurtosis", data["BTC-USD"]["series"]["excess_kurtosis"], "4.0")

# ------------------------------------------------ finding 1: the proxy decides

overlapping_corr = {"SPY": "0.983", "AAPL": "0.978", "BTC-USD": "0.974"}
one_bar_corr = {"SPY": "0.229", "AAPL": "0.186", "BTC-USD": "0.139"}
rolling_rank = {"SPY": 5, "AAPL": 6, "BTC-USD": 7}
for symbol in SYMBOLS:
    scores = data[symbol]["vol_forecast_scores"]["rolling_20d"]
    near(f"{symbol} rolling-20d correlation, overlapping proxy",
         scores["overlapping_20d"]["corr"], overlapping_corr[symbol])
    near(f"{symbol} rolling-20d correlation, one-bar proxy",
         scores["one_bar"]["corr"], one_bar_corr[symbol])
    exact(f"{symbol} rolling-20d ranks first under the overlapping proxy",
          ranking(symbol, "overlapping_20d")[0], "rolling_20d")
    exact(f"{symbol} rolling-20d rank under the one-bar proxy",
          ranking(symbol, "one_bar").index("rolling_20d") + 1, rolling_rank[symbol])

for target in ("one_bar", "forward_21d"):
    for symbol in SYMBOLS:
        exact(f"{symbol} GARCH family takes the top three under {target}",
              set(ranking(symbol, target)[:3]), set(GARCH))

garch_ranks = [ranking(s, "overlapping_20d").index(m) + 1 for s in SYMBOLS for m in GARCH]
exact("GARCH family sat third to sixth under the overlapping proxy",
      (min(garch_ranks), max(garch_ranks)), (3, 6))

table = {
    "gjr": ("-3.0512", "-1.7711", "-0.9189"),
    "egarch": ("-3.0450", "-1.7763", "-0.9262"),
    "garch": ("-2.9829", "-1.7470", "-0.9217"),
    "ewma_0.94": ("-2.9246", "-1.7094", "-0.8904"),
    "rolling_60d": ("-2.8305", "-1.6607", "-0.8559"),
    "rolling_20d": ("-2.8444", "-1.6234", "-0.7816"),
    "constant": ("-2.6887", "-1.6135", "-0.8296"),
}
for model, values in table.items():
    for symbol, quoted in zip(SYMBOLS, values):
        near(f"{symbol} {model} QLIKE, one-bar proxy", qlike(symbol, model, "one_bar"), quoted)

exact("SPY is the asset where the rolling windows swap",
      qlike("SPY", "rolling_20d", "one_bar") < qlike("SPY", "rolling_60d", "one_bar"), True)
exact("the constant is last on SPY", ranking("SPY", "one_bar")[-1], "constant")
exact("the constant is NOT last on BTC", ranking("BTC-USD", "one_bar")[-1], "rolling_20d")

exact("SPY winner", data["SPY"]["best_vol_forecaster"], "gjr")
exact("AAPL winner", data["AAPL"]["best_vol_forecaster"], "egarch")
exact("BTC winner", data["BTC-USD"]["best_vol_forecaster"], "egarch")
exact("BTC ordering within the GARCH family",
      [m for m in ranking("BTC-USD", "one_bar") if m in GARCH], ["egarch", "garch", "gjr"])

spread = {s: max(qlike(s, m, "one_bar") for m in GARCH) - min(qlike(s, m, "one_bar") for m in GARCH)
          for s in SYMBOLS}
near("GARCH family spread on SPY", spread["SPY"], "0.068")
near("GARCH family spread on AAPL", spread["AAPL"], "0.029")
near("GARCH family spread on BTC", spread["BTC-USD"], "0.007")

# ------------------------------------------- finding 2: the overlay underperforms

sharpe_table = {
    "buy_and_hold": ("+1.332", "+0.816", "+0.838"),
    "vol_target": ("+1.280", "+0.607", "+0.795"),
    "regime_only": ("+0.996", "+0.296", "+0.575"),
    "vol_and_regime": ("+0.953", "+0.191", "+0.538"),
}
for name, values in sharpe_table.items():
    for symbol, quoted in zip(SYMBOLS, values):
        near(f"{symbol} {name} Sharpe", strat(symbol, name, "sharpe"), quoted)

for symbol in SYMBOLS:
    bench = strat(symbol, "buy_and_hold", "sharpe")
    for name in ("vol_target", "regime_only", "vol_and_regime"):
        exact(f"{symbol} {name} is worse than buy and hold",
              strat(symbol, name, "sharpe") < bench, True)

spy_gap = 100 * (strat("SPY", "buy_and_hold", "sharpe") - strat("SPY", "vol_target", "sharpe")) \
    / strat("SPY", "buy_and_hold", "sharpe")
aapl_gap = 100 * (strat("AAPL", "buy_and_hold", "sharpe") - strat("AAPL", "vol_and_regime", "sharpe")) \
    / strat("AAPL", "buy_and_hold", "sharpe")
near("smallest shortfall, SPY volatility target", spy_gap, "4")
near("largest shortfall, AAPL combined book", aapl_gap, "77")

# -------------------------------------------- finding 3: exposure and drawdown

exposure = {"SPY": "78", "AAPL": "48", "BTC-USD": "24"}
for symbol, quoted in exposure.items():
    near(f"{symbol} mean exposure of the combined book",
         strat(symbol, "vol_and_regime", "mean_exposure") * 100, quoted)

drawdown = {
    ("buy_and_hold", "max_drawdown"): ("-18.8", "-33.4", "-53.1"),
    ("vol_and_regime", "max_drawdown"): ("-8.9", "-17.0", "-16.9"),
    ("vol_and_regime", "matched_max_drawdown"): ("-13.5", "-33.0", "-56.1"),
}
for (name, field), values in drawdown.items():
    for symbol, quoted in zip(SYMBOLS, values):
        near(f"{symbol} {name} {field}", strat(symbol, name, field) * 100, quoted)

exact("BTC matched drawdown is worse than holding",
      strat("BTC-USD", "vol_and_regime", "matched_max_drawdown")
      < strat("BTC-USD", "buy_and_hold", "max_drawdown"), True)
near("SPY retains this many points of drawdown advantage at matched risk",
     100 * (strat("SPY", "vol_and_regime", "matched_max_drawdown")
            - strat("SPY", "buy_and_hold", "max_drawdown")), "5.3")
near("AAPL matched drawdown lands this far from the benchmark",
     100 * abs(strat("AAPL", "vol_and_regime", "matched_max_drawdown")
               - strat("AAPL", "buy_and_hold", "max_drawdown")), "0.3")

for symbol in SYMBOLS:
    for name in data[symbol]["strategies"]:
        exact(f"{symbol} {name}: matching leaves Sharpe unchanged",
              strat(symbol, name, "matched_sharpe_equals_sharpe"), True)

# ----------------------------------------------- finding 4: the lookahead's price

lookahead = {
    "vol_and_regime": ("+0.953", "+0.191", "+0.538"),
    "vol_and_regime_LOOKAHEAD": ("+1.852", "+0.603", "+0.600"),
}
for name, values in lookahead.items():
    for symbol, quoted in zip(SYMBOLS, values):
        near(f"{symbol} {name} Sharpe", strat(symbol, name, "sharpe"), quoted)

inflation = {"SPY": "94", "AAPL": "216", "BTC-USD": "12"}
for symbol, quoted in inflation.items():
    near(f"{symbol} lookahead inflation", data[symbol]["lookahead_sharpe_gain_pct"], quoted)

spy_beat = 100 * (strat("SPY", "vol_and_regime_LOOKAHEAD", "sharpe")
                  - strat("SPY", "buy_and_hold", "sharpe")) / strat("SPY", "buy_and_hold", "sharpe")
near("SPY lookahead book beats the benchmark by", spy_beat, "39")
exact("the honest SPY book does not beat the benchmark",
      strat("SPY", "vol_and_regime", "sharpe") < strat("SPY", "buy_and_hold", "sharpe"), True)

# ------------------------------------------------------------ the regime model

regime = {
    "p_stay_calm": ("0.991", "0.930", "0.793"),
    "p_stay_turbulent": ("0.981", "0.767", "0.639"),
    "expected_duration_calm": ("106.0", "14.4", "4.8"),
    "expected_duration_turbulent": ("52.9", "4.3", "2.8"),
    "vol_ratio": ("2.17", "2.45", "2.86"),
}
for field, values in regime.items():
    for symbol, quoted in zip(SYMBOLS, values):
        near(f"{symbol} regime {field}", data[symbol]["regime"][field], quoted)

for symbol in SYMBOLS:
    exact(f"{symbol} regime fit converged", data[symbol]["regime"]["converged"], True)

# ------------------------------------------------------------------ turnover

turnover = {
    "buy_and_hold": ("0.3", "0.3", "0.2"),
    "regime_only": ("11.1", "20.8", "53.4"),
    "vol_and_regime": ("11.5", "15.8", "21.1"),
}
for name, values in turnover.items():
    for symbol, quoted in zip(SYMBOLS, values):
        near(f"{symbol} {name} annual turnover", strat(symbol, name, "annual_turnover"), quoted)

near("BTC regime-only annual cost drag",
     strat("BTC-USD", "regime_only", "cost_drag_annual") * 100, "2.67")
exact("BTC regime-only is the highest-turnover book",
      max(SYMBOLS, key=lambda s: strat(s, "regime_only", "annual_turnover")), "BTC-USD")

# -------------------------------------------------------------------- verdict

if failures:
    print(f"README verification FAILED: {len(failures)} of {checks} figures drifted\n")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)

print(f"all {checks} README figures match the generated reports")
print("\nnot checked, and not checkable: the four numbers in the correction section "
      "(-0.03, -18.9%, -0.29, -33.0%) describe the previous implementation, which is "
      "no longer in this repository.")
