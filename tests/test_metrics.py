import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from volbt import metrics as M  # noqa: E402


def series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


class TestReturnConventions(unittest.TestCase):
    def test_log_and_simple_round_trip(self):
        simple = series([0.01, -0.02, 0.005, 0.03])
        pd.testing.assert_series_equal(M.from_log(M.to_log(simple)), simple)

    def test_compounding_log_returns_as_simple_overstates_growth(self):
        """The old engine's error, quantified.

        It produced log returns upstream and then ran (1 + r).cumprod() on them.
        For a daily equity series the gap is small; at crypto volatility it is
        not, which is exactly where the old README quoted BTC results.
        """
        rng = np.random.default_rng(1)
        log_r = series(rng.normal(0.0008, 0.65 / np.sqrt(252), 1260))  # ~BTC vol
        correct = float((1 + M.from_log(log_r)).prod() - 1)
        wrong = float((1 + log_r).prod() - 1)
        self.assertGreater(abs(wrong - correct), 0.05,
                           "the convention error should be material at this volatility")

    def test_annualised_return_is_geometric_not_a_compounded_mean(self):
        """A flat series must annualise to zero under either horizon."""
        flat = series([0.01, -1 / 101] * 200)  # up 1%, exactly back down
        self.assertAlmostEqual(float((1 + flat).prod()), 1.0, places=10)
        self.assertAlmostEqual(M.annualised_return(flat), 0.0, places=10)

    def test_a_known_doubling_over_one_year_annualises_to_100_percent(self):
        daily = 2 ** (1 / 252) - 1
        r = series([daily] * 252)
        self.assertAlmostEqual(M.annualised_return(r), 1.0, places=8)


class TestDrawdown(unittest.TestCase):
    def test_drawdown_on_a_hand_checked_path(self):
        # 1.0 -> 1.10 -> 0.88 -> 0.968. Trough 0.88 against a peak of 1.10.
        r = series([0.10, -0.20, 0.10])
        self.assertAlmostEqual(M.max_drawdown(r), 0.88 / 1.10 - 1.0, places=12)

    def test_a_monotone_series_has_no_drawdown(self):
        self.assertAlmostEqual(M.max_drawdown(series([0.01] * 50)), 0.0, places=12)

    def test_the_equity_curve_never_resets(self):
        r = series([0.01] * 300)
        curve = M.equity_curve(r)
        self.assertTrue((curve.diff().dropna() > 0).all())
        self.assertAlmostEqual(curve.iloc[-1], 1.01 ** 300, places=8)


class TestSharpe(unittest.TestCase):
    def test_the_risk_free_rate_changes_the_answer(self):
        rng = np.random.default_rng(2)
        r = series(rng.normal(0.0004, 0.01, 1260))
        self.assertGreater(M.sharpe(r, 0.0), M.sharpe(r, 0.05))

    def test_a_constant_return_series_has_undefined_sharpe_not_infinity(self):
        self.assertTrue(np.isnan(M.sharpe(series([0.001] * 100))))

    def test_sortino_ignores_upside_volatility(self):
        """Same downside distribution, bigger upside, so Sortino must rise."""
        mild = series([0.010, -0.010, 0.012, -0.020] * 50)
        wild = series([0.050, -0.010, 0.060, -0.020] * 50)
        # The downside legs are identical by construction, which is the point.
        self.assertAlmostEqual(mild[mild < 0].std(ddof=1), wild[wild < 0].std(ddof=1), places=15)
        self.assertGreater(M.sortino(wild), M.sortino(mild))

    def test_a_series_whose_losses_are_all_identical_has_undefined_sortino(self):
        """Zero downside dispersion is undefined, not infinite."""
        self.assertTrue(np.isnan(M.sortino(series([0.01, -0.01] * 100))))


class TestSummary(unittest.TestCase):
    def test_calmar_is_return_over_drawdown_and_nan_without_one(self):
        r = series([0.10, -0.20, 0.10])
        s = M.summarise(r)
        self.assertAlmostEqual(s["calmar"], s["annualised_return"] / abs(s["max_drawdown"]), places=10)
        self.assertTrue(np.isnan(M.summarise(series([0.001] * 50))["calmar"]))

    def test_the_risk_free_rate_used_is_recorded(self):
        self.assertEqual(M.summarise(series([0.001, -0.002] * 50), 0.045)["risk_free_annual"], 0.045)


if __name__ == "__main__":
    unittest.main()
