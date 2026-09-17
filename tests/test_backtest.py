"""Regression tests. Each one is named after a defect found in the first version."""

import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from volbt import backtest as B  # noqa: E402
from volbt import metrics as M  # noqa: E402


def returns(n=1000, seed=0, vol=0.01, mu=0.0003):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, vol, n), index=idx)


class TestIdenticalToBenchmark(unittest.TestCase):
    """The decisive test.

    The old engine reported Sharpe -0.03 and a -18.9% drawdown for a strategy
    whose position was constant 1.0, against -0.29 and -33.0% for holding the
    asset. Those must be the same number. They differed because each
    walk-forward window restarted the portfolio at the initial capital, so the
    stitched equity curve gained a fictitious jump every 63 days that truncated
    drawdowns and flattered the Sharpe.
    """

    def test_constant_full_exposure_equals_buy_and_hold_exactly(self):
        r = returns()
        ones = pd.Series(1.0, index=r.index)
        strategy = B.run(r, ones, lag=0)
        benchmark = B.buy_and_hold(r)

        pd.testing.assert_series_equal(strategy.returns, benchmark.returns)
        for key, value in benchmark.metrics().items():
            if isinstance(value, float) and np.isnan(value):
                continue
            self.assertAlmostEqual(strategy.metrics()[key], value, places=12, msg=key)

    def test_metrics_are_computed_on_one_continuous_curve(self):
        """Segment-wise results must concatenate as returns, never as curves."""
        r = returns(600)
        ones = pd.Series(1.0, index=r.index)

        whole = B.run(r, ones, lag=0).returns
        # Split into three contiguous segments and rejoin, as a walk-forward
        # driver does. Joining returns is lossless; joining rebased curves is not.
        pieces = [B.run(r.iloc[a:b], ones.iloc[a:b], lag=0).returns
                  for a, b in ((0, 200), (200, 400), (400, 600))]
        rejoined = pd.concat(pieces)

        pd.testing.assert_series_equal(whole, rejoined)
        self.assertAlmostEqual(M.max_drawdown(whole), M.max_drawdown(rejoined), places=12)
        self.assertEqual(len(rejoined), len(r))


class TestPositionLag(unittest.TestCase):
    def test_a_position_does_not_scale_the_bar_it_was_built_from(self):
        r = returns(50)
        signal = pd.Series(0.0, index=r.index)
        signal.iloc[10] = 1.0
        result = B.run(r, signal, lag=1)
        # The weight set on bar 10 must be applied to bar 11.
        self.assertEqual(result.positions.iloc[10], 0.0)
        self.assertEqual(result.positions.iloc[11], 1.0)

    def test_perfect_foresight_is_obvious_at_lag_zero_and_gone_at_lag_one(self):
        """A lookahead signal should be spectacular unlagged and worthless lagged.

        This is the shape of the bug that smoothed regime probabilities
        introduced: a signal that knows the current bar earns a Sharpe no real
        strategy reaches. If a future change makes a lagged signal look like
        the unlagged one, this test fails.
        """
        r = returns(1000)
        cheat = pd.Series(np.sign(r.values), index=r.index).clip(lower=0.0)

        unlagged = B.run(r, cheat, lag=0).metrics()["sharpe"]
        lagged = B.run(r, cheat, lag=1).metrics()["sharpe"]

        self.assertGreater(unlagged, 8.0, "perfect foresight should be unmistakable")
        self.assertLess(lagged, 2.0, "lagging must destroy the advantage")

    def test_negative_lag_is_rejected(self):
        r = returns(20)
        with self.assertRaises(ValueError):
            B.run(r, pd.Series(1.0, index=r.index), lag=-1)

    def test_mismatched_index_is_rejected_rather_than_silently_aligned(self):
        r = returns(20)
        with self.assertRaises(ValueError):
            B.run(r, pd.Series(1.0, index=pd.RangeIndex(20)))


class TestTargetVolPosition(unittest.TestCase):
    def test_weight_is_inverse_to_forecast_vol(self):
        vol = pd.Series([0.10, 0.20, 0.40], index=pd.bdate_range("2020-01-01", periods=3))
        w = B.target_vol_position(vol, vol_target=0.20, max_leverage=10.0)
        np.testing.assert_allclose(w.values, [2.0, 1.0, 0.5])

    def test_leverage_cap_binds(self):
        vol = pd.Series([0.05], index=pd.bdate_range("2020-01-01", periods=1))
        self.assertEqual(B.target_vol_position(vol, 0.15, max_leverage=1.0).iloc[0], 1.0)
        self.assertAlmostEqual(B.target_vol_position(vol, 0.15, max_leverage=3.0).iloc[0], 3.0)

    def test_zero_forecast_does_not_produce_infinite_leverage(self):
        vol = pd.Series([0.0, 0.2], index=pd.bdate_range("2020-01-01", periods=2))
        w = B.target_vol_position(vol, 0.15, max_leverage=5.0)
        self.assertTrue(np.isnan(w.iloc[0]) or np.isfinite(w.iloc[0]))
        self.assertFalse(np.isinf(w.iloc[0]))


class TestExposureMatching(unittest.TestCase):
    def test_a_capped_overlay_is_under_invested_and_that_flatters_its_drawdown(self):
        """The reason the comparison has to be exposure-matched."""
        r = returns(1000, vol=0.012)
        vol_forecast = pd.Series(r.rolling(20).std().bfill() * np.sqrt(252), index=r.index)
        overlay = B.run(r, B.target_vol_position(vol_forecast, 0.15, max_leverage=1.0))
        bench = B.buy_and_hold(r)

        self.assertLess(overlay.metrics()["mean_exposure"], 1.0)
        self.assertLess(overlay.metrics()["annualised_vol"], bench.metrics()["annualised_vol"])
        # Shallower drawdown while carrying less risk is not yet evidence of skill.
        self.assertGreater(overlay.metrics()["max_drawdown"], bench.metrics()["max_drawdown"])

    def test_matching_equalises_realised_volatility(self):
        r = returns(1000, vol=0.012)
        vol_forecast = pd.Series(r.rolling(20).std().bfill() * np.sqrt(252), index=r.index)
        overlay = B.run(r, B.target_vol_position(vol_forecast, 0.15, max_leverage=1.0))
        bench_vol = M.annualised_vol(B.buy_and_hold(r).returns)

        matched, scale = B.match_exposure(overlay, bench_vol)
        self.assertAlmostEqual(M.annualised_vol(matched), bench_vol, places=10)
        self.assertGreater(scale, 1.0, "an under-invested book has to be levered up to match")

    def test_matching_leaves_sharpe_unchanged(self):
        """Scaling by a constant cannot change the risk-adjusted return."""
        r = returns(800, vol=0.012)
        vol_forecast = pd.Series(r.rolling(20).std().bfill() * np.sqrt(252), index=r.index)
        overlay = B.run(r, B.target_vol_position(vol_forecast, 0.15, max_leverage=1.0))
        matched, _ = B.match_exposure(overlay, 0.20)
        self.assertAlmostEqual(M.sharpe(overlay.returns), M.sharpe(matched), places=10)


class TestCosts(unittest.TestCase):
    def test_costs_reduce_net_returns_and_are_attributed(self):
        r = returns(500)
        vol_forecast = pd.Series(r.rolling(20).std().bfill() * np.sqrt(252), index=r.index)
        weights = B.target_vol_position(vol_forecast, 0.15, max_leverage=1.0)

        free = B.run(r, weights, cost_per_unit_turnover=0.0)
        costly = B.run(r, weights, cost_per_unit_turnover=0.0010)

        self.assertLess(costly.returns.sum(), free.returns.sum())
        self.assertAlmostEqual(free.metrics()["cost_drag_annual"], 0.0, places=12)
        self.assertGreater(costly.metrics()["cost_drag_annual"], 0.0)

    def test_buy_and_hold_never_trades_after_entry(self):
        r = returns(100)
        bench = B.buy_and_hold(r)
        self.assertAlmostEqual(bench.turnover.iloc[1:].sum(), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
