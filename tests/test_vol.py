import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from volbt import vol as V  # noqa: E402


def returns(n=800, seed=0, vol=0.01):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(0.0, vol, n), index=idx)


BASELINES = [V.ConstantVol(), V.RollingWindowVol(20), V.RollingWindowVol(60), V.EwmaVol(0.94)]


class TestNoLookahead(unittest.TestCase):
    """The property the old pipeline could not have satisfied."""

    def test_changing_a_future_return_cannot_change_an_earlier_forecast(self):
        r = returns()
        start = 300
        for model in BASELINES:
            base = model.forecast_span(r, start=start)
            tampered = r.copy()
            tampered.iloc[600] = 0.25          # a 25% day, far in the future
            after = model.forecast_span(tampered, start=start)
            pd.testing.assert_series_equal(
                base.iloc[start:600], after.iloc[start:600],
                obj=f"{model.name} forecasts before the tampered bar",
            )

    def test_the_tamper_is_actually_detectable_later(self):
        """Guards the test above: if nothing ever changes, it proves nothing."""
        r = returns()
        model = V.RollingWindowVol(20)
        tampered = r.copy()
        tampered.iloc[600] = 0.25
        base, after = model.forecast_span(r, 300), model.forecast_span(tampered, 300)
        self.assertGreater(after.iloc[610], base.iloc[610] * 1.5)

    def test_a_forecast_for_bar_i_excludes_bar_i(self):
        r = returns(400)
        model = V.RollingWindowVol(20)
        out = model.forecast_span(r, start=100)
        expected = float(np.std(r.to_numpy()[80:100], ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(out.iloc[100], expected, places=12)

    def test_no_forecast_is_produced_before_the_start_index(self):
        out = V.EwmaVol().forecast_span(returns(300), start=200)
        self.assertTrue(out.iloc[:200].isna().all())
        self.assertTrue(out.iloc[200:].notna().all())


class TestScoring(unittest.TestCase):
    def test_an_empty_overlap_raises_instead_of_reporting_a_score(self):
        """The old code's silent failure, now loud."""
        idx = pd.bdate_range("2020-01-01", periods=100)
        realised = pd.Series(np.full(100, 0.2), index=idx)
        forecast = pd.Series(np.nan, index=idx)          # nothing overlaps
        with self.assertRaises(ValueError) as ctx:
            V.score_forecasts(realised, forecast)
        self.assertIn("empty slice", str(ctx.exception))

    def test_a_perfect_forecast_scores_zero_error(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        realised = pd.Series(np.linspace(0.1, 0.3, 100), index=idx)
        s = V.score_forecasts(realised, realised.copy())
        self.assertAlmostEqual(s["rmse"], 0.0, places=12)
        self.assertAlmostEqual(s["bias"], 0.0, places=12)
        self.assertAlmostEqual(s["corr"], 1.0, places=10)

    def test_qlike_prefers_the_true_variance_over_a_biased_one(self):
        rng = np.random.default_rng(5)
        true_var = np.full(4000, 0.0004)
        realised_var = true_var * rng.chisquare(4, 4000) / 4      # noisy proxy
        self.assertLess(V.qlike(realised_var, true_var), V.qlike(realised_var, true_var * 2))
        self.assertLess(V.qlike(realised_var, true_var), V.qlike(realised_var, true_var * 0.5))

    def test_bias_has_a_sign(self):
        idx = pd.bdate_range("2020-01-01", periods=50)
        realised = pd.Series(np.full(50, 0.20), index=idx)
        self.assertGreater(V.score_forecasts(realised, realised + 0.05)["bias"], 0)
        self.assertLess(V.score_forecasts(realised, realised - 0.05)["bias"], 0)


class TestGarchRecursions(unittest.TestCase):
    """The recursions are tested here so the backtest does not depend on an
    untested library path for the step that turns parameters into forecasts."""

    def test_garch_filter_matches_a_hand_computed_path(self):
        omega, alpha, beta = 0.02, 0.10, 0.85
        resid = np.array([1.0, -2.0, 0.5])
        s = V.garch_filter(resid, omega, alpha, beta)
        s0 = omega / (1 - alpha - beta)
        self.assertAlmostEqual(s[0], s0, places=12)
        self.assertAlmostEqual(s[1], omega + alpha * 1.0 + beta * s0, places=12)
        self.assertAlmostEqual(s[2], omega + alpha * 4.0 + beta * s[1], places=12)
        self.assertAlmostEqual(s[3], omega + alpha * 0.25 + beta * s[2], places=12)

    def test_the_filter_returns_one_extra_slot_for_the_forecast(self):
        self.assertEqual(len(V.garch_filter(np.zeros(10), 0.02, 0.1, 0.85)), 11)
        self.assertEqual(len(V.egarch_filter(np.zeros(10), 0.0, 0.1, 0.9, 0.0)), 11)

    def test_gjr_amplifies_negative_shocks_only(self):
        omega, alpha, beta, gamma = 0.02, 0.05, 0.85, 0.10
        up = V.garch_filter(np.array([2.0]), omega, alpha, beta, gamma)
        down = V.garch_filter(np.array([-2.0]), omega, alpha, beta, gamma)
        self.assertGreater(down[-1], up[-1])
        # A symmetric GARCH must not care about the sign.
        sym_up = V.garch_filter(np.array([2.0]), omega, alpha, beta, 0.0)
        sym_down = V.garch_filter(np.array([-2.0]), omega, alpha, beta, 0.0)
        self.assertAlmostEqual(sym_up[-1], sym_down[-1], places=12)

    def test_egarch_gamma_creates_the_leverage_asymmetry(self):
        omega, alpha, beta = 0.0, 0.10, 0.90
        up = V.egarch_filter(np.array([2.0]), omega, alpha, beta, gamma=-0.10)
        down = V.egarch_filter(np.array([-2.0]), omega, alpha, beta, gamma=-0.10)
        self.assertGreater(down[-1], up[-1])

    def test_a_stationary_garch_filter_stays_bounded(self):
        rng = np.random.default_rng(7)
        s = V.garch_filter(rng.normal(0, 1, 5000), 0.02, 0.08, 0.90)
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertLess(s.max(), 100.0)

    def test_the_filter_recovers_the_variance_of_data_it_generated(self):
        """Simulate a GARCH path, filter it back, and the level should agree."""
        omega, alpha, beta = 0.02, 0.08, 0.90
        rng = np.random.default_rng(11)
        n = 20000
        sigma2 = np.empty(n)
        e = np.empty(n)
        sigma2[0] = omega / (1 - alpha - beta)
        for t in range(n):
            e[t] = np.sqrt(sigma2[t]) * rng.normal()
            if t + 1 < n:
                sigma2[t + 1] = omega + alpha * e[t] ** 2 + beta * sigma2[t]
        filtered = V.garch_filter(e, omega, alpha, beta)[:n]
        np.testing.assert_allclose(filtered, sigma2, rtol=1e-10)


class TestFactories(unittest.TestCase):
    def test_baselines_are_available_without_arch(self):
        names = [m.name for m in V.default_forecasters(include_garch=False)]
        self.assertEqual(names, ["constant", "rolling_20d", "rolling_60d", "ewma_0.94"])

    def test_requesting_garch_adds_three_arms(self):
        names = [m.name for m in V.default_forecasters(include_garch=True)]
        self.assertEqual(names[-3:], ["garch", "egarch", "gjr"])

    def test_bad_configuration_is_rejected_early(self):
        with self.assertRaises(ValueError):
            V.RollingWindowVol(1)
        with self.assertRaises(ValueError):
            V.EwmaVol(1.5)
        with self.assertRaises(ValueError):
            V.GarchVol("arima")
        with self.assertRaises(ValueError):
            V.ConstantVol().forecast_span(returns(100), start=500)


if __name__ == "__main__":
    unittest.main()


class TestScoringTargets(unittest.TestCase):
    """The proxy choice decided the winner, so the proxies get their own tests."""

    def test_the_one_bar_proxy_uses_only_its_own_bar(self):
        r = returns(200)
        proxy = V.squared_return_proxy(r)
        tampered = r.copy()
        tampered.iloc[100] = 0.30
        after = V.squared_return_proxy(tampered)
        # Only bar 100 moves. Nothing else can, because nothing else is used.
        self.assertGreater(abs(after.iloc[100] - proxy.iloc[100]), 1.0)
        pd.testing.assert_series_equal(proxy.drop(proxy.index[100]),
                                       after.drop(after.index[100]))

    def test_the_backward_proxy_overlaps_the_window_a_rolling_forecast_used(self):
        """Why the first ranking was rigged.

        A rolling-20d forecast for bar t is built from bars t-20..t-1. The
        backward 20-day realised measure at bar t covers t-19..t. Nineteen of
        twenty observations are shared, so one tampered return moves both, and
        the resulting correlation measures overlap rather than skill.
        """
        r = returns(200)
        tampered = r.copy()
        tampered.iloc[100] = 0.30

        backward = V.realised_vol(r, 20)
        backward_after = V.realised_vol(tampered, 20)
        moved_backward = (backward_after - backward).abs().iloc[101:120].max()

        forecast = V.RollingWindowVol(20).forecast_span(r, start=50)
        forecast_after = V.RollingWindowVol(20).forecast_span(tampered, start=50)
        moved_forecast = (forecast_after - forecast).abs().iloc[101:120].max()

        self.assertGreater(moved_backward, 0.05, "the backward proxy moves")
        self.assertGreater(moved_forecast, 0.05, "and so does the forecast, on the same bars")

        one_bar = V.squared_return_proxy(r)
        one_bar_after = V.squared_return_proxy(tampered)
        self.assertAlmostEqual(
            (one_bar_after - one_bar).abs().iloc[101:120].max(), 0.0, places=12,
            msg="the one-bar proxy must not move on any bar but its own",
        )

    def test_the_forward_proxy_uses_only_bars_at_or_after_t(self):
        r = returns(200)
        forward = V.forward_realised_vol(r, horizon=21)
        tampered = r.copy()
        tampered.iloc[100] = 0.30
        after = V.forward_realised_vol(tampered, horizon=21)
        # Bars 80..100 look forward across bar 100, so they move.
        self.assertGreater((after - forward).abs().iloc[80:101].max(), 0.05)
        # Bars after 100 do not look back, so they cannot.
        self.assertAlmostEqual((after - forward).abs().iloc[101:].max(), 0.0, places=12)

    def test_the_forward_proxy_covers_the_stated_horizon(self):
        r = returns(100)
        forward = V.forward_realised_vol(r, horizon=10)
        expected = float(r.iloc[20:30].std(ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(forward.iloc[20], expected, places=12)

    def test_qlike_on_the_one_bar_proxy_still_ranks_the_true_variance_first(self):
        """The proxy is noisy; QLIKE is the loss that tolerates that."""
        rng = np.random.default_rng(9)
        true_var = np.full(20000, 0.0004)
        r = rng.normal(0.0, np.sqrt(true_var))
        proxy_var = r**2
        self.assertLess(V.qlike(proxy_var, true_var), V.qlike(proxy_var, true_var * 1.5))
        self.assertLess(V.qlike(proxy_var, true_var), V.qlike(proxy_var, true_var * 0.66))
