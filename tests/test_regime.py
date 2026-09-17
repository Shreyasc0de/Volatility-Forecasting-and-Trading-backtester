import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from volbt import regime as R  # noqa: E402


def two_state_series(n=1500, seed=0, p_stay=(0.97, 0.93), vols=(0.007, 0.028)):
    """Simulate a genuine 2-state path so the fit has a truth to recover."""
    rng = np.random.default_rng(seed)
    state = 0
    states, values = [], []
    for _ in range(n):
        states.append(state)
        values.append(rng.normal(0.0, vols[state]))
        if rng.random() > p_stay[state]:
            state = 1 - state
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(values, index=idx), pd.Series(states, index=idx)


class TestFit(unittest.TestCase):
    def test_it_recovers_the_two_volatility_levels(self):
        r, _ = two_state_series()
        m = R.fit(r)
        recovered = np.sqrt(m.variances)
        self.assertAlmostEqual(recovered[0], 0.007, delta=0.003)
        self.assertAlmostEqual(recovered[1], 0.028, delta=0.008)

    def test_state_one_is_always_the_turbulent_one(self):
        """An EM fit can land on either labelling; a trading rule cannot."""
        for seed in range(6):
            m = R.fit(two_state_series(seed=seed)[0])
            self.assertGreater(m.variances[1], m.variances[0], f"seed {seed}")
            self.assertEqual(m.turbulent, 1)

    def test_it_recovers_persistent_transitions(self):
        r, _ = two_state_series()
        p = R.fit(r).persistence()
        self.assertGreater(p["p_stay_calm"], 0.85)
        self.assertGreater(p["p_stay_turbulent"], 0.80)
        self.assertGreater(p["vol_ratio"], 2.0)

    def test_the_likelihood_increases_and_the_fit_converges(self):
        m = R.fit(two_state_series()[0])
        self.assertTrue(np.isfinite(m.log_likelihood))
        self.assertTrue(m.converged, f"did not converge in {m.n_iter} iterations")

    def test_the_fit_is_deterministic(self):
        r, _ = two_state_series()
        a, b = R.fit(r), R.fit(r)
        np.testing.assert_allclose(a.variances, b.variances)
        np.testing.assert_allclose(a.transition, b.transition)

    def test_too_short_a_series_is_rejected(self):
        with self.assertRaises(ValueError):
            R.fit(two_state_series(n=30)[0])


class TestFilteredVersusSmoothed(unittest.TestCase):
    """The bug this module exists to prevent."""

    def test_a_later_observation_cannot_move_a_filtered_probability(self):
        r, _ = two_state_series()
        model = R.fit(r)

        tampered = r.copy()
        tampered.iloc[1000] = 0.20              # a 20% day, far in the future

        base = R.filtered_probabilities(model, r)["turbulent"]
        after = R.filtered_probabilities(model, tampered)["turbulent"]
        pd.testing.assert_series_equal(base.iloc[:1000], after.iloc[:1000])

    def test_a_later_observation_does_move_a_smoothed_probability(self):
        """Which is precisely why smoothed probabilities cannot be traded."""
        r, _ = two_state_series()
        model = R.fit(r)

        tampered = r.copy()
        tampered.iloc[1000] = 0.20

        base = R.smoothed_probabilities(model, r)["turbulent"]
        after = R.smoothed_probabilities(model, tampered)["turbulent"]
        moved = (after.iloc[:1000] - base.iloc[:1000]).abs().max()
        self.assertGreater(moved, 1e-6,
                           "if this does not move, the smoother is not using the future")

    def test_smoothed_beats_filtered_at_identifying_the_true_state(self):
        """Smoothing is better at describing history. That is its whole problem."""
        r, states = two_state_series()
        model = R.fit(r)
        filt = R.filtered_probabilities(model, r)["turbulent"]
        smooth = R.smoothed_probabilities(model, r)["turbulent"]
        filt_acc = ((filt > 0.5).astype(int) == states).mean()
        smooth_acc = ((smooth > 0.5).astype(int) == states).mean()
        self.assertGreater(smooth_acc, filt_acc)

    def test_both_are_probabilities(self):
        r, _ = two_state_series()
        model = R.fit(r)
        for frame in (R.filtered_probabilities(model, r), R.smoothed_probabilities(model, r)):
            np.testing.assert_allclose(frame.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
            self.assertTrue((frame >= -1e-12).all().all())
            self.assertTrue((frame <= 1 + 1e-12).all().all())


class TestWalkForward(unittest.TestCase):
    def test_a_future_return_cannot_change_any_earlier_probability(self):
        """The strictest version: parameters and filter both stop at bar i."""
        r, _ = two_state_series(n=700)
        tampered = r.copy()
        tampered.iloc[600] = 0.20

        base = R.walk_forward_filtered(r, start=400, refit_every=100)
        after = R.walk_forward_filtered(tampered, start=400, refit_every=100)
        pd.testing.assert_frame_equal(base.iloc[400:600], after.iloc[400:600])

    def test_nothing_is_produced_before_the_start_index(self):
        r, _ = two_state_series(n=600)
        out = R.walk_forward_filtered(r, start=400, refit_every=100)
        self.assertTrue(out.iloc[:400].isna().all().all())
        self.assertTrue(out.iloc[400:].notna().all().all())


class TestExposure(unittest.TestCase):
    def test_full_exposure_when_calm_and_floor_when_turbulent(self):
        p = pd.Series([0.0, 0.5, 1.0], index=pd.bdate_range("2020-01-01", periods=3))
        np.testing.assert_allclose(R.exposure_from_regime(p, floor=0.0).to_numpy(), [1.0, 0.5, 0.0])
        np.testing.assert_allclose(R.exposure_from_regime(p, floor=0.25).to_numpy(), [1.0, 0.625, 0.25])

    def test_an_invalid_floor_is_rejected(self):
        p = pd.Series([0.5], index=pd.bdate_range("2020-01-01", periods=1))
        with self.assertRaises(ValueError):
            R.exposure_from_regime(p, floor=1.5)


if __name__ == "__main__":
    unittest.main()
