import unittest
import numpy as np

from aleatory.processes import ContinuousTimeRandomWalk, ReflectingContinuousTimeRandomWalk


def check_state_space(paths, lower=None, upper=None):
    all_states = np.concatenate([states for _, states in paths])
    if lower is not None:
        assert np.all(all_states >= lower), "State fell below lower boundary"
    if upper is not None:
        assert np.all(all_states <= upper), "State exceeded upper boundary"
    return all_states.min(), all_states.max()


def check_jump_sizes(paths):
    for _, states in paths:
        diffs = np.diff(states)
        diffs = diffs[diffs != 0]
        assert np.all(np.isin(diffs, [-1, 1])), f"Bad jump sizes found: {np.unique(diffs)}"


def check_path_structure(paths, T):
    for times, states in paths:
        assert len(times) == len(states), "times/states length mismatch"
        assert times[0] == 0.0, "Path does not start at t=0"
        assert np.isclose(times[-1], T), "Path does not end at T"
        assert np.all(np.diff(times) >= 0), "Times are not nondecreasing"


def check_reflection_rule(paths, lower=None, upper=None):
    for _, states in paths:
        diffs = np.diff(states)
        for s, d in zip(states[:-1], diffs):
            if d == 0:
                continue
            if lower is not None and s == lower:
                assert d == 1, f"Illegal downward jump from lower boundary {lower}"
            if upper is not None and s == upper:
                assert d == -1, f"Illegal upward jump from upper boundary {upper}"


class TestReflectingContinuousTimeRandomWalkValidation(unittest.TestCase):
    def test_requires_at_least_one_boundary(self):
        with self.assertRaises(ValueError):
            ReflectingContinuousTimeRandomWalk(
                rate_up=0.5, rate_down=0.5, initial=0, lower=None, upper=None
            )

    def test_requires_lower_less_than_upper(self):
        bad_cases = [
            dict(lower=3, upper=3, initial=3),
            dict(lower=4, upper=3, initial=3),
        ]
        for kw in bad_cases:
            with self.subTest(case=kw):
                with self.assertRaises(ValueError):
                    ReflectingContinuousTimeRandomWalk(
                        rate_up=0.5, rate_down=0.5, **kw
                    )

    def test_initial_must_be_inside_bounds(self):
        bad_cases = [
            dict(lower=0, upper=3, initial=-1),
            dict(lower=-3, upper=0, initial=1),
        ]
        for kw in bad_cases:
            with self.subTest(case=kw):
                with self.assertRaises(ValueError):
                    ReflectingContinuousTimeRandomWalk(
                        rate_up=0.5, rate_down=0.5, **kw
                    )


class TestReflectingContinuousTimeRandomWalkInvariants(unittest.TestCase):
    def setUp(self):
        self.seed = 42

    def test_two_sided_reflecting_symmetric_invariants(self):
        lower = -5
        upper = 5
        T = 50.0

        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=0.5,
            rate_down=0.5,
            initial=0,
            lower=lower,
            upper=upper,
            rng=rng,
        )
        paths = p.simulate(N=2000, T=T)

        check_state_space(paths, lower=lower, upper=upper)
        check_jump_sizes(paths)
        check_path_structure(paths, T=T)
        check_reflection_rule(paths, lower=lower, upper=upper)

    def test_two_sided_reflecting_biased_invariants(self):
        lower = -5
        upper = 5
        T = 50.0

        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=0.8,
            rate_down=0.2,
            initial=0,
            lower=lower,
            upper=upper,
            rng=rng,
        )
        paths = p.simulate(N=2000, T=T)

        check_state_space(paths, lower=lower, upper=upper)
        check_jump_sizes(paths)
        check_path_structure(paths, T=T)
        check_reflection_rule(paths, lower=lower, upper=upper)

    def test_lower_reflecting_only_invariants(self):
        lower = 0
        upper = None
        T = 50.0

        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=0.5,
            rate_down=0.5,
            initial=0,
            lower=lower,
            upper=upper,
            rng=rng,
        )
        paths = p.simulate(N=2000, T=T)

        check_state_space(paths, lower=lower, upper=upper)
        check_jump_sizes(paths)
        check_path_structure(paths, T=T)
        check_reflection_rule(paths, lower=lower, upper=upper)

    def test_upper_reflecting_only_invariants(self):
        lower = None
        upper = 0
        T = 50.0

        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=0.5,
            rate_down=0.5,
            initial=0,
            lower=lower,
            upper=upper,
            rng=rng,
        )
        paths = p.simulate(N=2000, T=T)

        check_state_space(paths, lower=lower, upper=upper)
        check_jump_sizes(paths)
        check_path_structure(paths, T=T)
        check_reflection_rule(paths, lower=lower, upper=upper)


class TestReflectingContinuousTimeRandomWalkBehaviour(unittest.TestCase):
    def setUp(self):
        self.seed = 42

    def test_wide_interval_agrees_with_unbounded_walk(self):
        T = 2.0
        N = 20000

        rng1 = np.random.default_rng(self.seed)
        p_unbounded = ContinuousTimeRandomWalk(
            rate_up=0.5, rate_down=0.5, initial=0, rng=rng1
        )
        paths_u = p_unbounded.simulate(N=N, T=T)
        x_u = np.array([states[-1] for _, states in paths_u])

        rng2 = np.random.default_rng(self.seed)
        p_bounded = ReflectingContinuousTimeRandomWalk(
            rate_up=0.5,
            rate_down=0.5,
            initial=0,
            lower=-50,
            upper=50,
            rng=rng2,
        )
        paths_b = p_bounded.simulate(N=N, T=T)
        x_b = np.array([states[-1] for _, states in paths_b])

        self.assertAlmostEqual(x_u.mean(), x_b.mean(), places=1)
        self.assertAlmostEqual(x_u.var(), x_b.var(), places=1)

    def test_symmetric_two_sided_terminal_distribution_is_approximately_symmetric(self):
        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=0.5, rate_down=0.5, initial=0, lower=-4, upper=4, rng=rng
        )
        paths = p.simulate(N=50000, T=10.0)
        xT = np.array([states[-1] for _, states in paths])

        vals, counts = np.unique(xT, return_counts=True)
        freq = dict(zip(vals, counts / counts.sum()))

        for k in range(1, 5):
            with self.subTest(k=k):
                self.assertAlmostEqual(
                    freq.get(k, 0.0),
                    freq.get(-k, 0.0),
                    places=2,
                )

    def test_symmetric_two_sided_long_run_is_approximately_uniform(self):
        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=0.5, rate_down=0.5, initial=0, lower=-3, upper=3, rng=rng
        )
        paths = p.simulate(N=50000, T=100.0)
        xT = np.array([states[-1] for _, states in paths])

        vals, counts = np.unique(xT, return_counts=True)
        freq = dict(zip(vals, counts / counts.sum()))

        expected = 1.0 / 7.0
        for k in range(-3, 4):
            with self.subTest(state=k):
                self.assertAlmostEqual(freq.get(k, 0.0), expected, places=2)

    def test_lower_boundary_can_trap_if_upward_rate_is_zero(self):
        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=0.0, rate_down=1.0, initial=0, lower=0, upper=None, rng=rng
        )
        _, states = p.sample(T=20.0)
        self.assertTrue(np.all(states == 0))

    def test_upper_boundary_can_trap_if_downward_rate_is_zero(self):
        rng = np.random.default_rng(self.seed)
        p = ReflectingContinuousTimeRandomWalk(
            rate_up=1.0, rate_down=0.0, initial=0, lower=None, upper=0, rng=rng
        )
        _, states = p.sample(T=20.0)
        self.assertTrue(np.all(states == 0))


if __name__ == "__main__":
    unittest.main()