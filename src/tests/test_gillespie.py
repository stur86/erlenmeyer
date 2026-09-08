import numpy as np
import pytest

from erlenmeyer.gillespie import GillespieSimulator, _stochastic_propensity
from erlenmeyer.reaction import Reaction, ReactionSystem
from erlenmeyer.simulator import SimulationTrajectory
from erlenmeyer.symbols import Species


def _decay_system(k=1.0):
    """A simple first-order decay A -> B with rate k."""
    a, b = Species("A"), Species("B")
    system = ReactionSystem([a, b])
    system.add_reaction(Reaction(a, b, k))
    return system


class TestStochasticPropensity:
    def test_monomolecular(self):
        # A -> B, k = 2, population A = 5: propensity is k * 5 = 10
        rates = _stochastic_propensity(
            np.array([5, 0]),
            np.array([[1.0, 0.0]]),
            np.array([2.0]),
        )
        assert rates.tolist() == [10.0]

    def test_bimolecular_is_a_falling_factorial(self):
        # 2A -> B, k = 2, population A = 5: propensity is k * 5 * 4 = 40
        rates = _stochastic_propensity(
            np.array([5, 0]),
            np.array([[2.0, 0.0]]),
            np.array([2.0]),
        )
        assert rates.tolist() == [40.0]

    def test_bimolecular_of_one_is_zero(self):
        # 2A -> B with only one A available: no reaction possible
        rates = _stochastic_propensity(
            np.array([1, 0]),
            np.array([[2.0, 0.0]]),
            np.array([2.0]),
        )
        assert rates.tolist() == [0.0]

    def test_rate_constant_scales_linear_propensity(self):
        fast = _stochastic_propensity(
            np.array([5, 0]), np.array([[1.0, 0.0]]), np.array([3.0])
        )
        slow = _stochastic_propensity(
            np.array([5, 0]), np.array([[1.0, 0.0]]), np.array([1.0])
        )
        assert fast.tolist() == [15.0]
        assert slow.tolist() == [5.0]


class TestStochasticPropensityShape:
    def test_one_propensity_per_reaction(self):
        rates = _stochastic_propensity(
            np.array([5, 0]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([2.0, 3.0]),
        )
        assert rates.shape == (2,)


class TestGillespieSimulatorTrajectory:
    def test_returns_a_simulation_trajectory(self):
        result = GillespieSimulator(_decay_system()).run({"A": 50}, t_end=1.0)
        assert isinstance(result, SimulationTrajectory)

    def test_species_names_are_strings(self):
        result = GillespieSimulator(_decay_system()).run({"A": 50}, t_end=1.0)
        assert result.species == ["A", "B"]
        assert all(isinstance(s, str) for s in result.species)

    def test_starts_at_initial_populations(self):
        result = GillespieSimulator(_decay_system()).run({"A": 50}, t_end=1.0)
        assert result.values[0].tolist() == [50, 0]

    def test_populations_stay_non_negative(self):
        result = GillespieSimulator(_decay_system()).run({"A": 50}, t_end=5.0)
        assert np.all(result.values >= 0)

    def test_populations_are_integers(self):
        result = GillespieSimulator(_decay_system()).run({"A": 50}, t_end=5.0)
        assert result.values.dtype == np.int64

    def test_accepts_integer_valued_float_initial(self):
        result = GillespieSimulator(_decay_system()).run({"A": 50.0}, t_end=1.0)
        assert result.values[0].tolist() == [50, 0]

    def test_non_integer_initial_raises(self):
        with pytest.raises(ValueError):
            GillespieSimulator(_decay_system()).run({"A": 50.5}, t_end=1.0)


class TestGillespieSimulatorPhysics:
    def test_decay_conserves_total_population(self):
        # A -> B: the total A + B is constant throughout
        result = GillespieSimulator(_decay_system()).run({"A": 50}, t_end=5.0)
        totals = result.values.sum(axis=1)
        assert np.all(totals == 50)

    def test_decay_drives_a_to_zero(self):
        result = GillespieSimulator(_decay_system(k=2.0)).run(
            {"A": 50}, t_end=20.0
        )
        assert result.values[-1][0] == 0.0
        assert result.values[-1][1] == 50.0

    def test_equilibrated_system_stops(self):
        # No reactions possible once A is gone; time jumps to t_end
        result = GillespieSimulator(_decay_system()).run({"B": 50}, t_end=3.0)
        assert result.times[-1] == 3.0
        assert result.values[-1].tolist() == [0.0, 50.0]


class TestGillespieSimulatorSeeding:
    def test_same_seed_is_reproducible(self):
        sys = _decay_system()
        r1 = GillespieSimulator(sys).run({"A": 50}, t_end=2.0, seed=7)
        r2 = GillespieSimulator(sys).run({"A": 50}, t_end=2.0, seed=7)
        assert np.array_equal(r1.values, r2.values)

    def test_different_seeds_differ(self):
        sys = _decay_system()
        r1 = GillespieSimulator(sys).run({"A": 50}, t_end=2.0, seed=7)
        r2 = GillespieSimulator(sys).run({"A": 50}, t_end=2.0, seed=8)
        assert not np.array_equal(r1.values, r2.values)

    def test_accepts_no_seed(self):
        result = GillespieSimulator(_decay_system()).run({"A": 50}, t_end=1.0)
        assert isinstance(result, SimulationTrajectory)
