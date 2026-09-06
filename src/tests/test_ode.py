import numpy as np
import pytest

from erlenmeyer.ode import ODESimulator, _ode_kernel
from erlenmeyer.reaction import Reaction, ReactionSystem
from erlenmeyer.simulator import SimulationTrajectory
from erlenmeyer.symbols import Species


def _decay_system(k=1.0):
    """A simple first-order decay A -> B with rate k."""
    a, b = Species("A"), Species("B")
    system = ReactionSystem([a, b])
    system.add_reaction(Reaction(a, b, k))
    return system


class TestOdeKernel:
    def test_first_order_decay(self):
        # A -> B, k = 2, current state A=3, B=1
        y = np.array([3.0, 1.0])
        t = 0.0
        r_m = np.array([[1.0, 0.0]])
        p_m = np.array([[0.0, 1.0]])
        r_v = np.array([2.0])
        dy = _ode_kernel(t, y, r_m, p_m, r_v)
        assert np.allclose(dy, [-6.0, 6.0])

    def test_return_shape_matches_species(self):
        y = np.array([3.0, 1.0, 4.0])
        r_m = np.zeros((1, 3))
        p_m = np.zeros((1, 3))
        r_v = np.zeros(1)
        dy = _ode_kernel(0.0, y, r_m, p_m, r_v)
        assert dy.shape == (3,)


class TestOdeSimulatorMatchesAnalytic:
    def test_first_order_decay(self):
        k = 2.0
        t_end = 3.0
        sim = ODESimulator(_decay_system(k))
        result = sim.run(np.array([5.0, 0.0]), t_end=t_end, rtol=1e-10, atol=1e-12)
        expected_a = 5.0 * np.exp(-k * t_end)
        assert np.isclose(result.values[-1, 0], expected_a)
        assert np.isclose(result.values[-1, 1], 5.0 - expected_a)

    def test_rate_constant_is_respected(self):
        t_end = 2.0
        fast = ODESimulator(_decay_system(k=3.0)).run(
            np.array([1.0, 0.0]), t_end=t_end
        )
        slow = ODESimulator(_decay_system(k=1.0)).run(
            np.array([1.0, 0.0]), t_end=t_end
        )
        # The slower decay leaves more A behind
        assert fast.values[-1, 0] < slow.values[-1, 0]


class TestOdeSimulatorTrajectory:
    def test_species_names_are_strings(self):
        sim = ODESimulator(_decay_system())
        result = sim.run(np.array([1.0, 0.0]))
        assert result.species == ["A", "B"]
        assert all(isinstance(s, str) for s in result.species)

    def test_values_shape_is_times_by_species(self):
        sim = ODESimulator(_decay_system())
        result = sim.run(np.array([1.0, 0.0]), steps=50)
        assert result.values.shape == (50, 2)

    def test_returns_a_simulation_trajectory(self):
        sim = ODESimulator(_decay_system())
        result = sim.run(np.array([1.0, 0.0]))
        assert isinstance(result, SimulationTrajectory)


class TestOdeSimulatorTimeAxes:
    @pytest.mark.parametrize("t_start,t_end", [(0.0, 5.0), (1.0, 4.0), (-2.0, 2.0)])
    def test_respects_start_and_end(self, t_start, t_end):
        sim = ODESimulator(_decay_system())
        result = sim.run(np.array([1.0, 0.0]), t_start=t_start, t_end=t_end)
        assert result.times[0] == t_start
        assert result.times[-1] == t_end

    def test_respects_step_count(self):
        sim = ODESimulator(_decay_system())
        result = sim.run(np.array([1.0, 0.0]), steps=10)
        assert len(result.times) == 10

    def test_defaults(self):
        sim = ODESimulator(_decay_system())
        result = sim.run(np.array([1.0, 0.0]))
        assert result.times[0] == 0.0
        assert result.times[-1] == 1.0
        assert len(result.times) == 100


class TestOdeSimulatorForwardsKwargs:
    def test_solver_kwargs_are_forwarded(self):
        sim = ODESimulator(_decay_system())
        result = sim.run(np.array([1.0, 0.0]), rtol=1e-10, atol=1e-12)
        assert isinstance(result, SimulationTrajectory)

    def test_invalid_solver_kwarg_raises(self):
        sim = ODESimulator(_decay_system())
        with pytest.raises(ValueError):
            sim.run(np.array([1.0, 0.0]), method="NOT_A_REAL_METHOD")
