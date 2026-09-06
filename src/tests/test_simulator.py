import dataclasses

import numpy as np
import pytest

from erlenmeyer.reaction import ReactionSystem
from erlenmeyer.simulator import (
    AbstractSimulator,
    SimulationTrajectory,
)
from erlenmeyer.symbols import Species


class TestSimulationTrajectoryConstruction:
    def test_valid_trajectory(self):
        t = SimulationTrajectory(
            species=["H", "O"], times=np.array([0.0, 1.0]), values=np.array([[1.0, 0.0], [0.5, 0.5]])
        )
        assert t.species == ["H", "O"]
        assert t.times.tolist() == [0.0, 1.0]
        assert t.values.shape == (2, 2)

    @pytest.mark.parametrize(
        "times",
        [np.array([[0.0, 1.0]]), np.zeros((2, 2)), np.zeros((1, 1, 1))],
    )
    def test_times_must_be_one_dimensional(self, times):
        with pytest.raises(ValueError):
            SimulationTrajectory(
                species=["H"], times=times, values=np.zeros((times.shape[0], 1))
            )

    def test_values_must_match_times_and_species(self):
        with pytest.raises(ValueError):
            SimulationTrajectory(
                species=["H", "O"], times=np.array([0.0, 1.0]), values=np.zeros((2, 3))
            )
        with pytest.raises(ValueError):
            SimulationTrajectory(
                species=["H", "O"], times=np.array([0.0, 1.0]), values=np.zeros((5, 2))
            )

    def test_frozen_instance(self):
        t = SimulationTrajectory(
            species=["H"], times=np.array([0.0]), values=np.array([[1.0]])
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.species = ["O"]  # type: ignore

    def test_species_must_be_strings(self):
        h = Species("H")
        with pytest.raises(ValueError):
            SimulationTrajectory(
                species=[h], times=np.array([0.0]), values=np.array([[1.0]]) # type: ignore
            )


class TestAbstractSimulator:
    def _make_system(self):
        h, o, h2o = Species("H"), Species("O"), Species("H2O")
        system = ReactionSystem([h, o, h2o])
        return system

    def test_cannot_be_instantiated(self):
        system = self._make_system()
        with pytest.raises(TypeError):
            AbstractSimulator(system)  # type: ignore[abstract]

    def test_subclass_must_implement_simulate(self):
        class Incomplete(AbstractSimulator):
            pass

        system = self._make_system()
        with pytest.raises(TypeError):
            Incomplete(system)  # type: ignore[abstract]


class TestAbstractSimulatorRun:
    def _make_simulator(self):
        class RecordingSimulator(AbstractSimulator):
            def _simulate(self, initial, **kwargs):
                t = np.array([0.0, 1.0])
                values = np.stack([initial, initial])
                return SimulationTrajectory(
                    species=[s.species for s in self._system.species],
                    times=t,
                    values=values,
                )

        system = ReactionSystem([Species("H"), Species("O")])
        return RecordingSimulator(system)

    def test_run_returns_a_trajectory(self):
        sim = self._make_simulator()
        result = sim.run(np.array([1.0, 2.0]))
        assert isinstance(result, SimulationTrajectory)
        assert result.species == ["H", "O"]

    def test_run_passes_initial_to_simulate(self):
        class CapturingSimulator(AbstractSimulator):
            def __init__(self, system):
                super().__init__(system)
                self.received = None

            def _simulate(self, initial, **kwargs):
                self.received = initial
                return SimulationTrajectory(
                    species=[s.species for s in self._system.species],
                    times=np.array([0.0]),
                    values=initial[None, :],
                )

        sim = CapturingSimulator(ReactionSystem([Species("H")]))
        initial = np.array([3.0])
        sim.run(initial)
        assert sim.received is initial

    def test_run_forwards_kwargs(self):
        class CapturingSimulator(AbstractSimulator):
            def __init__(self, system):
                super().__init__(system)
                self.received_kwargs = None

            def _simulate(self, initial, **kwargs):
                self.received_kwargs = kwargs
                return SimulationTrajectory(
                    species=[s.species for s in self._system.species],
                    times=np.array([0.0]),
                    values=initial[None, :],
                )

        sim = CapturingSimulator(ReactionSystem([Species("H")]))
        sim.run(np.array([1.0]), dt=0.1, steps=100)
        assert sim.received_kwargs == {"dt": 0.1, "steps": 100}

    def test_initial_shape_must_match_species_count(self):
        sim = self._make_simulator()
        with pytest.raises(ValueError):
            sim.run(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            sim.run(np.array([1.0]))

    def test_initial_shape_must_be_one_dimensional(self):
        sim = self._make_simulator()
        with pytest.raises(ValueError):
            sim.run(np.zeros((2, 1)))

    def test_negative_initial_raises(self):
        sim = self._make_simulator()
        with pytest.raises(ValueError):
            sim.run(np.array([-1.0, 2.0]))

    def test_zero_initial_is_allowed(self):
        sim = self._make_simulator()
        result = sim.run(np.array([0.0, 0.0]))
        assert isinstance(result, SimulationTrajectory)
