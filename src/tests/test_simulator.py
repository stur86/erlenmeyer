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

    def test_slice_produces_slice(self):
        t = np.arange(10)
        vals = np.zeros((10,2))
        vals[:,0] = t
        vals[:,1] = 2*t
        traj = SimulationTrajectory(species=["H", "O"], times=t, values=vals)
        idx = slice(2, 8, 2)
        traj_slice = traj.slice(idx)
        assert traj_slice.species == traj.species
        assert list(traj_slice.times) == [2, 4, 6]
        assert traj_slice.values.shape == (3, 2)
        assert list(traj_slice.values[:,0]) == [2, 4, 6]
        assert list(traj_slice.values[:,1]) == [4, 8, 12]


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
        result = sim.run({"H": 1.0, "O": 2.0})
        assert isinstance(result, SimulationTrajectory)
        assert result.species == ["H", "O"]

    def test_run_passes_initial_to_simulate(self):
        sim = self._make_simulator()
        result = sim.run({"H": 1.0, "O": 2.0})
        assert result.values[0].tolist() == [1.0, 2.0]

    def test_initial_follows_system_species_order(self):
        # The dictionary order must not matter: the system's order wins
        sim = self._make_simulator()
        result = sim.run({"O": 2.0, "H": 1.0})
        assert result.values[0].tolist() == [1.0, 2.0]

    def test_species_objects_are_valid_keys(self):
        sim = self._make_simulator()
        result = sim.run({Species("H"): 1.0, Species("O"): 2.0})
        assert result.values[0].tolist() == [1.0, 2.0]

    def test_species_names_and_objects_can_be_mixed(self):
        sim = self._make_simulator()
        result = sim.run({Species("H"): 1.0, "O": 2.0})
        assert result.values[0].tolist() == [1.0, 2.0]

    def test_missing_species_start_at_zero(self):
        sim = self._make_simulator()
        result = sim.run({"O": 2.0})
        assert result.values[0].tolist() == [0.0, 2.0]

    def test_empty_initial_starts_everything_at_zero(self):
        sim = self._make_simulator()
        result = sim.run({})
        assert result.values[0].tolist() == [0.0, 0.0]

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
        sim.run({"H": 1.0}, dt=0.1, steps=100)
        assert sim.received_kwargs == {"dt": 0.1, "steps": 100}

    def test_unknown_species_raises(self):
        sim = self._make_simulator()
        with pytest.raises(ValueError):
            sim.run({"H": 1.0, "Xe": 2.0})

    def test_same_species_given_twice_raises(self):
        sim = self._make_simulator()
        with pytest.raises(ValueError):
            sim.run({Species("H"): 1.0, "H": 2.0})

    def test_non_dictionary_initial_raises(self):
        sim = self._make_simulator()
        with pytest.raises(TypeError):
            sim.run(np.array([1.0, 2.0]))  # type: ignore[arg-type]

    def test_negative_initial_raises(self):
        sim = self._make_simulator()
        with pytest.raises(ValueError):
            sim.run({"H": -1.0, "O": 2.0})

    def test_zero_initial_is_allowed(self):
        sim = self._make_simulator()
        result = sim.run({"H": 0.0, "O": 0.0})
        assert isinstance(result, SimulationTrajectory)
