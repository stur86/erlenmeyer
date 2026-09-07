from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from erlenmeyer.reaction import ReactionSystem


@dataclass(frozen=True)
class SimulationTrajectory:
    """The result of a simulation.

    ``times`` holds the time points at which the system was sampled, and
    ``values`` the corresponding concentrations, with shape [times, species].
    ``species`` names the columns of ``values`` in order.
    """

    species: list[str]
    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.times.shape) != 1:
            raise ValueError("Times axis must be one-dimensional")
        if self.values.shape != (self.times.shape[0], len(self.species)):
            raise ValueError("Values must have shape [times, species]")
        if not all(isinstance(s, str) for s in self.species):
            raise ValueError("Species must be strings")

    def slice(self, idx: slice) -> "SimulationTrajectory":
        return SimulationTrajectory(
            species=self.species,
            times=self.times[idx],
            values=self.values[idx]
        )

    def __len__(self) -> int:
        return self.times.shape[0]

class AbstractSimulator(ABC):
    """A base class for solvers that evolve a :class:`ReactionSystem` in time."""

    _system: ReactionSystem

    def __init__(self, reaction_system: ReactionSystem) -> None:
        self._system = reaction_system

    def run(self, initial: np.ndarray, **kwargs) -> SimulationTrajectory:
        """Simulate the system from the given initial concentrations.

        ``initial`` must have one entry per species, and all entries must be
        non-negative. Any extra keyword arguments are passed on to
        :meth:`_simulate`.
        """
        if initial.shape != (len(self._system.species),):
            raise ValueError("Initial concentrations must have shape [species]")
        if np.any(initial < 0):
            raise ValueError("Initial concentrations must be non-negative")
        return self._simulate(initial=initial, **kwargs)

    @abstractmethod
    def _simulate(self, initial: np.ndarray, **kwargs) -> SimulationTrajectory:
        """Perform the actual simulation. Implemented by subclasses."""
