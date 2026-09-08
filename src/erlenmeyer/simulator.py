from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from erlenmeyer.reaction import ReactionSystem
from erlenmeyer.symbols import Species


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

    def run(
        self, initial: dict[Species | str, float], **kwargs
    ) -> SimulationTrajectory:
        """Simulate the system from the given initial concentrations.

        ``initial`` maps species to their starting amount. A species may be
        given either as a :class:`Species` or as its name, and any species of
        the system left out of the dictionary starts at zero. All amounts must
        be non-negative. Any extra keyword arguments are passed on to
        :meth:`_simulate`.
        """
        return self._simulate(initial=self._initial_vector(initial), **kwargs)

    def _initial_vector(self, initial: dict[Species | str, float]) -> np.ndarray:
        """Turn a dictionary of initial amounts into a vector in species order.

        Species missing from the dictionary start at zero. Raises ValueError
        for an unknown species, for one given twice under different keys, or
        for a negative amount.
        """
        if not isinstance(initial, dict):
            raise TypeError("Initial concentrations must be given as a dictionary")
        vector = np.zeros(len(self._system.species))
        seen: set[int] = set()
        for species, amount in initial.items():
            index = self._system.get_species_index(species)
            if index in seen:
                raise ValueError(f"Species {species} given twice")
            seen.add(index)
            if amount < 0:
                raise ValueError("Initial concentrations must be non-negative")
            vector[index] = amount
        return vector

    @abstractmethod
    def _simulate(self, initial: np.ndarray, **kwargs) -> SimulationTrajectory:
        """Perform the actual simulation. Implemented by subclasses."""
