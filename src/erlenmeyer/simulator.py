from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from erlenmeyer.reaction import ReactionSystem
from erlenmeyer.symbols import Species


class SimulationType(Enum):
    """Identifies how a trajectory was simulated.

    The member of a :class:`SimulationTrajectory` tells downstream tools how
    to read its values. ``ODE`` marks trajectories produced by
    :class:`~erlenmeyer.ode.ODESimulator`, which hold continuous
    concentrations. ``GILLESPIE`` marks trajectories produced by
    :class:`~erlenmeyer.gillespie.GillespieSimulator`, which hold integer
    molecule counts.
    """

    ODE = "ode"
    GILLESPIE = "gillespie"


@dataclass(frozen=True)
class SimulationTrajectory:
    """The result of a simulation.

    Holds the time axis of a simulation, the values sampled along it, and the
    metadata needed to interpret them.

    Attributes
    ----------
    species : list of str
        Name of each species, in system order. These name the columns of
        ``values``.
    times : numpy.ndarray
        Time points at which the system was sampled, shape ``[times]``.
    values : numpy.ndarray
        Amount of each species at each time point, shape ``[times, species]``.
        ODE trajectories hold floating point concentrations, Gillespie ones
        integer molecule counts.
    simulation_type : SimulationType
        How the trajectory was produced, which determines how ``values``
        must be read.
    """

    species: list[str]
    times: np.ndarray
    values: np.ndarray
    simulation_type: SimulationType

    def __post_init__(self) -> None:
        if len(self.times.shape) != 1:
            raise ValueError("Times axis must be one-dimensional")
        if self.values.shape != (self.times.shape[0], len(self.species)):
            raise ValueError("Values must have shape [times, species]")
        if not all(isinstance(s, str) for s in self.species):
            raise ValueError("Species must be strings")
        if self.simulation_type is SimulationType.GILLESPIE and not np.issubdtype(
            self.values.dtype, np.integer
        ):
            raise ValueError("Values must be integers for a Gillespie trajectory")

    def slice(self, idx: slice) -> "SimulationTrajectory":
        """Return a new trajectory restricted to a slice of its time points.

        Parameters
        ----------
        idx : slice
            Index slice applied to the ``times`` and ``values`` axes.

        Returns
        -------
        SimulationTrajectory
            A new trajectory with the sliced ``times`` and ``values`` and the
            same ``species`` and ``simulation_type``.
        """
        return SimulationTrajectory(
            species=self.species,
            times=self.times[idx],
            values=self.values[idx],
            simulation_type=self.simulation_type,
        )

    def __len__(self) -> int:
        """Return the number of time points in the trajectory."""
        return self.times.shape[0]


class AbstractSimulator(ABC):
    """Base class for solvers that evolve a :class:`ReactionSystem` in time.

    Subclasses must implement :meth:`_simulate` to produce the actual
    trajectory; the shared :meth:`run` machinery takes care of the rest.
    """

    _system: ReactionSystem

    def __init__(self, reaction_system: ReactionSystem) -> None:
        self._system = reaction_system

    def run(
        self, initial: dict[Species | str, float], **kwargs
    ) -> SimulationTrajectory:
        """Simulate the system from the given initial concentrations.

        Parameters
        ----------
        initial : dict of Species or str to float
            Maps each species to its starting amount. A species may be given
            either as a :class:`Species` or as its name; any species of the
            system left out of the dictionary starts at zero. All amounts
            must be non-negative.
        **kwargs
            Passed on to :meth:`_simulate`.

        Returns
        -------
        SimulationTrajectory
            The simulated trajectory.

        Raises
        ------
        TypeError
            If ``initial`` is not a dictionary.
        ValueError
            If an unknown species is given, the same species is given twice
            under different keys, or an amount is negative.
        """
        return self._simulate(initial=self._initial_vector(initial), **kwargs)

    def _initial_vector(self, initial: dict[Species | str, float]) -> np.ndarray:
        """Turn a dictionary of initial amounts into a vector in species order.

        Parameters
        ----------
        initial : dict of Species or str to float
            Maps species to starting amounts. A species may be given either
            as a :class:`Species` or as its name.

        Returns
        -------
        numpy.ndarray
            The initial amounts as a vector in system species order. Species
            missing from the dictionary start at zero.

        Raises
        ------
        TypeError
            If ``initial`` is not a dictionary.
        ValueError
            If an unknown species is given, the same species is given twice
            under different keys, or an amount is negative.
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
        """Perform the actual simulation.

        Parameters
        ----------
        initial : numpy.ndarray
            Initial state as a vector in species order.
        **kwargs
            Simulation-specific keyword arguments.

        Returns
        -------
        SimulationTrajectory
            The simulated trajectory.
        """
