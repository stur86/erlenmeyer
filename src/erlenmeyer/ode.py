import numba
import numpy as np
from scipy.integrate import solve_ivp

from erlenmeyer.simulator import AbstractSimulator, SimulationTrajectory, SimulationType


@numba.njit
def safe_log_fast(y):
    # Masked entries go to a large negative sentinel, so that a reaction
    # whose reactant is absent has rate exp(-750) -> 0 instead of exp(0) -> 1
    out = np.full_like(y, -750.0)
    # Numba will auto-vectorize this loop
    for i in np.ndindex(y.shape):
        if y[i] > 0:
            out[i] = np.log(y[i])
    return out

@numba.njit
def _individual_reaction_rates(y, r_m, p_m, r_v) -> np.ndarray:
    # Compute the reaction rates (log-y avoids taking logarithms of zero)
    log_y = safe_log_fast(y)
    r = np.exp(r_m @ log_y) * r_v
    return r

@numba.njit
def _ode_kernel(t, y, r_m, p_m, r_v):
    """Right-hand side of the mass-action ODE system.

    ``y`` holds the current concentrations of every species. The rate of each
    reaction follows mass-action kinetics, and each reaction contributes to a
    species according to the difference between its stochiometric products and
    reagents.
    """
    # Each reaction i contributes r_i * (p_ij - r_ij) to species j
    dy = _individual_reaction_rates(y, r_m, p_m, r_v)[:,None]*(p_m-r_m)
    return np.sum(dy, axis=0)


class ODESimulator(AbstractSimulator):
    """A simulator that integrates the mass-action kinetics with ``solve_ivp``.

    Each run produces a deterministic trajectory of continuous
    concentrations, sampled on a linearly spaced time axis.
    """

    def _simulate(
        self,
        initial: np.ndarray,
        t_start: float = 0.0,
        t_end: float = 1.0,
        steps: int = 100,
        **solver_kwargs,
    ) -> SimulationTrajectory:
        """Integrate the mass-action kinetics of the system over time.

        Parameters
        ----------
        initial : numpy.ndarray
            Initial concentrations, in species order.
        t_start : float, default 0.0
            Start of the time interval.
        t_end : float, default 1.0
            End of the time interval.
        steps : int, default 100
            Number of linearly spaced points on the time axis.
        **solver_kwargs
            Passed straight to :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        SimulationTrajectory
            A trajectory of type ``ODE``, with continuous concentrations on a
            linearly spaced time axis.
        """
        matrices = self._system.get_reaction_matrices()
        t = np.linspace(t_start, t_end, steps)
        sol = solve_ivp(
            _ode_kernel,
            (t_start, t_end),
            initial,
            t_eval=t,
            args=(matrices.reagents_m, matrices.products_m, matrices.rates_v),
            **solver_kwargs,
        )
        return SimulationTrajectory(
            species=[s.species for s in self._system.species],
            times=t,
            values=sol.y.T,
            simulation_type=SimulationType.ODE,
        )
