import numba
import numpy as np
from scipy.integrate import solve_ivp

from erlenmeyer.simulator import AbstractSimulator, SimulationTrajectory


@numba.njit
def safe_log_fast(y):
    out = np.zeros_like(y, dtype=np.float64)
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
    """A simulator that integrates the mass-action kinetics with ``solve_ivp``."""

    def _simulate(
        self,
        initial: np.ndarray,
        t_start: float = 0.0,
        t_end: float = 1.0,
        steps: int = 100,
        **solver_kwargs,
    ) -> SimulationTrajectory:
        """Integrate the system from ``initial`` over ``[t_start, t_end]``.

        The time axis uses ``steps`` linearly spaced points. Any further
        keyword arguments are passed straight to
        :func:`scipy.integrate.solve_ivp`.
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
        )
