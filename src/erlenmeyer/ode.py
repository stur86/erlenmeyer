import numpy as np
from scipy.integrate import odeint

from erlenmeyer.simulator import AbstractSimulator, SimulationTrajectory


def _ode_kernel(y, t, r_m, p_m, r_v):
    """Right-hand side of the mass-action ODE system.

    ``y`` holds the current concentrations of every species. The rate of each
    reaction follows mass-action kinetics, and each reaction contributes to a
    species according to the difference between its stochiometric products and
    reagents.
    """
    # Compute the reaction rates (log-y avoids taking logarithms of zero)
    log_y = np.log(y, out=np.zeros_like(y), where=(y > 0))
    r = np.exp(r_m @ log_y) * r_v
    # Each reaction i contributes r_i * (p_ij - r_ij) to species j
    dy = r[:, None] * (p_m - r_m)
    return np.sum(dy, axis=0)


class ODESimulator(AbstractSimulator):
    """A simulator that integrates the mass-action kinetics with ``odeint``."""

    def _simulate(
        self,
        initial: np.ndarray,
        t_start: float = 0.0,
        t_end: float = 1.0,
        steps: int = 100,
        **odeint_kwargs,
    ) -> SimulationTrajectory:
        """Integrate the system from ``initial`` over ``[t_start, t_end]``.

        The time axis uses ``steps`` linearly spaced points. Any further
        keyword arguments are passed straight to :func:`scipy.integrate.odeint`.
        """
        matrices = self._system.get_reaction_matrices()
        t = np.linspace(t_start, t_end, steps)
        sol = odeint(
            _ode_kernel,
            initial,
            t,
            args=(matrices.reagents_m, matrices.products_m, matrices.rates_v),
            **odeint_kwargs,
        )
        return SimulationTrajectory(
            species=[s.species for s in self._system.species], times=t, values=sol
        )
