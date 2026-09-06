import numba
import numpy as np

from erlenmeyer.ode import _individual_reaction_rates
from erlenmeyer.simulator import AbstractSimulator, SimulationTrajectory

_f64_eps = np.finfo(np.float64).eps


#@numba.njit
def _gillespie_kernel(y, r_m, p_m, r_v, rng: np.random.Generator):
    # Rates for any reaction that doesn't have enough reagents go to zero
    r_v_eff = np.array(r_v)
    r_n = len(r_v)
    for r_i in range(r_n):
        if np.any(y - r_m[r_i] < 0):
            r_v_eff[r_i] = 0.0

    rates = _individual_reaction_rates(y, r_m, p_m, r_v_eff)
    total_rate = np.sum(rates)
    if total_rate == 0:
        # No more reactions possible
        return 0.0, np.zeros_like(y)
    norm_rates = rates / total_rate
    # What time?
    dt = -np.log(rng.uniform(_f64_eps, 1.0)) / total_rate
    # Which reaction happens?
    r_i = rng.choice(len(rates), p=norm_rates)
    return dt, p_m[r_i] - r_m[r_i]


class GillespieSimulator(AbstractSimulator):

    def _simulate(
        self, initial: np.ndarray, t_end: float = 1.0, seed: int | None = None, **kwargs
    ) -> SimulationTrajectory:
        y = np.array(initial)
        t = 0.0
        times = [0.0]
        traj = [initial.copy()]
        rng = np.random.default_rng(seed)
        matrices = self._system.get_reaction_matrices()
        while t < t_end:
            dt, dy = _gillespie_kernel(
                y,
                r_m=matrices.reagents_m,
                p_m=matrices.products_m,
                r_v=matrices.rates_v,
                rng=rng,
            )
            if dt == 0:
                # System's equilibrated
                times.append(t_end)
                traj.append(y.copy())
                break
            t += dt
            y += dy.astype(np.int64)
            times.append(t)
            traj.append(y.copy())

        return SimulationTrajectory(
            species=[s.species for s in self._system.species],
            times=np.array(times),
            values=np.array(traj),
        )
