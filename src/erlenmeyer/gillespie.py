import numba
import numpy as np

from erlenmeyer.simulator import AbstractSimulator, SimulationTrajectory, SimulationType

_f64_eps = np.finfo(np.float64).eps


@numba.njit
def _seed_rng(seed):
    """Seed the RNG used inside compiled kernels, if a seed is given."""
    if seed is not None:
        np.random.seed(seed)


@numba.njit
def _stochastic_propensity(y, r_m, r_v):
    """Propensities for every reaction under the stochastic formulation.

    The propensity of a reaction is ``k_i * prod_j n_j!/(n_j - r_ij)!``, i.e.
    the product over its reagents of the falling factorial of the current
    population. For a reactant present ``n`` times with stochiometric
    coefficient ``a`` this is ``n*(n-1)*...*(n-a+1)``, which is the number of
    distinct ways to pick the ``a`` colliding molecules. A reaction whose
    reagents cannot be satisfied has propensity zero.
    """
    n_rxn, n_spec = r_m.shape
    rates = np.zeros(n_rxn)
    for i in range(n_rxn):
        prop = r_v[i]
        for j in range(n_spec):
            n = int(r_m[i, j])
            if n > 0:
                if y[j] < n:
                    prop = 0.0
                    break
                for k in range(n):
                    prop *= y[j] - k
        rates[i] = prop
    return rates


@numba.njit
def _select_reaction(u, cum_rates):
    """Pick a reaction index for a uniform ``u`` in [0, 1) over cum_rates."""
    target = u * cum_rates[-1]
    for i in range(len(cum_rates)):
        if target < cum_rates[i]:
            return i
    return len(cum_rates) - 1


@numba.njit
def _gillespie_kernel(y, r_m, p_m, r_v):
    """Perform one Gillespie step, returning ``(dt, dcounts)``.

    ``y`` holds the current integer population of every species. The step
    draws the time to the next reaction from an exponential distribution and
    selects which reaction fires, weighted by its stochastic propensity.

    If no reaction is possible, returns ``(0.0, zeros)`` so the caller can
    stop.
    """
    rates = _stochastic_propensity(y, r_m, r_v)
    total_rate = np.sum(rates)
    if total_rate == 0:
        return 0.0, np.zeros_like(y)

    dt = -np.log(np.random.uniform(_f64_eps, 1.0)) / total_rate
    # Cumulative rates, then pick a reaction weighted by propensity
    cum = np.cumsum(rates)
    r_i = _select_reaction(np.random.rand(), cum)
    return dt, (p_m[r_i] - r_m[r_i]).astype(np.int64)


class GillespieSimulator(AbstractSimulator):
    """A simulator that samples trajectories with the Gillespie algorithm.

    Each run produces one stochastic trajectory, where the values are integer
    molecule counts and the time points are the times of the individual
    reaction events. They are therefore not evenly spaced and their number
    changes from run to run.
    """

    def _simulate(
        self,
        initial: np.ndarray,
        t_end: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ) -> SimulationTrajectory:
        """Simulate the system stochastically from ``initial`` up to ``t_end``.

        Parameters
        ----------
        initial : numpy.ndarray
            Initial integer populations, in species order.
        t_end : float, default 1.0
            Time limit of the simulation.
        seed : int or None, default None
            Seed for the global NumPy random number generator. When given,
            runs are reproducible.
        **kwargs
            Ignored.

        Returns
        -------
        SimulationTrajectory
            A trajectory of type ``GILLESPIE``, with integer molecule counts
            and the reaction event times.

        Raises
        ------
        ValueError
            If ``initial`` contains non-integer amounts.
        """
        _seed_rng(seed)
        if not np.all(np.equal(np.mod(initial, 1), 0)):
            raise ValueError("Initial populations must be integers")
        y = np.array(initial, dtype=np.int64)
        t = 0.0
        times = [0.0]
        traj = [y.copy()]
        matrices = self._system.get_reaction_matrices()
        while t < t_end:
            dt, dy = _gillespie_kernel(
                y,
                r_m=matrices.reagents_m,
                p_m=matrices.products_m,
                r_v=matrices.rates_v,
            )
            if dt == 0:
                # System's equilibrated
                times.append(t_end)
                traj.append(y.copy())
                break
            t += dt
            y += dy
            times.append(t)
            traj.append(y.copy())

        return SimulationTrajectory(
            species=[s.species for s in self._system.species],
            times=np.array(times),
            values=np.array(traj),
            simulation_type=SimulationType.GILLESPIE,
        )
