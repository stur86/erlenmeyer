from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from erlenmeyer.simulator import SimulationTrajectory, SimulationType


def _only_one_of(a: Any, b: Any) -> bool:
    return ((a is None)+(b is None)) == 1

def _split_n_by_proportions(n: int, proportions: np.ndarray) -> np.ndarray:
    proportions = np.asarray(proportions)
    # Normalize proportions just in case they don't sum to 1.0
    proportions = proportions / proportions.sum()

    # 1. Calculate ideal float sizes and their floor integers
    exact_counts = n * proportions
    floor_counts = np.floor(exact_counts).astype(int)

    # 2. Find out how many elements are missing due to floor rounding
    remainder = n - floor_counts.sum()

    # 3. Allocate remaining units to the bins with the largest fractional parts
    fractional_parts = exact_counts - floor_counts
    # Get indices sorted by largest fractional part descending
    rank_indices = np.argsort(-fractional_parts)

    # Add 1 to the top 'remainder' bins
    floor_counts[rank_indices[:remainder]] += 1

    return floor_counts

def sample_trajectory(
    traj: SimulationTrajectory,
    sample_size: int | None = None,
    volume_fraction: float | None = None,
    with_replacement: bool = True,
    rng: np.random.Generator | int | None = None,
    times: np.ndarray | None = None,
    volume: float = 1.0
) -> np.ndarray:
    """Draw and count a finite sample at each point of a trajectory.

    This emulates measuring only a small aliquot of the system instead of its
    entirety. For every time point it returns the counts of each species found
    in the aliquot, with shape ``[times, species]``.

    With replacement, each time point of ``traj.values`` is treated as a set
    of probabilities and sampled from a multinomial distribution; the values
    are normalized if needed, and a time point empty of every species yields
    an all-zero sample. Without replacement, the values are treated as a
    population of distinguishable items and sampled from a multivariate
    hypergeometric distribution; they must therefore be integers, and the
    number drawn can not exceed the population at any time point.

    The values of the trajectory are taken to be particles per unit volume,
    so the amount of particles present at a time point is ``volume`` times
    the sum of its values. This only comes into play when sampling by
    ``volume_fraction``: ``volume`` fixes how many particles the fraction
    applies to.

    Sampling can be done on a user-defined time axis. Absent that, the
    trajectory's own times are used.

    Parameters
    ----------
    traj : SimulationTrajectory
        The trajectory to sample.
    sample_size : int, optional
        Number of particles to draw at each time point. Mutually exclusive
        with ``volume_fraction``.
    volume_fraction : float, optional
        Fraction of the particles present at each time point to draw. The
        number drawn at a time point follows a binomial distribution with
        ``volume`` times the total amount present there. Mutually exclusive
        with ``sample_size``.
    with_replacement : bool, default True
        Whether to sample from a multinomial (with replacement) or from a
        multivariate hypergeometric (without replacement) distribution.
    rng : numpy.random.Generator or int or None, default None
        A random number generator, or a seed for a fresh one. ``None`` uses
        unseeded randomness.
    times : numpy.ndarray, optional
        Time points to sample at instead of the trajectory's own ones. They
        must lie within the trajectory's time range. Gillespie trajectories
        are interpolated with the previous value, ODE ones linearly.
    volume : float, default 1.0
        Volume of the system. The particles of a time point are ``volume``
        times the sum of its values, which sets how many are available when
        sampling by ``volume_fraction``. Ignored when ``sample_size`` is
        given. Must be positive.

    Returns
    -------
    numpy.ndarray
        Integer counts of each species at every time point, with shape
        ``[times, species]``.

    Raises
    ------
    ValueError
        If both or neither of ``sample_size`` and ``volume_fraction`` are
        given, if ``volume`` is not positive, or if values are sampled
        without replacement and are not integers.
    """
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    if volume <= 0:
        raise ValueError("Volume must be positive")
    if not _only_one_of(sample_size, volume_fraction):
        raise ValueError("One and only one between sample_size and volume_fraction must be used")

    if times is None:
        values = traj.values
    else:
        # Extract times by interpolation with closest previous value
        if np.any(times < traj.times[0]) or np.any(times > traj.times[-1]):
            raise ValueError(
                "Requested sampling times are outside the trajectory's time range "
                f"[{traj.times[0]}, {traj.times[-1]}]"
            )
        # Interpolation type: previous for a Gillespie (stepwise constant)
        # trajectory, linear otherwise
        _kind = (
            "previous" if traj.simulation_type is SimulationType.GILLESPIE else "linear"
        )
        values = interp1d(traj.times, traj.values, kind=_kind, axis=0)(times)
        values = values.astype(traj.values.dtype)

    totals = np.sum(values, axis=1)
    # Sample counts
    if sample_size is not None:
        sample_counts = np.full_like(totals, sample_size, dtype=np.int64)
    else:
        # Volume multiplies the totals if we're doing a volume fraction measurement,
        # as we interpret them as densities, particles/unit volume. It's ignored otherwise
        sample_counts = generator.binomial(np.floor(volume*totals).astype(np.int64), volume_fraction) # type: ignore

    sample_traj = np.zeros(values.shape, dtype=np.int64)

    if with_replacement:
        # Time points with nothing in them have nothing to sample, and would
        # not be normalizable; they simply yield an all-zero count
        populated = totals > 0

        # Sample with multinomial distribution, one draw per time point
        sample_traj[populated] = generator.multinomial(
            sample_counts[populated], values[populated] / totals[populated][:, None]
        )
    else:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(
                "Can not do sampling without replacement on a floating point sample"
            )

        for i, (v, s_n) in enumerate(zip(values, sample_counts)):
            # Proportions, if necessary
            if volume != 1:
                # We need to reassign counts
                tot_v = np.sum(v)
                v = _split_n_by_proportions(tot_v, (1.0*v)/tot_v)
            sample_traj[i] = generator.multivariate_hypergeometric(v, s_n)

    return sample_traj
