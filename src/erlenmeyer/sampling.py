import numpy as np
from scipy.interpolate import interp1d

from erlenmeyer.simulator import SimulationTrajectory


def sample_trajectory(
    traj: SimulationTrajectory,
    sample_size: int,
    with_replacement: bool = True,
    rng: np.random.Generator | int | None = None,
    times: np.ndarray | None = None
) -> np.ndarray:
    """Draw a finite sample of ``sample_size`` items at each point of a trajectory.

    This emulates the effect of measuring only a small aliquot of the system
    instead of its entirety, and returns the counts of each species found in
    that aliquot, with shape [times, species].

    With replacement, each time point of ``traj.values`` is treated as a set of
    probabilities and sampled from a multinomial distribution; the values are
    normalized if necessary, and a time point that is empty of every species
    yields an all-zero sample. Without replacement, they are treated as populations
    of distinguishable items and sampled from a multivariate hypergeometric
    distribution; the values must therefore be integers, and ``sample_size`` can
    not exceed the total population at any time point.

    Sampling can be done on a user-defined time axis. Absent that, it will just
    use the trajectory's own times.

    ``rng`` can be a :class:`numpy.random.Generator`, a seed, or ``None`` for
    unseeded randomness.
    """
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    if times is None:
        values = traj.values
    else:
        # Extract times by interpolation with closest previous value
        if np.any(times < traj.times[0]) or np.any(times > traj.times[-1]):
            raise ValueError(
                "Requested sampling times are outside the trajectory's time range "
                f"[{traj.times[0]}, {traj.times[-1]}]"
            )
        values = interp1d(traj.times, traj.values, kind='previous', axis=0)(times)
        values = values.astype(traj.values.dtype)

    if with_replacement:
        totals = np.sum(values, axis=1)
        # Time points with nothing in them have nothing to sample, and would
        # not be normalizable; they simply yield an all-zero count
        populated = totals > 0
        sample_traj = np.zeros(values.shape, dtype=np.int64)
        # Sample with multinomial distribution, one draw per time point
        sample_traj[populated] = generator.multinomial(
            sample_size, values[populated] / totals[populated][:, None]
        )
        return sample_traj

    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(
            "Can not do sampling without replacement on a floating point sample"
        )

    return np.array(
        [generator.multivariate_hypergeometric(v, sample_size) for v in values]
    )
