import numpy as np

from erlenmeyer.simulator import SimulationTrajectory


def sample_trajectory(
    traj: SimulationTrajectory,
    sample_size: int,
    with_replacement: bool = True,
    rng: np.random.Generator | int | None = None,
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

    ``rng`` can be a :class:`numpy.random.Generator`, a seed, or ``None`` for
    unseeded randomness.
    """
    values = traj.values
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )

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
