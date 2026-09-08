import warnings

import numpy as np
import pytest

from erlenmeyer.sampling import sample_trajectory
from erlenmeyer.simulator import SimulationTrajectory


def _trajectory(values, species=("A", "B")):
    values = np.asarray(values)
    return SimulationTrajectory(
        species=list(species),
        times=np.arange(values.shape[0], dtype=float),
        values=values,
    )


class TestSampleWithReplacement:
    def test_shape_matches_trajectory(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8], [1.0, 0.0]])
        result = sample_trajectory(traj, 100, rng=0)
        assert result.shape == traj.values.shape

    def test_counts_add_up_to_sample_size(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8], [0.1, 0.9]])
        result = sample_trajectory(traj, 37, rng=0)
        assert result.sum(axis=1).tolist() == [37, 37, 37]

    def test_zero_probability_species_is_never_sampled(self):
        traj = _trajectory([[1.0, 0.0], [1.0, 0.0]])
        result = sample_trajectory(traj, 50, rng=0)
        assert result.tolist() == [[50, 0], [50, 0]]

    def test_can_exceed_the_population(self):
        # With replacement there is no population to exhaust
        traj = _trajectory([[0.5, 0.5]])
        result = sample_trajectory(traj, 10**6, rng=0)
        assert result.sum() == 10**6

    def test_floating_point_values_are_allowed(self):
        traj = _trajectory(np.array([[0.25, 0.75]]))
        result = sample_trajectory(traj, 10, with_replacement=True, rng=0)
        assert result.shape == (1, 2)

    def test_converges_to_the_probabilities(self):
        traj = _trajectory([[0.3, 0.7]])
        result = sample_trajectory(traj, 100000, rng=0)
        assert result[0, 0] / 100000 == pytest.approx(0.3, abs=0.01)


class TestNormalization:
    def test_unnormalized_values_are_normalized(self):
        traj = _trajectory([[3.0, 7.0]])
        result = sample_trajectory(traj, 100000, rng=0)
        assert result.sum() == 100000
        assert result[0, 0] / 100000 == pytest.approx(0.3, abs=0.01)

    def test_integer_populations_are_normalized(self):
        traj = _trajectory(np.array([[30, 70]]))
        result = sample_trajectory(traj, 100000, rng=0)
        assert result[0, 0] / 100000 == pytest.approx(0.3, abs=0.01)

    def test_each_time_point_is_normalized_separately(self):
        # The second row has ten times the total of the first, but the same ratio
        traj = _trajectory([[3.0, 7.0], [30.0, 70.0]])
        result = sample_trajectory(traj, 100000, rng=0)
        assert result.sum(axis=1).tolist() == [100000, 100000]
        assert result[0, 0] / 100000 == pytest.approx(0.3, abs=0.01)
        assert result[1, 0] / 100000 == pytest.approx(0.3, abs=0.01)

    def test_scaling_the_values_does_not_change_the_sample(self):
        values = np.array([[0.5, 0.5], [0.2, 0.8]])
        normalized = sample_trajectory(_trajectory(values), 100, rng=3)
        scaled = sample_trajectory(_trajectory(values * 17.0), 100, rng=3)
        assert normalized.tolist() == scaled.tolist()


class TestEmptyTimePoints:
    def test_empty_time_point_yields_an_all_zero_sample(self):
        traj = _trajectory([[0.0, 0.0]])
        result = sample_trajectory(traj, 100, rng=0)
        assert result.tolist() == [[0, 0]]

    def test_empty_time_points_do_not_disturb_the_others(self):
        traj = _trajectory([[0.3, 0.7], [0.0, 0.0], [1.0, 0.0]])
        result = sample_trajectory(traj, 100, rng=0)
        assert result.sum(axis=1).tolist() == [100, 0, 100]
        assert result[2].tolist() == [100, 0]

    def test_an_all_empty_trajectory_is_all_zero(self):
        traj = _trajectory(np.zeros((4, 2)))
        result = sample_trajectory(traj, 100, rng=0)
        assert result.shape == (4, 2)
        assert not result.any()

    def test_no_warning_is_emitted(self):
        traj = _trajectory([[0.0, 0.0], [0.5, 0.5]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sample_trajectory(traj, 100, rng=0)

    def test_counts_are_integers(self):
        traj = _trajectory([[0.0, 0.0], [0.5, 0.5]])
        result = sample_trajectory(traj, 100, rng=0)
        assert np.issubdtype(result.dtype, np.integer)


class TestSampleWithoutReplacement:
    def test_shape_matches_trajectory(self):
        traj = _trajectory(np.array([[50, 50], [20, 80]]))
        result = sample_trajectory(traj, 10, with_replacement=False, rng=0)
        assert result.shape == traj.values.shape

    def test_counts_add_up_to_sample_size(self):
        traj = _trajectory(np.array([[50, 50], [20, 80]]))
        result = sample_trajectory(traj, 13, with_replacement=False, rng=0)
        assert result.sum(axis=1).tolist() == [13, 13]

    def test_never_samples_more_than_the_population(self):
        traj = _trajectory(np.array([[3, 97]] * 20))
        result = sample_trajectory(traj, 10, with_replacement=False, rng=0)
        assert np.all(result <= traj.values)

    def test_sampling_the_whole_population_returns_it(self):
        traj = _trajectory(np.array([[4, 6], [1, 9]]))
        result = sample_trajectory(traj, 10, with_replacement=False, rng=0)
        assert result.tolist() == [[4, 6], [1, 9]]

    def test_floating_point_values_raise(self):
        traj = _trajectory(np.array([[0.5, 0.5]]))
        with pytest.raises(ValueError, match="floating point"):
            sample_trajectory(traj, 1, with_replacement=False)

    def test_sample_larger_than_population_raises(self):
        traj = _trajectory(np.array([[1, 2]]))
        with pytest.raises(ValueError):
            sample_trajectory(traj, 10, with_replacement=False, rng=0)


class TestRandomness:
    @pytest.mark.parametrize("with_replacement", [True, False])
    def test_same_seed_gives_same_sample(self, with_replacement):
        values = (
            np.array([[0.5, 0.5], [0.2, 0.8]])
            if with_replacement
            else np.array([[50, 50], [20, 80]])
        )
        traj = _trajectory(values)
        first = sample_trajectory(traj, 10, with_replacement, rng=42)
        second = sample_trajectory(traj, 10, with_replacement, rng=42)
        assert first.tolist() == second.tolist()

    def test_different_seeds_give_different_samples(self):
        traj = _trajectory([[0.5, 0.5]] * 20)
        first = sample_trajectory(traj, 100, rng=1)
        second = sample_trajectory(traj, 100, rng=2)
        assert first.tolist() != second.tolist()

    def test_accepts_a_generator(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8]])
        first = sample_trajectory(traj, 10, rng=np.random.default_rng(7))
        second = sample_trajectory(traj, 10, rng=np.random.default_rng(7))
        assert first.tolist() == second.tolist()

    def test_a_generator_advances_between_calls(self):
        traj = _trajectory([[0.5, 0.5]] * 20)
        generator = np.random.default_rng(7)
        first = sample_trajectory(traj, 100, rng=generator)
        second = sample_trajectory(traj, 100, rng=generator)
        assert first.tolist() != second.tolist()

    def test_no_seed_still_works(self):
        traj = _trajectory([[0.5, 0.5]])
        result = sample_trajectory(traj, 10)
        assert result.sum() == 10


class TestTimes:
    def test_shape_matches_number_of_times(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        times = np.array([0.0, 1.5, 2.0])
        result = sample_trajectory(traj, 100, times=times, rng=0)
        assert result.shape == (3, 2)

    def test_counts_add_up_to_sample_size(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        times = np.array([0.0, 0.5, 1.0, 2.0])
        result = sample_trajectory(traj, 50, times=times, rng=0)
        assert result.sum(axis=1).tolist() == [50, 50, 50, 50]

    def test_exact_trajectory_times(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        result = sample_trajectory(traj, 100, times=traj.times, rng=0)
        assert result.shape == traj.values.shape

    def test_between_times_uses_previous_value(self):
        traj = _trajectory([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        # Time 0.5 should use value from t=0 (previous)
        result = sample_trajectory(traj, 100, times=np.array([0.5]), rng=0)
        assert result[0].tolist() == [100, 0]

    def test_below_first_time_raises(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8]])
        with pytest.raises(ValueError, match="outside the trajectory's time range"):
            sample_trajectory(traj, 100, times=np.array([-1.0]), rng=0)

    def test_after_last_time_raises(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8]])
        with pytest.raises(ValueError, match="outside the trajectory's time range"):
            sample_trajectory(traj, 100, times=np.array([5.0]), rng=0)

    def test_any_out_of_range_time_raises(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        times = np.array([0.0, 3.0, 1.0])
        with pytest.raises(ValueError, match="outside the trajectory's time range"):
            sample_trajectory(traj, 100, times=times, rng=0)

    def test_boundary_times_are_allowed(self):
        traj = _trajectory([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])
        times = np.array([0.0, 2.0])
        result = sample_trajectory(traj, 100, times=times, rng=0)
        assert result.shape == (2, 2)

    def test_without_replacement(self):
        traj = _trajectory(np.array([[50, 50], [20, 80]]))
        times = np.array([0.0, 1.0])
        result = sample_trajectory(traj, 10, with_replacement=False, times=times, rng=0)
        assert result.shape == (2, 2)
        assert result.sum(axis=1).tolist() == [10, 10]

    def test_single_time_point(self):
        traj = _trajectory([[0.3, 0.7], [0.6, 0.4]])
        result = sample_trajectory(traj, 100, times=np.array([1.0]), rng=0)
        assert result.shape == (1, 2)
        assert result.sum() == 100
