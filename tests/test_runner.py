import numpy as np
import pytest

from dmas.simulation.runner import play_encounter
from dmas.strategies.strategy import LAG, TAG


def test_encounter_payoffs_sum_to_zero():
    """Mean payoffs must sum to zero because Leduc Poker is zero-sum."""
    rng = np.random.default_rng(42)

    mean_focal, mean_comparison = play_encounter(
        TAG,
        LAG,
        10,
        rng,
    )

    assert np.isclose(mean_focal + mean_comparison, 0.0)


def test_encounter_is_reproducible():
    """The same seed must produce the same encounter result."""
    mean_focal_1, mean_comparison_1 = play_encounter(
        TAG,
        LAG,
        10,
        np.random.default_rng(42),
    )

    mean_focal_2, mean_comparison_2 = play_encounter(
        TAG,
        LAG,
        10,
        np.random.default_rng(42),
    )

    assert mean_focal_1 == pytest.approx(mean_focal_2)
    assert mean_comparison_1 == pytest.approx(mean_comparison_2)


def test_encounter_requires_positive_k():
    """K must be a positive number of hands."""
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        play_encounter(TAG, LAG, 0, rng)

    with pytest.raises(ValueError):
        play_encounter(TAG, LAG, -1, rng)


def test_replicate_shape_and_invariants():
    from dmas.simulation.runner import SimulationConfig, run_replicate

    config = SimulationConfig(n=20, generations=3, replicates=1, k=2)
    result = run_replicate("tag_majority", 0, config)

    assert result.shares.shape == (4, 4)
    np.testing.assert_allclose(result.shares.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.shares[0], (0.55, 0.15, 0.15, 0.15))
    assert result.transitions.shape == (3, 4, 4)
    np.testing.assert_array_equal(result.transitions.sum(axis=(1, 2)), result.switches)
    assert np.trace(result.transitions, axis1=1, axis2=2).sum() == 0


def test_run_stops_and_pads_after_fixation():
    from dmas.simulation.runner import SimulationConfig, run_replicate

    config = SimulationConfig(
        n=20,
        generations=6,
        replicates=1,
        k=2,
        initial_conditions={"all_tag": (1.0, 0.0, 0.0, 0.0)},
    )
    result = run_replicate("all_tag", 0, config)

    assert result.fixation_generation == 0
    assert result.fixed_strategy == "TAG"
    assert result.switches.sum() == 0
    np.testing.assert_allclose(result.shares, np.tile(result.shares[0], (7, 1)))
