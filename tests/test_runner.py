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

    assert mean_focal_1 == mean_focal_2
    assert mean_comparison_1 == mean_comparison_2


def test_encounter_requires_positive_k():
    """K must be a positive number of hands."""
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        play_encounter(TAG, LAG, 0, rng)

    with pytest.raises(ValueError):
        play_encounter(TAG, LAG, -1, rng)