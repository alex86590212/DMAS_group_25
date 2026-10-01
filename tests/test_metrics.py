import numpy as np
import pandas as pd
import pytest

from dmas.analysis.metrics import (
    cumulative_transitions,
    entropy,
    fixation,
    oscillation,
    share_trajectories,
    switching_rate,
)


@pytest.fixture
def example_results():
    return pd.DataFrame(
        {
            "condition": ["balanced"] * 5,
            "replicate": [0] * 5,
            "generation": [0, 1, 2, 3, 4],
            "share_TAG": [0.25, 0.30, 0.35, 0.32, 0.31],
            "share_LAG": [0.25, 0.20, 0.25, 0.28, 0.27],
            "share_TP": [0.25, 0.25, 0.20, 0.20, 0.22],
            "share_LP": [0.25, 0.25, 0.20, 0.20, 0.20],
            "switches": [0, 5, 4, 3, 2],
            "trans_TAG_LAG": [0, 2, 1, 0, 0],
            "trans_LAG_TAG": [0, 0, 1, 2, 1],
            "trans_TAG_TP": [0, 1, 0, 0, 0],
            "trans_TP_TAG": [0, 0, 0, 1, 0],
            "trans_LAG_TP": [0, 1, 1, 0, 0],
            "trans_TP_LAG": [0, 0, 0, 0, 0],
            "fixed": [False, False, False, False, False],
            "fixation_generation": [-1] * 5,
        }
    )


def test_share_trajectories(example_results):
    result = share_trajectories(example_results)

    assert list(result.columns) == [
        "share_TAG",
        "share_LAG",
        "share_TP",
        "share_LP",
    ]
    assert list(result.index) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(result.loc[2, "share_TAG"], 0.35)


def test_entropy(example_results):
    result = entropy(example_results)

    assert result.loc[0] == pytest.approx(np.log(4))
    assert result.loc[4] > 0


def test_switching_rate(example_results):
    result = switching_rate(example_results, population_size=100)

    np.testing.assert_allclose(
        result.to_numpy(),
        [0.00, 0.05, 0.04, 0.03, 0.02],
    )


def test_fixation(example_results):
    fixed = example_results.copy()
    fixed.loc[4, "share_TAG"] = 1.0
    fixed.loc[4, "share_LAG"] = 0.0
    fixed.loc[4, "share_TP"] = 0.0
    fixed.loc[4, "share_LP"] = 0.0
    fixed["fixed"] = [False, False, False, False, True]
    fixed["fixation_generation"] = [4] * 5

    assert fixation(fixed) == ("TAG", 4)
    assert fixation(example_results) == (None, None)


def test_cumulative_transitions(example_results):
    result = cumulative_transitions(example_results)

    np.testing.assert_array_equal(
        result["trans_TAG_LAG"].to_numpy(),
        [0, 2, 3, 3, 3],
    )
    np.testing.assert_array_equal(
        result["trans_LAG_TAG"].to_numpy(),
        [0, 0, 1, 3, 4],
    )


def test_oscillation(example_results):
    result = oscillation(example_results)

    assert result["TAG"] == 1
    assert result["LAG"] == 2
    assert result["TP"] == 1
    assert result["LP"] == 0
