import numpy as np
import pytest

from dmas.agents.agent import Agent
from dmas.evolution.adaptation import adapt, copy_probability


def test_p_is_half_when_payoffs_equal():
    assert copy_probability(1.0, 1.0) == pytest.approx(0.5)


def test_p_rises_with_delta():
    ps = [copy_probability(0.0, delta) for delta in (-5, -1, 0, 1, 5)]
    assert ps == sorted(ps)
    assert ps[0] < 0.5 < ps[-1]


def test_beta_zero_gives_half():
    for delta in (-13, 0, 13):
        assert copy_probability(0.0, delta, beta=0.0) == pytest.approx(0.5)


def test_extreme_beta_does_not_overflow():
    assert copy_probability(13.0, -13.0, beta=1e9) == pytest.approx(0.0)
    assert copy_probability(-13.0, 13.0, beta=1e9) == pytest.approx(1.0)


def test_normaliser_is_configurable():
    assert copy_probability(0.0, 1.0, normaliser=1.0) > copy_probability(0.0, 1.0, normaliser=26.0)


def test_only_focal_changes_and_transition_reported():
    focal, comparison = Agent(0, "LP"), Agent(1, "TAG")
    # Huge beta and a big payoff gap make copying certain.
    result = adapt(focal, comparison, -10.0, 10.0, np.random.default_rng(0), beta=1e6)
    assert result.switched
    assert (result.old, result.new) == ("LP", "TAG")
    assert focal.strategy == "TAG"
    assert comparison.strategy == "TAG"
    assert comparison.id == 1


def test_comparison_never_changes():
    focal, comparison = Agent(0, "TAG"), Agent(1, "LP")
    for seed in range(50):
        adapt(focal, comparison, 0.0, 10.0, np.random.default_rng(seed), beta=1e6)
        focal.strategy = "TAG"
        assert comparison.strategy == "LP"


def test_same_strategy_is_never_a_switch():
    focal, comparison = Agent(0, "LAG"), Agent(1, "LAG")
    for seed in range(50):
        result = adapt(focal, comparison, -10.0, 10.0, np.random.default_rng(seed), beta=1e6)
        assert not result.switched
        assert result.old == result.new == "LAG"


def test_no_switch_when_copy_probability_zero():
    focal, comparison = Agent(0, "TAG"), Agent(1, "LP")
    result = adapt(focal, comparison, 13.0, -13.0, np.random.default_rng(0), beta=1e9)
    assert not result.switched
    assert focal.strategy == "TAG"


@pytest.mark.parametrize("beta", [0.0, 4.0])
def test_copy_frequency_matches_probability(beta):
    rng = np.random.default_rng(3)
    p = copy_probability(0.0, 6.0, beta=beta)
    hits = 0
    for _ in range(4000):
        focal = Agent(0, "LP")
        hits += adapt(focal, Agent(1, "TAG"), 0.0, 6.0, rng, beta=beta).switched
    assert hits / 4000 == pytest.approx(p, abs=0.03)
