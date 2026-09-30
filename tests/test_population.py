import numpy as np
import pytest

from dmas.evolution.population import Population

STRATEGIES = ("TAG", "LAG", "TP", "LP")


def make(counts=(55, 15, 15, 15), seed=0):
    return Population.from_composition(counts, STRATEGIES, np.random.default_rng(seed))


def test_initial_counts_match_composition_exactly():
    pop = make()
    assert len(pop) == 100
    for strategy, count in zip(STRATEGIES, (55, 15, 15, 15), strict=True):
        assert sum(a.strategy == strategy for a in pop.agents) == count


def test_population_is_shuffled_and_ids_unique():
    pop = make()
    assert [a.id for a in pop.agents] == list(range(100))
    assert [a.strategy for a in pop.agents[:55]] != ["TAG"] * 55


def test_sampled_agents_always_differ():
    pop = make(counts=(1, 1, 0, 0))
    rng = np.random.default_rng(1)
    for _ in range(1000):
        focal, comparison = pop.sample_pair(rng)
        assert focal is not comparison


def test_shares_sum_to_one():
    shares = make().shares()
    assert list(shares) == list(STRATEGIES)
    assert shares["TAG"] == 0.55
    assert sum(shares.values()) == pytest.approx(1.0)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        Population.from_composition((1, 2), STRATEGIES, np.random.default_rng(0))
