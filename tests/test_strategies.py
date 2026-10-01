import numpy as np
import pytest

from dmas.leduc.game import U_MAX, Action, Observation, play_encounter, play_hand
from dmas.strategies.equity import equity
from dmas.strategies.strategy import NAMES, STRATEGIES, ThresholdStrategy


def observations(facing_bet, can_raise):
    """All 12 information states in the given betting situation."""
    for private_rank in range(3):
        for public_rank in (None, 0, 1, 2):
            betting_round = 1 if public_rank is None else 2
            yield Observation(private_rank, public_rank, betting_round, facing_bet, can_raise)


SITUATIONS = [(False, True), (True, True), (True, False)]


def all_observations():
    for facing_bet, can_raise in SITUATIONS:
        yield from observations(facing_bet, can_raise)


def test_named_instances_have_the_right_parameters():
    assert NAMES == ("TAG", "LAG", "TP", "LP")
    assert STRATEGIES["TAG"] == ThresholdStrategy(tau=0.65, alpha=0.80)
    assert STRATEGIES["LAG"] == ThresholdStrategy(tau=0.40, alpha=0.80)
    assert STRATEGIES["TP"] == ThresholdStrategy(tau=0.65, alpha=0.20)
    assert STRATEGIES["LP"] == ThresholdStrategy(tau=0.40, alpha=0.20)


def test_alpha_0_never_bets_or_raises():
    strategy = ThresholdStrategy(tau=0.0, alpha=0.0)
    rng = np.random.default_rng(0)
    for obs in all_observations():
        for _ in range(50):
            assert strategy(obs, rng) != Action.RAISE


def test_alpha_1_always_bets_or_raises_above_tau():
    rng = np.random.default_rng(0)
    for tau in (0.40, 0.65):
        strategy = ThresholdStrategy(tau=tau, alpha=1.0)
        for obs in all_observations():
            if equity(obs.private_rank, obs.public_rank) >= tau and obs.can_raise:
                for _ in range(50):
                    assert strategy(obs, rng) == Action.RAISE


def test_never_bets_below_tau():
    rng = np.random.default_rng(0)
    for strategy in STRATEGIES.values():
        for obs in observations(facing_bet=False, can_raise=True):
            if equity(obs.private_rank, obs.public_rank) < strategy.tau:
                assert strategy(obs, rng) == Action.CALL


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_always_folds_below_tau_when_facing_a_bet(alpha):
    rng = np.random.default_rng(0)
    for tau in (0.40, 0.65):
        strategy = ThresholdStrategy(tau=tau, alpha=alpha)
        for obs in all_observations():
            if obs.facing_bet and equity(obs.private_rank, obs.public_rank) < tau:
                assert strategy(obs, rng) == Action.FOLD


def test_never_raises_when_raise_is_not_legal():
    strategy = ThresholdStrategy(tau=0.0, alpha=1.0)
    rng = np.random.default_rng(0)
    for obs in observations(facing_bet=True, can_raise=False):
        for _ in range(50):
            assert strategy(obs, rng) == Action.CALL


def test_alpha_is_the_raise_frequency():
    strategy = ThresholdStrategy(tau=0.0, alpha=0.8)
    rng = np.random.default_rng(0)
    obs = Observation(2, None, 1, facing_bet=False, can_raise=True)
    raises = sum(strategy(obs, rng) == Action.RAISE for _ in range(10_000))
    assert raises / 10_000 == pytest.approx(0.8, abs=0.02)


@pytest.mark.parametrize("name0", NAMES)
@pytest.mark.parametrize("name1", NAMES)
def test_strategies_play_each_other(name0, name1):
    rng = np.random.default_rng(0)
    for _ in range(200):
        u0, u1 = play_hand(STRATEGIES[name0], STRATEGIES[name1], rng)
        assert u0 + u1 == 0
        assert abs(u0) <= U_MAX


def test_tag_beats_lp_on_average():
    # Expected payoff of TAG vs LP is +0.471 chips per hand (analysis/tau_pilot.py).
    rng = np.random.default_rng(0)
    u_tag, _ = play_encounter(STRATEGIES["TAG"], STRATEGIES["LP"], 4000, rng)
    assert u_tag == pytest.approx(0.471, abs=0.15)
