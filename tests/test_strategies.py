import numpy as np

from dmas.leduc.game import Action, Observation, play_hand
from dmas.strategies.strategy import LAG, LP, TAG, TP, ThresholdStrategy


def obs(private_rank, public_rank, *, facing_bet, can_raise, betting_round=1):
    return Observation(
        private_rank=private_rank,
        public_rank=public_rank,
        betting_round=betting_round,
        facing_bet=facing_bet,
        can_raise=can_raise,
    )


def test_alpha_zero_never_bets_or_raises():
    strategy = ThresholdStrategy(tau=0.40, alpha=0.0)
    rng = np.random.default_rng(0)
    no_bet = obs(2, None, facing_bet=False, can_raise=True)  # K pre-flop, equity 0.70 >= tau
    facing_bet = obs(2, None, facing_bet=True, can_raise=True)
    assert {strategy(no_bet, rng) for _ in range(50)} == {Action.CALL}
    assert {strategy(facing_bet, rng) for _ in range(50)} == {Action.CALL}


def test_alpha_one_always_bets_or_raises_above_tau():
    strategy = ThresholdStrategy(tau=0.40, alpha=1.0)
    rng = np.random.default_rng(0)
    no_bet = obs(2, None, facing_bet=False, can_raise=True)  # K pre-flop, equity 0.70 >= tau
    facing_bet = obs(2, None, facing_bet=True, can_raise=True)
    assert {strategy(no_bet, rng) for _ in range(50)} == {Action.RAISE}
    assert {strategy(facing_bet, rng) for _ in range(50)} == {Action.RAISE}


def test_always_folds_below_tau_when_facing_bet():
    strategy = ThresholdStrategy(tau=0.65, alpha=1.0)
    rng = np.random.default_rng(0)
    weak = obs(0, 1, facing_bet=True, can_raise=True)  # J with public Q, equity 0.125 < tau
    assert {strategy(weak, rng) for _ in range(50)} == {Action.FOLD}


def test_never_raises_when_raise_not_legal():
    strategy = ThresholdStrategy(tau=0.40, alpha=1.0)
    rng = np.random.default_rng(0)
    cannot_raise = obs(2, None, facing_bet=True, can_raise=False)  # K pre-flop, equity 0.70 >= tau
    assert {strategy(cannot_raise, rng) for _ in range(50)} == {Action.CALL}


def test_named_instances_have_correct_parameters():
    assert (TAG.tau, TAG.alpha) == (0.65, 0.80)
    assert (LAG.tau, LAG.alpha) == (0.40, 0.80)
    assert (TP.tau, TP.alpha) == (0.65, 0.20)
    assert (LP.tau, LP.alpha) == (0.40, 0.20)


def test_named_strategies_play_each_other_through_wrapper():
    rng = np.random.default_rng(0)
    strategies = [TAG, LAG, TP, LP]
    for s0 in strategies:
        for s1 in strategies:
            for _ in range(5):
                u0, u1 = play_hand(s0, s1, rng)
                assert u0 + u1 == 0
