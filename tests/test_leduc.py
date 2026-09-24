import numpy as np
import pytest

from dmas.leduc.game import U_MAX, Action, Observation, play_hand, rank

J, Q, K = 0, 2, 4  # one OpenSpiel card id per rank; J2, Q2, K2 are the second copies
J2, Q2, K2 = 1, 3, 5


def check_call(obs, rng):
    return Action.CALL


def always_raise(obs, rng):
    return Action.RAISE if obs.can_raise else Action.CALL


def always_fold(obs, rng):
    return Action.FOLD if obs.facing_bet else Action.CALL


def recording(policy, log, seat):
    def wrapped(obs, rng):
        log.append((seat, obs))
        return policy(obs, rng)

    return wrapped


def play(p0, p1, cards):
    return play_hand(p0, p1, np.random.default_rng(0), cards=cards)


def test_u_max_is_13():
    assert U_MAX == 13


def test_rank_mapping():
    assert [rank(c) for c in range(6)] == [0, 0, 1, 1, 2, 2]


def test_higher_card_wins_showdown():
    assert play(check_call, check_call, (K, J, Q)) == (1.0, -1.0)
    assert play(check_call, check_call, (J, K, Q)) == (-1.0, 1.0)


def test_pair_beats_high_card():
    assert play(check_call, check_call, (J, K, J2)) == (1.0, -1.0)


def test_equal_ranks_split_pot():
    assert play(check_call, check_call, (Q, Q2, K)) == (0.0, 0.0)


def test_fold_loses_contribution():
    # seat 0 bets 2, seat 1 folds and loses only its ante
    assert play(always_raise, always_fold, (J, K, Q)) == (1.0, -1.0)


def test_max_payoff_with_raise_cap_reached_in_both_rounds():
    assert play(always_raise, always_raise, (K, J, Q)) == (U_MAX, -U_MAX)


def test_raise_cap_is_two_per_round():
    log = []
    play(recording(always_raise, log, 0), recording(always_raise, log, 1), (K, J, Q))
    round1 = [(seat, obs) for seat, obs in log if obs.betting_round == 1]
    assert [(seat, obs.facing_bet, obs.can_raise) for seat, obs in round1] == [
        (0, False, True),
        (1, True, True),
        (0, True, False),
    ]


def test_player_0_acts_first_in_both_rounds():
    log = []
    play(recording(check_call, log, 0), recording(check_call, log, 1), (K, J, Q))
    assert [(seat, obs.betting_round) for seat, obs in log] == [(0, 1), (1, 1), (0, 2), (1, 2)]


def test_observations():
    log = []
    play(recording(check_call, log, 0), recording(check_call, log, 1), (K, J, Q))
    assert log[0] == (0, Observation(2, None, 1, facing_bet=False, can_raise=True))
    assert log[1] == (1, Observation(0, None, 1, facing_bet=False, can_raise=True))
    assert log[2] == (0, Observation(2, 1, 2, facing_bet=False, can_raise=True))


def test_zero_sum_on_random_hands():
    rng = np.random.default_rng(1)
    for _ in range(200):
        u0, u1 = play_hand(always_raise, check_call, rng)
        assert u0 + u1 == 0
        assert abs(u0) <= U_MAX


def test_same_seed_same_hands():
    def hands(seed):
        rng = np.random.default_rng(seed)
        log = []
        for _ in range(50):
            play_hand(recording(check_call, log, 0), recording(check_call, log, 1), rng)
        return log

    first_run, second_run, other_seed = hands(7), hands(7), hands(8)
    assert first_run == second_run
    assert first_run != other_seed


def test_illegal_action_raises():
    with pytest.raises(ValueError):
        play(lambda obs, rng: Action.FOLD, check_call, (K, J, Q))
