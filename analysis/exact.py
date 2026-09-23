"""Exact evaluation of the threshold strategies by enumerating the OpenSpiel Leduc game tree.

Shared by the pilot scripts in this folder. This is an independent reference implementation
used to justify parameter choices (issues #2 and #3) and to validate the simulator; it is
never used for adaptation in the experiments.
"""

import numpy as np
import pyspiel

GAME = pyspiel.load_game("leduc_poker")
U_MAX = int(GAME.max_utility())
NAMES = ["TAG", "LAG", "TP", "LP"]

FOLD, CALL, RAISE = 0, 1, 2


def _showdown(mine, theirs, public):
    if mine == public and theirs != public:
        return 1.0
    if theirs == public and mine != public:
        return 0.0
    if mine != theirs:
        return float(mine > theirs)
    return 0.5


def _equity_table():
    table = {}
    for card in range(3):
        deck = [0, 0, 1, 1, 2, 2]
        deck.remove(card)
        preflop = [_showdown(card, deck[i], deck[j]) for i in range(5) for j in range(5) if i != j]
        table[card, None] = sum(preflop) / len(preflop)
        for public in set(deck):
            rest = deck.copy()
            rest.remove(public)
            table[card, public] = sum(_showdown(card, o, public) for o in rest) / len(rest)
    return table


EQUITY = _equity_table()


def strategies(tau_tight=0.65, tau_loose=0.40, alpha_aggr=0.80, alpha_pass=0.20):
    return {
        "TAG": (tau_tight, alpha_aggr),
        "LAG": (tau_loose, alpha_aggr),
        "TP": (tau_tight, alpha_pass),
        "LP": (tau_loose, alpha_pass),
    }


def action_probs(strategy, equity, legal):
    tau, alpha = strategy
    if FOLD not in legal:
        return {RAISE: alpha, CALL: 1 - alpha} if equity >= tau else {CALL: 1.0}
    if equity < tau:
        return {FOLD: 1.0}
    if RAISE in legal:
        return {RAISE: alpha, CALL: 1 - alpha}
    return {CALL: 1.0}


def payoff_distribution(seat0, seat1):
    """Exact distribution of the seat-0 player's payoff; index u + U_MAX holds P(u)."""
    dist = np.zeros(2 * U_MAX + 1)

    def rec(state, ranks, public, prob):
        if state.is_terminal():
            dist[int(round(state.returns()[0])) + U_MAX] += prob
            return
        if state.is_chance_node():
            for action, p in state.chance_outcomes():
                child = state.child(action)
                if len(ranks) < 2:
                    rec(child, ranks + [action // 2], public, prob * p)
                else:
                    rec(child, ranks, action // 2, prob * p)
            return
        player = state.current_player()
        strategy = seat0 if player == 0 else seat1
        equity = EQUITY[ranks[player], public]
        for action, p in action_probs(strategy, equity, state.legal_actions()).items():
            if p > 0:
                rec(state.child(action), ranks, public, prob * p)

    rec(GAME.new_initial_state(), [], None, 1.0)
    return dist


def _power(dist, n):
    out = np.array([1.0])
    for _ in range(n):
        out = np.convolve(out, dist)
    return out


def encounter_sum_distribution(a, b, k):
    """Distribution of a's summed payoff over k hands vs b with alternating seats.

    Index s + k * U_MAX holds P(sum = s). For k = 1 the seat is chosen at random.
    """
    as_seat0 = payoff_distribution(a, b)
    as_seat1 = payoff_distribution(b, a)[::-1]
    if k == 1:
        return 0.5 * (as_seat0 + as_seat1)
    return np.convolve(_power(as_seat0, (k + 1) // 2), _power(as_seat1, k // 2))


def payoff_matrix(strats):
    """Expected per-hand payoff of row vs column, averaged over both seats."""
    values = np.arange(-U_MAX, U_MAX + 1)
    m = np.zeros((4, 4))
    for i, a in enumerate(NAMES):
        for j, b in enumerate(NAMES):
            if i != j:
                m[i, j] = encounter_sum_distribution(strats[a], strats[b], 1) @ values
    return m


def copy_table(strats, beta, k):
    """C[i, j] = exact probability that a focal agent using i adopts j after one encounter."""
    c = np.zeros((4, 4))
    for i, a in enumerate(NAMES):
        for j, b in enumerate(NAMES):
            if i == j:
                continue
            dist = encounter_sum_distribution(strats[b], strats[a], k)
            mean_j = (np.arange(len(dist)) - k * U_MAX) / k
            d = 2 * mean_j / (2 * U_MAX)  # zero-sum: delta = u_j - u_i = 2 * u_j
            c[i, j] = dist @ (0.5 * (1 + np.tanh(beta * d / 2)))
    return c
