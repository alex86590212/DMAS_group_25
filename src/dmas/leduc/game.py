"""Thin wrapper around OpenSpiel's leduc_poker.

Chance events are sampled with the caller's numpy Generator so that hands are reproducible
from our own seeds. Policies see a small Observation and return an Action.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pyspiel

GAME = pyspiel.load_game("leduc_poker")
U_MAX = int(GAME.max_utility())
RANKS = ("J", "Q", "K")


class Action(IntEnum):
    """OpenSpiel action ids. CALL is a check and RAISE a bet when no bet is outstanding."""

    FOLD = 0
    CALL = 1
    RAISE = 2


@dataclass(frozen=True)
class Observation:
    private_rank: int
    public_rank: int | None
    betting_round: int
    facing_bet: bool
    can_raise: bool


Policy = Callable[[Observation, np.random.Generator], Action]


def rank(card: int) -> int:
    return card // 2


def play_hand(
    policy0: Policy,
    policy1: Policy,
    rng: np.random.Generator,
    cards: tuple[int, int, int] | None = None,
) -> tuple[float, float]:
    """Play one hand and return (u0, u1). Seat 0 acts first in both betting rounds.

    `cards` fixes the deal as OpenSpiel card ids (private0, private1, public); otherwise cards
    are drawn with `rng`.
    """
    state = GAME.new_initial_state()
    policies = (policy0, policy1)
    dealt: list[int] = []

    while not state.is_terminal():
        if state.is_chance_node():
            if cards is not None:
                card = cards[len(dealt)]
            else:
                outcomes, probs = zip(*state.chance_outcomes(), strict=True)
                card = outcomes[rng.choice(len(outcomes), p=probs)]
            dealt.append(card)
            state.apply_action(card)
            continue

        player = state.current_player()
        legal = state.legal_actions()
        public = dealt[2] if len(dealt) == 3 else None
        obs = Observation(
            private_rank=rank(dealt[player]),
            public_rank=None if public is None else rank(public),
            betting_round=1 if public is None else 2,
            facing_bet=Action.FOLD in legal,
            can_raise=Action.RAISE in legal,
        )
        action = policies[player](obs, rng)
        if action not in legal:
            raise ValueError(f"illegal action {action!r} for {obs}")
        state.apply_action(int(action))

    u0, u1 = state.returns()
    return u0, u1
