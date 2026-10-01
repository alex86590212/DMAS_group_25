"""TAG, LAG, TP and LP.

A strategy continues with a hand when its exact equity reaches `tau`, and when it continues it
bets/raises with probability `alpha`.
"""

from dataclasses import dataclass

import numpy as np

from dmas.leduc.game import Action, Observation
from dmas.strategies.equity import equity


@dataclass(frozen=True)
class ThresholdStrategy:
    tau: float
    alpha: float

    def __call__(self, obs: Observation, rng: np.random.Generator) -> Action:
        e = equity(obs.private_rank, obs.public_rank)

        if not obs.facing_bet:
            if e >= self.tau and rng.random() < self.alpha:
                return Action.RAISE
            return Action.CALL

        if e < self.tau:
            return Action.FOLD
        if obs.can_raise and rng.random() < self.alpha:
            return Action.RAISE
        return Action.CALL


STRATEGY_PARAMS = {
    "TAG": (0.65, 0.80),
    "LAG": (0.40, 0.80),
    "TP": (0.65, 0.20),
    "LP": (0.40, 0.20),
}

STRATEGIES = {name: ThresholdStrategy(tau, alpha) for name, (tau, alpha) in STRATEGY_PARAMS.items()}
NAMES = tuple(STRATEGIES)
