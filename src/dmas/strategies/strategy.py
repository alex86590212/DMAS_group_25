"""Threshold strategy: bet/raise with probability alpha when equity >= tau, else check/fold."""

from dataclasses import dataclass

import numpy as np

from dmas.leduc.game import Action, Observation
from dmas.strategies.equity import equity


@dataclass(frozen=True)
class ThresholdStrategy:
    """A Policy: bets/raises above tau, folds below tau when facing a bet."""

    tau: float
    alpha: float

    def __call__(self, obs: Observation, rng: np.random.Generator) -> Action:
        hand_equity = equity(obs.private_rank, obs.public_rank)

        if not obs.facing_bet:
            if hand_equity >= self.tau and rng.random() < self.alpha:
                return Action.RAISE
            return Action.CALL

        if hand_equity < self.tau:
            return Action.FOLD
        if obs.can_raise and rng.random() < self.alpha:
            return Action.RAISE
        return Action.CALL


# Single source of truth for the four named strategies: (tau, alpha).
STRATEGY_PARAMS = {
    "TAG": (0.65, 0.80),
    "LAG": (0.40, 0.80),
    "TP": (0.65, 0.20),
    "LP": (0.40, 0.20),
}

TAG, LAG, TP, LP = (ThresholdStrategy(tau, alpha) for tau, alpha in STRATEGY_PARAMS.values())
