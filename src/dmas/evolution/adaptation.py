import math
from dataclasses import dataclass

import numpy as np

from dmas.agents.agent import Agent
from dmas.leduc.game import U_MAX

DEFAULT_BETA = 4.0
DEFAULT_NORMALISER = 2 * U_MAX


@dataclass(frozen=True)
class AdaptationResult:
    switched: bool
    old: str
    new: str


def copy_probability(
    focal_payoff: float,
    comparison_payoff: float,
    beta: float = DEFAULT_BETA,
    normaliser: float = DEFAULT_NORMALISER,
) -> float:
    """Fermi rule: 1 / (1 + exp(-beta * d)) with d = (u_comparison - u_focal) / normaliser."""
    d = (comparison_payoff - focal_payoff) / normaliser
    x = -beta * d
    # Avoid OverflowError in math.exp for extreme beta.
    if x > 700:
        return 0.0
    return 1.0 / (1.0 + math.exp(x))


def adapt(
    focal: Agent,
    comparison: Agent,
    focal_payoff: float,
    comparison_payoff: float,
    rng: np.random.Generator,
    beta: float = DEFAULT_BETA,
    normaliser: float = DEFAULT_NORMALISER,
) -> AdaptationResult:
    """Let the focal agent copy the comparison agent's strategy with the Fermi probability.

    Only the focal agent changes. Copying an identical strategy is never a switch.
    """
    old = focal.strategy
    if old == comparison.strategy:
        return AdaptationResult(False, old, old)

    p = copy_probability(focal_payoff, comparison_payoff, beta, normaliser)
    if rng.random() < p:
        focal.strategy = comparison.strategy
        return AdaptationResult(True, old, focal.strategy)
    return AdaptationResult(False, old, old)
