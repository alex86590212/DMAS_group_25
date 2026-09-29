import numpy as np

from dmas.leduc.game import Policy, play_hand


def play_encounter(
    focal: Policy,
    comparison: Policy,
    k: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Play k hands between two fixed strategies.

    Seats alternate across hands because OpenSpiel's player 0
    always acts first.

    Returns:
        Mean payoff of the focal strategy and comparison strategy.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    focal_total = 0.0
    comparison_total = 0.0

    for hand in range(k):
        if hand % 2 == 0:
            # Focal is player 0.
            u_focal, u_comparison = play_hand(
                focal,
                comparison,
                rng,
            )
        else:
            # Comparison is player 0.
            u_comparison, u_focal = play_hand(
                comparison,
                focal,
                rng,
            )

        focal_total += u_focal
        comparison_total += u_comparison

    mean_focal = focal_total / k
    mean_comparison = comparison_total / k

    return mean_focal, mean_comparison
