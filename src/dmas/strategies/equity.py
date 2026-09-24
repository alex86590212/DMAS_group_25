"""Exact hand equity for OpenSpiel Leduc Poker."""

import pyspiel

_GAME = pyspiel.load_game("leduc_poker")

# OpenSpiel Leduc action:
# 1 = check when no bet is outstanding, otherwise call.
_CHECK_CALL = 1

# OpenSpiel card ids:
# 0,1 = J
# 2,3 = Q
# 4,5 = K
_RANKS = range(3)


def _rank(card: int) -> int:
    """Return the rank of an OpenSpiel Leduc card."""
    return card // 2


def _outcome_from_return(player_return: float) -> float:
    """Convert OpenSpiel payoff to win/tie/loss equity."""
    if player_return > 0:
        return 1.0
    if player_return < 0:
        return 0.0
    return 0.5


def _exact_equity(
    private_rank: int,
    public_rank: int | None,
) -> float:
    """Compute equity by enumerating OpenSpiel chance outcomes.

    Player 0 is treated as the focal player.

    If public_rank is None, all possible public cards are included,
    giving pre-flop equity.

    Otherwise, the calculation is conditioned on that public rank.
    """
    total_equity = 0.0
    total_probability = 0.0

    initial_state = _GAME.new_initial_state()

    # First chance node: player 0 private card.
    for hero_card, hero_prob in initial_state.chance_outcomes():
        if _rank(hero_card) != private_rank:
            continue

        hero_state = initial_state.clone()
        hero_state.apply_action(hero_card)

        # Second chance node: player 1 private card.
        for opponent_card, opponent_prob in hero_state.chance_outcomes():
            opponent_state = hero_state.clone()
            opponent_state.apply_action(opponent_card)

            # Check/check through the first betting round.
            opponent_state.apply_action(_CHECK_CALL)
            opponent_state.apply_action(_CHECK_CALL)

            # Third chance node: public card.
            for public_card, public_prob in opponent_state.chance_outcomes():
                if (
                    public_rank is not None
                    and _rank(public_card) != public_rank
                ):
                    continue

                showdown_state = opponent_state.clone()
                showdown_state.apply_action(public_card)

                # Check/check through the second betting round.
                showdown_state.apply_action(_CHECK_CALL)
                showdown_state.apply_action(_CHECK_CALL)

                result = _outcome_from_return(
                    showdown_state.returns()[0]
                )

                probability = (
                    hero_prob
                    * opponent_prob
                    * public_prob
                )

                total_equity += probability * result
                total_probability += probability

    if total_probability == 0:
        raise ValueError(
            f"Impossible information state: "
            f"private_rank={private_rank}, "
            f"public_rank={public_rank}"
        )

    # We conditioned on the requested information state, so normalize
    # by the total probability of all deals consistent with it.
    return total_equity / total_probability


# Precompute all 12 information states once.
EQUITY_TABLE = {
    (private_rank, public_rank): _exact_equity(
        private_rank,
        public_rank,
    )
    for private_rank in _RANKS
    for public_rank in (None, *_RANKS)
}


def equity(
    private_rank: int,
    public_rank: int | None = None,
) -> float:
    """Return exact equity for a Leduc information state.

    Args:
        private_rank:
            0 = J, 1 = Q, 2 = K.
        public_rank:
            0 = J, 1 = Q, 2 = K, or None pre-flop.
    """
    if private_rank not in _RANKS:
        raise ValueError(f"Invalid private rank: {private_rank}")

    if public_rank is not None and public_rank not in _RANKS:
        raise ValueError(f"Invalid public rank: {public_rank}")

    return EQUITY_TABLE[(private_rank, public_rank)]



def main() -> None:
    """Print the precomputed equity table."""
    rank_names = {0: "J", 1: "Q", 2: "K"}

    print(
        f"{'Private':<8}"
        f"{'Pre-flop':>10}"
        f"{'Public J':>10}"
        f"{'Public Q':>10}"
        f"{'Public K':>10}"
    )

    for private_rank in _RANKS:
        print(
            f"{rank_names[private_rank]:<8}"
            f"{EQUITY_TABLE[(private_rank, None)]:>10.3f}"
            f"{EQUITY_TABLE[(private_rank, 0)]:>10.3f}"
            f"{EQUITY_TABLE[(private_rank, 1)]:>10.3f}"
            f"{EQUITY_TABLE[(private_rank, 2)]:>10.3f}"
        )


if __name__ == "__main__":
    main()