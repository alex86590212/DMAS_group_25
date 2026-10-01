"""Neutral-drift null model for Issue #11.

At beta=0, the Fermi adoption probability is exactly 0.5 for every
pair of different strategies, so payoffs and K do not affect adaptation.

Each replicate starts from one of the five experimental compositions and
runs until one strategy reaches fixation.

Results are saved under results/neutral_drift/.

Run:
    uv run python analysis/neutral_drift.py --reps 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = ("TAG", "LAG", "TP", "LP")

CONDITIONS = {
    "balanced": (0.25, 0.25, 0.25, 0.25),
    "tag_majority": (0.55, 0.15, 0.15, 0.15),
    "lag_majority": (0.15, 0.55, 0.15, 0.15),
    "tp_majority": (0.15, 0.15, 0.55, 0.15),
    "lp_majority": (0.15, 0.15, 0.15, 0.55),
}

RESULTS_DIR = Path("results/neutral_drift")


def simulate_replicate(
    composition: tuple[float, ...],
    n: int,
    rng: np.random.Generator,
) -> tuple[str, int]:
    """Simulate one neutral-drift population until fixation.

    At beta=0, a focal agent copies a different comparison strategy
    with probability 0.5.

    Returns:
        fixed_strategy: strategy that eventually reaches fixation.
        events: number of focal-agent update events until fixation.
    """
    counts = np.round(np.array(composition) * n).astype(int)

    if counts.sum() != n:
        raise ValueError("Initial composition must contain exactly n agents")

    population = np.repeat(np.arange(len(STRATEGIES)), counts)
    rng.shuffle(population)

    events = 0

    while True:
        counts = np.bincount(population, minlength=len(STRATEGIES))

        if counts.max() == n:
            fixed_index = int(np.argmax(counts))
            return STRATEGIES[fixed_index], events

        # Choose two distinct agents.
        focal = int(rng.integers(0, n))
        comparison = int(rng.integers(0, n - 1))

        if comparison >= focal:
            comparison += 1

        # Agents with the same strategy cannot change.
        if population[focal] == population[comparison]:
            events += 1
            continue

        # beta = 0 -> Fermi probability = 0.5.
        if rng.random() < 0.5:
            population[focal] = population[comparison]

        events += 1


def run_condition(
    condition: str,
    composition: tuple[float, ...],
    n: int,
    reps: int,
    seed: int,
) -> list[dict]:
    """Run all neutral-drift replicates for one condition."""
    rng = np.random.default_rng(seed)

    rows = []

    for replicate in range(reps):
        fixed_strategy, events = simulate_replicate(
            composition,
            n,
            rng,
        )

        rows.append(
            {
                "condition": condition,
                "replicate": replicate,
                "fixed_strategy": fixed_strategy,
                "events_to_fixation": events,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for condition_index, (condition, composition) in enumerate(
        CONDITIONS.items()
    ):
        rows = run_condition(
            condition=condition,
            composition=composition,
            n=args.n,
            reps=args.reps,
            seed=args.seed + condition_index,
        )

        all_rows.extend(rows)

    replicate_df = pd.DataFrame(all_rows)

    replicate_df.to_csv(
        RESULTS_DIR / "replicates.csv",
        index=False,
    )

    # Calculate fixation frequencies.
    summary_rows = []

    for condition, composition in CONDITIONS.items():
        condition_df = replicate_df[
            replicate_df["condition"] == condition
        ]

        for strategy, initial_share in zip(
            STRATEGIES,
            composition,
            strict=True,
        ):
            observed = (
                condition_df["fixed_strategy"] == strategy
            ).mean()

            summary_rows.append(
                {
                    "condition": condition,
                    "strategy": strategy,
                    "initial_share": initial_share,
                    "fixation_frequency": observed,
                    "difference": observed - initial_share,
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        RESULTS_DIR / "fixation_summary.csv",
        index=False,
    )

    metadata = {
        "model": "neutral_drift",
        "beta": 0,
        "adoption_probability": 0.5,
        "n": args.n,
        "replicates": args.reps,
        "seed": args.seed,
        "strategies": STRATEGIES,
        "conditions": CONDITIONS,
        "stopping_rule": "run until one strategy reaches fixation",
    }

    (RESULTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nNeutral drift (beta=0)")
    print(f"N={args.n}, replicates={args.reps}")
    print("Fixation frequencies vs. initial shares:\n")

    for condition in CONDITIONS:
        condition_summary = summary_df[
            summary_df["condition"] == condition
        ]

        print(condition)

        for _, row in condition_summary.iterrows():
            print(
                f"  {row['strategy']:>3}: "
                f"initial={row['initial_share']:.2f}, "
                f"fixation={row['fixation_frequency']:.3f}, "
                f"difference={row['difference']:+.3f}"
            )

        print()

    print(f"Saved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()