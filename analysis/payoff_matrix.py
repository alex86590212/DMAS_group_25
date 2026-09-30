"""Issue #7 follow-up: Monte Carlo payoff matrix and histograms of the encounter payoff gap.

For interpretation only; never used for adaptation. Plays real hands with the simulator's
Leduc wrapper, alternating seats, and writes to results/payoff_matrix/.

Run: uv run python analysis/payoff_matrix.py --hands 1000000 --encounters 5000
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dmas.leduc.game import play_hand
from dmas.simulation.runner import play_encounter
from dmas.strategies.strategy import STRATEGY_PARAMS, ThresholdStrategy

NAMES = list(STRATEGY_PARAMS)
POLICIES = [ThresholdStrategy(*STRATEGY_PARAMS[s]) for s in NAMES]
KS = (1, 10, 50)
OUT = Path("results/payoff_matrix")


def hand_stats(args):
    """Mean and standard error of the row strategy's per-hand payoff against the column."""
    i, j, hands, seed = args
    rng = np.random.default_rng(seed)
    payoffs = np.empty(hands)
    for h in range(hands):
        if h % 2 == 0:
            payoffs[h] = play_hand(POLICIES[i], POLICIES[j], rng)[0]
        else:
            payoffs[h] = play_hand(POLICIES[j], POLICIES[i], rng)[1]
    return i, j, payoffs.mean(), payoffs.std(ddof=1) / np.sqrt(hands)


def gap_samples(args):
    """Delta = comparison mean payoff - focal mean payoff over many K-hand encounters."""
    i, j, k, encounters, seed = args
    rng = np.random.default_rng(seed)
    out = np.empty(encounters)
    for e in range(encounters):
        u_focal, u_comparison = play_encounter(POLICIES[i], POLICIES[j], k, rng)
        out[e] = u_comparison - u_focal
    return i, j, k, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1_000_000)
    parser.add_argument("--encounters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ss = np.random.SeedSequence(args.seed)

    pairs = list(combinations_with_replacement(range(4), 2))
    seeds = ss.spawn(len(pairs) * (1 + len(KS)))
    hand_jobs = [(i, j, args.hands, seeds[n]) for n, (i, j) in enumerate(pairs)]
    gap_jobs = [
        (i, j, k, args.encounters, seeds[len(pairs) * (1 + m) + n])
        for m, k in enumerate(KS)
        for n, (i, j) in enumerate(pairs)
        if i != j
    ]

    with ProcessPoolExecutor() as pool:
        hand_res = list(pool.map(hand_stats, hand_jobs))
        gap_res = list(pool.map(gap_samples, gap_jobs))

    mean = np.zeros((4, 4))
    se = np.zeros((4, 4))
    for i, j, m, s in hand_res:
        # Same hands in both directions, so the matrix is exactly antisymmetric.
        mean[i, j], mean[j, i] = m, -m
        se[i, j] = se[j, i] = s
    ci = 1.96 * se
    for name, arr in (("mean", mean), ("ci95", ci)):
        pd.DataFrame(arr, index=NAMES, columns=NAMES).to_csv(OUT / f"payoff_{name}.csv")

    print(f"Expected payoff per hand, row vs column ({args.hands:,} hands per pair):")
    for i in range(4):
        print(NAMES[i].ljust(4), *[f"{mean[i, j]:+.3f}±{ci[i, j]:.3f}" for j in range(4)])

    offdiag = [(i, j) for i, j in pairs if i != j]
    fig, axes = plt.subplots(len(KS), len(offdiag), figsize=(3 * len(offdiag), 2.6 * len(KS)))
    rows = []
    for k_idx, k in enumerate(KS):
        for col, (i, j) in enumerate(offdiag):
            d = next(r[3] for r in gap_res if (r[0], r[1], r[2]) == (i, j, k))
            ax = axes[k_idx, col]
            ax.hist(d, bins=40, color="tab:blue")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{NAMES[i]} vs {NAMES[j]}, K={k}", fontsize=9)
            ax.set_xlabel("Δ = ū_comparison − ū_focal", fontsize=8)
            rows.append(
                {
                    "focal": NAMES[i],
                    "comparison": NAMES[j],
                    "K": k,
                    "mean": d.mean(),
                    "std": d.std(ddof=1),
                    "p_copy_bias_gt0": (d > 0).mean(),
                    "abs_p95": np.percentile(np.abs(d), 95),
                }
            )
    fig.tight_layout()
    fig.savefig(OUT / "delta_histograms.png", dpi=130)
    pd.DataFrame(rows).to_csv(OUT / "delta_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
