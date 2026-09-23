"""Issue #3 pilot: population dynamics simulated from exact per-pair adoption probabilities.

Each encounter is replaced by a single draw with the exact probability that the focal agent
adopts the comparison agent's strategy, so the population process has the same distribution
as the full model without playing hands.

Run: uv run python analysis/beta_pilot.py --k 10 --beta 0 4 16
"""

import argparse

import numpy as np
from exact import NAMES, copy_table, strategies

CONDITIONS = {
    "balanced": (0.25, 0.25, 0.25, 0.25),
    "tag_majority": (0.55, 0.15, 0.15, 0.15),
    "lag_majority": (0.15, 0.55, 0.15, 0.15),
    "tp_majority": (0.15, 0.15, 0.55, 0.15),
    "lp_majority": (0.15, 0.15, 0.15, 0.55),
}


def simulate(c, composition, n, generations, reps, rng):
    fixed = np.zeros(5)  # TAG, LAG, TP, LP, none
    final = np.zeros(4)
    events = n * generations
    for _ in range(reps):
        pop = np.repeat(np.arange(4), np.round(np.array(composition) * n).astype(int))
        rng.shuffle(pop)
        focal = rng.integers(0, n, events)
        other = rng.integers(0, n - 1, events)
        other += other >= focal
        draws = rng.random(events)
        for f, o, u in zip(focal, other, draws, strict=True):
            sf, so = pop[f], pop[o]
            if sf != so and u < c[sf, so]:
                pop[f] = so
        counts = np.bincount(pop, minlength=4)
        final += counts / n
        fixed[int(np.argmax(counts)) if counts.max() == n else 4] += 1
    return fixed / reps, final / reps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--beta", type=float, nargs="+", default=[0, 4, 16])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    strats = strategies()
    for beta in args.beta:
        c = copy_table(strats, beta, args.k)
        print(f"\nK={args.k} beta={beta:g}")
        print("adoption bias C[j,i] - C[i,j] (row beats column if > 0):")
        print("       " + "".join(f"{s:>8}" for s in NAMES))
        for name, row in zip(NAMES, c.T - c, strict=True):
            print(f"{name:>6} " + "".join(f"{v:8.3f}" for v in row))
        print("fixation share (TAG LAG TP LP none) | mean final share (TAG LAG TP LP)")
        for i, (cond, comp) in enumerate(CONDITIONS.items()):
            rng = np.random.default_rng([args.seed, args.k, int(beta * 1000), i])
            fixed, final = simulate(c, comp, args.n, args.generations, args.reps, rng)
            print(f"  {cond:>13}: {np.round(fixed, 2)} | {np.round(final, 2)}")


if __name__ == "__main__":
    main()
