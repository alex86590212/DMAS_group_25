"""Issue #2 pilot: payoff matrices for candidate thresholds and a search for cyclic dominance.

Run: uv run python analysis/tau_pilot.py
"""

import itertools

from exact import EQUITY, NAMES, payoff_matrix, strategies

TIE = 0.02
CANDIDATES = [(0.60, 0.40), (0.65, 0.40), (0.65, 0.25), (0.65, 0.10), (0.60, 0.25)]
TAU_GRID = [0.10, 0.25, 0.40, 0.60, 0.65, 0.80]  # one representative per behaviour class
ALPHA_AGGR = [0.6, 0.8, 1.0]
ALPHA_PASS = [0.0, 0.2, 0.4]


def print_matrix(m):
    print("       " + "".join(f"{n:>8}" for n in NAMES))
    for name, row in zip(NAMES, m, strict=True):
        print(f"{name:>6} " + "".join(f"{v:8.3f}" for v in row))


def classify(m):
    dominant = [NAMES[i] for i in range(4) if all(m[i, j] > TIE for j in range(4) if j != i)]
    cycles = [
        c
        for c in itertools.permutations(range(4), 3)
        if c[0] == min(c) and all(m[c[x], c[(x + 1) % 3]] > TIE for x in range(3))
    ]
    ties = [
        (NAMES[i], NAMES[j]) for i, j in itertools.combinations(range(4), 2) if abs(m[i, j]) <= TIE
    ]
    return dominant, cycles, ties


def main():
    print("Equity values e(I) by (private rank, public rank); 0=J 1=Q 2=K, None=pre-flop")
    for key, value in sorted(EQUITY.items(), key=lambda kv: (kv[0][0], kv[0][1] is not None)):
        print(f"  {key}: {value:.3f}")

    for tight, loose in CANDIDATES:
        m = payoff_matrix(strategies(tight, loose))
        dominant, cycles, ties = classify(m)
        print(f"\n(tau_T, tau_L) = ({tight}, {loose}); row vs column, chips per hand")
        print_matrix(m)
        print(f"dominant={dominant} cycles={len(cycles)} ties={ties}")

    counts = {"cycle": 0, "dominant": 0, "neither": 0}
    for tight, loose in itertools.product(TAU_GRID, TAU_GRID):
        if tight <= loose:
            continue
        for aa, ap in itertools.product(ALPHA_AGGR, ALPHA_PASS):
            dominant, cycles, _ = classify(payoff_matrix(strategies(tight, loose, aa, ap)))
            key = "cycle" if cycles else "dominant" if dominant else "neither"
            counts[key] += 1
    total = sum(counts.values())
    print(f"\nGrid search over {total} (tau, alpha) settings: {counts}")


if __name__ == "__main__":
    main()
