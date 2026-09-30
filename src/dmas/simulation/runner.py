import zlib
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from dmas.evolution.adaptation import DEFAULT_BETA, DEFAULT_NORMALISER, adapt
from dmas.evolution.population import Population
from dmas.leduc.game import Policy, play_hand
from dmas.strategies.strategy import STRATEGY_PARAMS, ThresholdStrategy


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


INITIAL_CONDITIONS = {
    "balanced": (0.25, 0.25, 0.25, 0.25),
    "tag_majority": (0.55, 0.15, 0.15, 0.15),
    "lag_majority": (0.15, 0.55, 0.15, 0.15),
    "tp_majority": (0.15, 0.15, 0.55, 0.15),
    "lp_majority": (0.15, 0.15, 0.15, 0.55),
}


@dataclass(frozen=True)
class SimulationConfig:
    """Every simulation parameter. Everything except the initial condition is shared."""

    n: int = 100
    generations: int = 200
    replicates: int = 50
    k: int = 10
    beta: float = DEFAULT_BETA
    normaliser: float = DEFAULT_NORMALISER
    initial_conditions: Mapping[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(INITIAL_CONDITIONS)
    )
    # (tau, alpha) per strategy; key order is the strategy order of initial_conditions.
    strategy_params: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(STRATEGY_PARAMS)
    )
    base_seed: int = 0
    # Fixation is absorbing (no mutation), so by default a run stops at fixation and
    # the remaining generations are filled with the fixed state.
    stop_at_fixation: bool = True

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(self.strategy_params)

    def counts(self, condition: str) -> tuple[int, ...]:
        """Exact agent counts for a condition; shares must give whole agents summing to n."""
        shares = self.initial_conditions[condition]
        counts = tuple(round(s * self.n) for s in shares)
        if sum(counts) != self.n or any(
            abs(c - s * self.n) > 1e-9 for c, s in zip(counts, shares, strict=True)
        ):
            raise ValueError(
                f"condition {condition!r} does not split into whole agents for n={self.n}"
            )
        return counts


@dataclass(frozen=True)
class ReplicateResult:
    condition: str
    replicate: int
    seed: tuple[int, ...]
    strategies: tuple[str, ...]
    # shares[g] is the composition after generation g (row 0 is the initial state).
    shares: np.ndarray  # (generations + 1, n_strategies)
    switches: np.ndarray  # (generations,) strategy changes per generation
    transitions: np.ndarray  # (generations, n_strategies, n_strategies) counts, [g, old, new]
    fixed_strategy: str | None
    fixation_generation: int | None


def replicate_seed(condition: str, replicate: int, base_seed: int = 0) -> np.random.SeedSequence:
    """Seed derived from (condition, replicate); independent of ordering and worker count."""
    return np.random.SeedSequence([base_seed, zlib.crc32(condition.encode()), replicate])


def _is_fixed(row: np.ndarray) -> bool:
    return bool(np.isclose(row.max(), 1.0))


def _shares_row(population: Population) -> np.ndarray:
    return np.array(list(population.shares().values()))


def run_replicate(condition: str, replicate: int, config: SimulationConfig) -> ReplicateResult:
    seed = replicate_seed(condition, replicate, config.base_seed)
    rng = np.random.default_rng(seed)

    strategies = config.strategies
    index = {s: i for i, s in enumerate(strategies)}
    policies = {s: ThresholdStrategy(*p) for s, p in config.strategy_params.items()}
    population = Population.from_composition(config.counts(condition), strategies, rng)

    shares = np.empty((config.generations + 1, len(strategies)))
    switches = np.zeros(config.generations, dtype=int)
    transitions = np.zeros((config.generations, len(strategies), len(strategies)), dtype=int)
    shares[0] = _shares_row(population)

    fixation_generation = 0 if _is_fixed(shares[0]) else None
    for g in range(1, config.generations + 1):
        if fixation_generation is not None and config.stop_at_fixation:
            shares[g:] = shares[g - 1]
            break
        for _ in range(config.n):
            focal, comparison = population.sample_pair(rng)
            if focal.strategy == comparison.strategy:
                continue  # cannot cause a switch, so skip playing the hands
            u_focal, u_comparison = play_encounter(
                policies[focal.strategy], policies[comparison.strategy], config.k, rng
            )
            result = adapt(
                focal, comparison, u_focal, u_comparison, rng, config.beta, config.normaliser
            )
            if result.switched:
                switches[g - 1] += 1
                transitions[g - 1, index[result.old], index[result.new]] += 1
        shares[g] = _shares_row(population)
        if fixation_generation is None and _is_fixed(shares[g]):
            fixation_generation = g

    fixed = strategies[int(shares[-1].argmax())] if _is_fixed(shares[-1]) else None
    return ReplicateResult(
        condition=condition,
        replicate=replicate,
        seed=tuple(seed.entropy),
        strategies=strategies,
        shares=shares,
        switches=switches,
        transitions=transitions,
        fixed_strategy=fixed,
        fixation_generation=fixation_generation,
    )


def _run_one(args: tuple[str, int, SimulationConfig]) -> ReplicateResult:
    return run_replicate(*args)


def run_condition(
    condition: str, config: SimulationConfig, workers: int | None = None
) -> list[ReplicateResult]:
    """Run all replicates of one condition; results are ordered by replicate index."""
    jobs = [(condition, r, config) for r in range(config.replicates)]
    if workers == 1:
        return [_run_one(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs))
