from collections.abc import Sequence

import numpy as np

from dmas.agents.agent import Agent


class Population:
    def __init__(self, agents: list[Agent], strategies: Sequence[str]):
        self.agents = agents
        self.strategies = tuple(strategies)

    @classmethod
    def from_composition(
        cls,
        counts: Sequence[int],
        strategies: Sequence[str],
        rng: np.random.Generator,
    ) -> "Population":
        """Build a shuffled population with exactly counts[i] agents of strategies[i]."""
        if len(counts) != len(strategies):
            raise ValueError("counts and strategies must have the same length")
        if any(c < 0 for c in counts):
            raise ValueError("counts must be non-negative")

        labels = [s for s, c in zip(strategies, counts, strict=True) for _ in range(c)]
        rng.shuffle(labels)
        agents = [Agent(id=i, strategy=s) for i, s in enumerate(labels)]
        return cls(agents, strategies)

    def __len__(self) -> int:
        return len(self.agents)

    def sample_pair(self, rng: np.random.Generator) -> tuple[Agent, Agent]:
        """Return two distinct agents: (focal, comparison)."""
        if len(self.agents) < 2:
            raise ValueError("need at least two agents")
        i, j = rng.choice(len(self.agents), size=2, replace=False)
        return self.agents[i], self.agents[j]

    def shares(self) -> dict[str, float]:
        """Fraction of agents per strategy, in strategy order."""
        n = len(self.agents)
        counts = dict.fromkeys(self.strategies, 0)
        for agent in self.agents:
            counts[agent.strategy] += 1
        return {s: c / n for s, c in counts.items()}
