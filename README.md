# Multi-Agent Strategy Evolution in Leduc Poker

## Goal

This project studies how the **initial composition of a population of autonomous poker agents affects how strategy use evolves over time**.

### Research Question

> How does the initial composition of heterogeneous autonomous agents affect the evolution of strategy use through repeated interaction in Leduc Poker?

The project is a **multi-agent simulation**, not a machine-learning task.

The poker strategies themselves remain fixed. Agents adapt by switching between strategies after interacting with other agents.

---

## Model

The simulation contains `N = 100` agents.

Each agent currently uses one of four strategies:

- `TAG` = Tight-Aggressive
- `LAG` = Loose-Aggressive
- `TP` = Tight-Passive
- `LP` = Loose-Passive

High-level process:

```text
initialize population
        ↓
randomly select two agents
        ↓
play several Leduc Poker hands
        ↓
compare realized payoffs
        ↓
focal agent may copy opponent strategy
        ↓
population composition changes
        ↓
repeat
```

Leduc Poker remains a normal **1-v-1 game**. The multi-agent system is the larger population from which players are repeatedly selected.

---

## Leduc Poker

We use OpenSpiel's `leduc_poker` (`pyspiel.load_game("leduc_poker")`, open_spiel 2.0.2) with default parameters: a two-player game with:

- 6-card deck
- 3 ranks
- 2 copies of each rank
- 1 private card per player
- 1 public card
- 2 betting rounds
- stochastic card dealing
- zero-sum payoff

Agents adapt using the **actual realized results of played games**.

Do not use a fixed expected-payoff matrix for adaptation.

### Game Rules

Verified against OpenSpiel 2.0.2:

| Rule | Value |
|---|---|
| Ante | 1 chip per player |
| Bet / raise size | 2 in round 1, 4 in round 2 |
| Raise cap | 2 per round (the first bet counts as one) |
| First to act | player 0, in both rounds |
| Cards | `0`–`5`, rank = `card // 2` (`0,1` = J, `2,3` = Q, `4,5` = K) |
| Showdown | pair with the public card wins, otherwise higher rank wins, equal ranks split the pot |
| Payoff | chips won or lost, zero-sum (`state.returns()`) |
| `U_MAX` | 13 (`game.max_utility()`): ante 1 + 2 raises of 2 + 2 raises of 4 |

Actions:

| Id | Name | Meaning |
|---:|---|---|
| 0 | Fold | only legal when facing a bet |
| 1 | Call | check when no bet is outstanding, otherwise call |
| 2 | Raise | bet when no bet is outstanding, otherwise raise |

Because player 0 always acts first, seats must alternate between the two agents across the `K` hands of an encounter.

---

## Strategies

Each strategy is defined by:

- `tau`: continuation threshold
- `alpha`: aggression probability

| Strategy | tau | alpha |
|---|---:|---:|
| TAG | 0.65 | 0.80 |
| LAG | 0.40 | 0.80 |
| TP | 0.65 | 0.20 |
| LP | 0.40 | 0.20 |

### Hand Strength

For information state `I`:

```math
e(I) = P(\text{win} \mid I) + \frac{1}{2}P(\text{tie} \mid I)
```

Calculate this exactly by enumerating possible hidden cards.

Leduc has only three ranks, so `e(I)` takes only six values:

| Private card | Pre-flop | Post-flop, public J | Post-flop, public Q | Post-flop, public K |
|---|---:|---:|---:|---:|
| J | 0.30 | 1.0 | 0.125 | 0.125 |
| Q | 0.50 | 0.125 | 1.0 | 0.625 |
| K | 0.70 | 0.625 | 0.625 | 1.0 |

A threshold `tau` only changes behaviour when it crosses one of these values. The chosen thresholds give:

| | Pre-flop continues with | Post-flop continues with |
|---|---|---|
| Tight (`tau = 0.65`) | K | pair |
| Loose (`tau = 0.40`) | K, Q | pair, K-high, Q-high under a K |

Rejected alternatives:

- `(0.60, 0.40)`: tight and loose differ only with a Q before the flop.
- `(0.65, 0.25)`: loose plays every hand pre-flop, and LAG vs LP becomes an exact tie, so aggression has no effect between loose strategies.
- `tau_L <= 0.125`: loose never folds.
- `tau_T > 0.70`: tight never continues pre-flop.

Exact expected payoff per hand (row vs column, averaged over both seats) with the chosen parameters:

| | TAG | LAG | TP | LP |
|---|---:|---:|---:|---:|
| TAG | 0 | +0.150 | +0.080 | +0.471 |
| LAG | -0.150 | 0 | +0.028 | +0.166 |
| TP | -0.080 | -0.028 | 0 | +0.137 |
| LP | -0.471 | -0.166 | -0.137 | 0 |

This gives a strict dominance order, TAG > LAG > TP > LP. No tested combination of `tau` and `alpha` produced a cycle (rock-paper-scissors).

### Decision Rule

When no bet is outstanding:

```python
if equity >= tau:
    if rng.random() < alpha:
        bet()
    else:
        check()
else:
    check()
```

When facing a bet:

```python
if equity < tau:
    fold()
else:
    if raise_is_legal() and rng.random() < alpha:
        raise_bet()
    else:
        call()
```

The parameters of TAG, LAG, TP, and LP never change during a simulation.

---

## Initial Population Conditions

Strategy order:

```text
(TAG, LAG, TP, LP)
```

Use five initial conditions:

```python
INITIAL_CONDITIONS = {
    "balanced":     (0.25, 0.25, 0.25, 0.25),
    "tag_majority": (0.55, 0.15, 0.15, 0.15),
    "lag_majority": (0.15, 0.55, 0.15, 0.15),
    "tp_majority":  (0.15, 0.15, 0.55, 0.15),
    "lp_majority":  (0.15, 0.15, 0.15, 0.55),
}
```

Everything except the initial population composition remains identical between conditions.

---

## Interaction

At each interaction event:

1. Randomly select two different agents.
2. Assign one as the focal agent.
3. Assign the other as the comparison agent.
4. Play `K` independent Leduc hands.
5. Keep both strategies fixed during those hands.
6. Calculate the mean realized payoff of each agent.
7. Apply the adaptation rule to the focal agent.

Baseline:

```python
K = 10
```

For an encounter:

```math
\bar{u}_i = \frac{1}{K} \sum_{h=1}^{K} u_i^{(h)}
```

```math
\bar{u}_j = \frac{1}{K} \sum_{h=1}^{K} u_j^{(h)}
```

---

## Adaptation

Calculate the payoff difference:

```math
\Delta_{ij} = \bar{u}_j-\bar{u}_i
```

Normalize it:

```math
d_{ij} = \frac{\Delta_{ij}}{2U_{\max}}
```

The probability that focal agent `i` copies comparison agent `j` is:

```math
P(i \leftarrow j) = \frac{1}{1+\exp(-\beta d_{ij})}
```

Baseline:

```python
BETA = 4.0
```

Implementation:

```python
delta = comparison_payoff - focal_payoff
d = delta / (2 * U_MAX)

p_copy = 1.0 / (1.0 + math.exp(-BETA * d))

if rng.random() < p_copy:
    focal.strategy = comparison.strategy
```

Only the focal agent updates.

There is:

- no policy learning
- no neural-network training
- no mutation

If a strategy disappears from the population, it cannot return during that simulation run.

---

## Simulation

Population size:

```python
N = 100
```

One generation contains `N` interaction events:

```python
INTERACTIONS_PER_GENERATION = N
```

Baseline experiment:

```python
N = 100
GENERATIONS = 200
REPLICATES = 50
K = 10
BETA = 4.0
```

Every replicate must use a recorded random seed.

---

## Main Outputs

Record the population share of each strategy after every generation.

For strategy `s`:

```math
p_s(g) = \frac{\text{number of agents using } s}{N}
```

Record:

```text
TAG share
LAG share
TP share
LP share
```

Also record:

- number of strategy switches
- switching rate
- strategy transition counts
- population entropy
- fixation
- fixation generation

### Switching Rate

```math
R_{\text{switch}}(g) = \frac{\text{number of strategy changes in generation } g}{N}
```

### Population Diversity

```math
H(g) = -\sum_s p_s(g)\log p_s(g)
```

### Fixation

Fixation occurs when:

```math
p_s(g)=1
```

for one strategy.

---

## Main Analysis

Compare the five initial population conditions.

Main questions:

1. Do different starting populations converge to the same strategy distribution?
2. Do different initial compositions produce different long-run states?
3. Does one strategy frequently dominate?
4. Do multiple strategies coexist?
5. Do strategy frequencies oscillate?
6. Which strategy transitions occur most often?
7. Which strategies reach fixation most often?
8. Does the initial majority strategy affect the final outcome?

---

## Sensitivity Experiments

After the baseline experiment works, test:

```python
K_VALUES = [1, 10, 50]
BETA_VALUES = [1, 4, 8]
TAU_VALUES = [(0.65, 0.40), (0.60, 0.40), (0.65, 0.25)]   # (tight, loose)
ALPHA_VALUES = [(0.80, 0.20), (0.70, 0.30), (0.90, 0.10)]  # (aggressive, passive)
```

This tests sensitivity to:

- short-term poker randomness
- strength of payoff-based selection

---

## Repository Structure

```text
src/dmas/
├── leduc/
│   └── game.py
├── strategies/
│   ├── equity.py
│   └── strategy.py
├── agents/
│   └── agent.py
├── evolution/
│   ├── adaptation.py
│   └── population.py
├── simulation/
│   └── runner.py
└── analysis/
    └── metrics.py

tests/
├── test_leduc.py
├── test_equity.py
├── test_strategies.py
├── test_adaptation.py
└── test_reproducibility.py
```

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # create .venv and install dependencies
uv run pytest            # run tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

CI runs lint, format check and tests on every push to `main` and on pull requests.

---

## Minimal Simulation Pseudocode

```python
for condition_name, composition in INITIAL_CONDITIONS.items():

    for replicate in range(REPLICATES):

        rng = create_rng(condition_name, replicate)

        population = initialize_population(
            size=N,
            composition=composition,
            rng=rng,
        )

        record_population_state(
            condition=condition_name,
            replicate=replicate,
            generation=0,
            population=population,
        )

        for generation in range(1, GENERATIONS + 1):

            for _ in range(N):

                focal, comparison = sample_two_agents(
                    population,
                    rng,
                )

                focal_payoff, comparison_payoff = play_encounter(
                    focal=focal,
                    comparison=comparison,
                    hands=K,
                    rng=rng,
                )

                delta = comparison_payoff - focal_payoff
                d = delta / (2 * U_MAX)

                p_copy = 1.0 / (
                    1.0 + math.exp(-BETA * d)
                )

                old_strategy = focal.strategy

                if rng.random() < p_copy:
                    focal.strategy = comparison.strategy

                if focal.strategy != old_strategy:
                    record_transition(
                        old_strategy,
                        focal.strategy,
                    )

            record_population_state(
                condition=condition_name,
                replicate=replicate,
                generation=generation,
                population=population,
            )
```

---

## Implementation Order

```text
1. Leduc Poker environment
2. exact hand-equity calculation
3. parameterized TAG/LAG/TP/LP policies
4. unit tests for poker rules
5. unit tests for strategies
6. agent and population classes
7. Fermi adaptation rule
8. simulation loop
9. result logging
10. baseline experiments
11. analysis
12. sensitivity experiments
```

---

## Core Design Principle

```text
Poker strategies are fixed.

Agents change which strategy they use.

Population composition evolves.
```

The research object is the **population-level dynamics that emerge from repeated interaction between autonomous agents**.