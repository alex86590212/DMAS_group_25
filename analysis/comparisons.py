"""Compare beta = 0 and beta = 4 simulation results.

Run from the project root with:

    uv run python analysis/comparisons.py baseline-results neutral-drift

Most analyses are performed separately for each beta.

The direct beta comparison is limited to:

    1. probability of reaching any fixation
    2. probability of each individual strategy reaching fixation

Outputs:

    results/comparisons/<run_a>_vs_<run_b>/

        final_share_kruskal_wallis.csv
        long_run_outcome_tests.csv
        dominance_and_fixation.csv
        fixation_outcomes.csv
        coexistence_summary.csv
        coexistence_pairs.csv
        oscillation_summary.csv
        transition_counts_beta4.csv
        fixation_probability_beta_comparison.csv
        strategy_fixation_probability_beta_comparison.csv
        comparison_metadata.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    binomtest,
    chi2,
    fisher_exact,
    kruskal,
)

from dmas.simulation.results import load_run

from metrics import (
    entropy,
    fixation,
    oscillation,
    share_trajectories,
)


# ============================================================
# Settings
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

COMPARISONS_DIR = (
    RESULTS_DIR
    / "comparisons"
)

ALPHA = 0.05

N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 42


# ============================================================
# General helpers
# ============================================================

def get_strategies(
    df: pd.DataFrame,
) -> list[str]:
    """Return strategy names in stored order."""
    shares = share_trajectories(
        df
    )

    return [
        column.removeprefix(
            "share_"
        )
        for column
        in shares.columns
    ]


def get_conditions(
    df: pd.DataFrame,
) -> list[str]:
    """Return initial conditions in stored order."""
    return list(
        df[
            "condition"
        ].drop_duplicates()
    )


def run_beta(
    metadata: dict,
) -> float:
    """Read beta from saved run metadata."""
    return float(
        metadata[
            "config"
        ][
            "beta"
        ]
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Return numerator / denominator, or NaN for zero denominator."""
    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
    )


def bh_adjust(
    p_values: pd.Series,
) -> pd.Series:
    """Benjamini-Hochberg multiple-testing correction."""
    result = pd.Series(
        np.nan,
        index=p_values.index,
        dtype=float,
    )

    valid = (
        p_values
        .dropna()
    )

    if valid.empty:
        return result

    values = valid.to_numpy(
        dtype=float
    )

    order = np.argsort(
        values
    )

    sorted_values = (
        values[
            order
        ]
    )

    n = len(
        sorted_values
    )

    adjusted = (
        sorted_values
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    adjusted = (
        np.minimum.accumulate(
            adjusted[::-1]
        )[::-1]
    )

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    unsorted = np.empty_like(
        adjusted
    )

    unsorted[
        order
    ] = adjusted

    result.loc[
        valid.index
    ] = unsorted

    return result


def exact_binomial_ci(
    successes: int,
    total: int,
) -> tuple[float, float]:
    """Return an exact 95% binomial confidence interval."""
    if total == 0:
        return (
            np.nan,
            np.nan,
        )

    interval = (
        binomtest(
            successes,
            total,
        )
        .proportion_ci(
            confidence_level=0.95,
            method="exact",
        )
    )

    return (
        float(
            interval.low
        ),
        float(
            interval.high
        ),
    )


# ============================================================
# Internal final-state data
#
# One row per replicate is needed internally for the tests,
# but this table is not written to disk.
# ============================================================

def build_final_states(
    run_name: str,
    beta: float,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one internal row per replicate."""
    strategies = get_strategies(
        df
    )

    rows = []

    for (
        condition,
        replicate,
    ), replicate_df in df.groupby(
        [
            "condition",
            "replicate",
        ],
        sort=False,
    ):
        replicate_df = (
            replicate_df
            .sort_values(
                "generation"
            )
        )

        shares = share_trajectories(
            replicate_df
        )

        final_shares = (
            shares.iloc[-1]
        )

        (
            fixed_strategy,
            fixation_generation,
        ) = fixation(
            replicate_df
        )

        oscillations = oscillation(
            replicate_df
        )

        row = {
            "run": run_name,
            "beta": beta,
            "condition": condition,
            "replicate": replicate,

            "fixed": (
                fixed_strategy
                is not None
            ),

            "winner": (
                fixed_strategy
                if fixed_strategy
                is not None
                else "not_fixed"
            ),

            "fixation_generation": (
                fixation_generation
                if fixation_generation
                is not None
                else -1
            ),

            "leading_strategy": (
                final_shares
                .idxmax()
                .removeprefix(
                    "share_"
                )
            ),

            "max_final_share": float(
                final_shares.max()
            ),

            "final_entropy": float(
                entropy(
                    replicate_df
                ).iloc[-1]
            ),

            "oscillation_total": int(
                oscillations.sum()
            ),
        }

        for strategy in strategies:
            row[
                f"final_share_{strategy}"
            ] = float(
                final_shares[
                    f"share_{strategy}"
                ]
            )

            row[
                f"oscillation_{strategy}"
            ] = int(
                oscillations[
                    strategy
                ]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 1. Final share distributions
# ============================================================

def final_share_tests(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether final strategy shares differ across
    initial conditions.

    One Kruskal-Wallis test is performed per strategy
    and per beta.
    """
    rows = []

    for (
        run_name,
        beta,
    ), run_df in states.groupby(
        [
            "run",
            "beta",
        ],
        sort=False,
    ):
        conditions = list(
            run_df[
                "condition"
            ].drop_duplicates()
        )

        share_columns = [
            column
            for column
            in run_df.columns
            if column.startswith(
                "final_share_"
            )
        ]

        for column in share_columns:
            strategy = (
                column
                .removeprefix(
                    "final_share_"
                )
            )

            groups = [
                run_df.loc[
                    run_df[
                        "condition"
                    ]
                    == condition,
                    column,
                ].to_numpy()
                for condition
                in conditions
            ]

            try:
                statistic, p_value = (
                    kruskal(
                        *groups
                    )
                )

            except ValueError:
                statistic = 0.0
                p_value = 1.0

            n = sum(
                len(group)
                for group
                in groups
            )

            k = len(
                groups
            )

            if n > k:
                epsilon_squared = max(
                    0.0,
                    (
                        statistic
                        - k
                        + 1
                    )
                    / (
                        n
                        - k
                    ),
                )

            else:
                epsilon_squared = np.nan

            rows.append(
                {
                    "run": run_name,
                    "beta": beta,
                    "strategy": strategy,
                    "H": statistic,
                    "p_value": p_value,
                    "epsilon_squared": (
                        epsilon_squared
                    ),
                    "n": n,
                }
            )

    result = pd.DataFrame(
        rows
    )

    result[
        "p_adjusted_bh"
    ] = np.nan

    for (
        run_name,
        beta,
    ), indices in result.groupby(
        [
            "run",
            "beta",
        ]
    ).groups.items():

        result.loc[
            indices,
            "p_adjusted_bh",
        ] = bh_adjust(
            result.loc[
                indices,
                "p_value",
            ]
        )

    result[
        "significant_bh_0_05"
    ] = (
        result[
            "p_adjusted_bh"
        ]
        < ALPHA
    )

    return result


# ============================================================
# 2. Long-run outcome association
#
# Pearson chi-square statistic with a permutation p-value.
# ============================================================

def contingency_table_from_codes(
    condition_codes: np.ndarray,
    outcome_codes: np.ndarray,
    n_conditions: int,
    n_outcomes: int,
) -> np.ndarray:
    """Build a contingency table from integer codes."""
    table = np.zeros(
        (
            n_conditions,
            n_outcomes,
        ),
        dtype=int,
    )

    np.add.at(
        table,
        (
            condition_codes,
            outcome_codes,
        ),
        1,
    )

    return table


def chi_square_statistic(
    table: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
]:
    """Return Pearson chi-square statistic and expected counts."""
    row_totals = (
        table.sum(
            axis=1,
            keepdims=True,
        )
    )

    column_totals = (
        table.sum(
            axis=0,
            keepdims=True,
        )
    )

    total = (
        table.sum()
    )

    expected = (
        row_totals
        @ column_totals
        / total
    )

    mask = (
        expected > 0
    )

    statistic = float(
        np.sum(
            (
                table[
                    mask
                ]
                - expected[
                    mask
                ]
            )
            ** 2
            / expected[
                mask
            ]
        )
    )

    return (
        statistic,
        expected,
    )


def permutation_chi_square(
    subset: pd.DataFrame,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """
    Test association between initial condition and final outcome
    using a permutation p-value.
    """
    condition_values = (
        subset[
            "condition"
        ]
        .astype(str)
        .to_numpy()
    )

    outcome_values = (
        subset[
            "winner"
        ]
        .astype(str)
        .to_numpy()
    )

    condition_levels = list(
        pd.unique(
            condition_values
        )
    )

    outcome_levels = list(
        pd.unique(
            outcome_values
        )
    )

    if (
        len(condition_levels) < 2
        or len(outcome_levels) < 2
    ):
        return {
            "chi_square": np.nan,
            "degrees_of_freedom": np.nan,
            "asymptotic_p_value": np.nan,
            "permutation_p_value": np.nan,
            "cramers_v": np.nan,
            "minimum_expected_count": np.nan,
            "n": len(
                subset
            ),
        }

    condition_map = {
        value: i
        for i, value
        in enumerate(
            condition_levels
        )
    }

    outcome_map = {
        value: i
        for i, value
        in enumerate(
            outcome_levels
        )
    }

    condition_codes = np.array(
        [
            condition_map[
                value
            ]
            for value
            in condition_values
        ],
        dtype=int,
    )

    outcome_codes = np.array(
        [
            outcome_map[
                value
            ]
            for value
            in outcome_values
        ],
        dtype=int,
    )

    observed_table = (
        contingency_table_from_codes(
            condition_codes,
            outcome_codes,
            len(
                condition_levels
            ),
            len(
                outcome_levels
            ),
        )
    )

    (
        observed_statistic,
        expected,
    ) = chi_square_statistic(
        observed_table
    )

    degrees_of_freedom = (
        (
            len(
                condition_levels
            )
            - 1
        )
        * (
            len(
                outcome_levels
            )
            - 1
        )
    )

    asymptotic_p_value = float(
        chi2.sf(
            observed_statistic,
            degrees_of_freedom,
        )
    )

    extreme = 0

    for _ in range(
        n_permutations
    ):
        shuffled_conditions = (
            rng.permutation(
                condition_codes
            )
        )

        permuted_table = (
            contingency_table_from_codes(
                shuffled_conditions,
                outcome_codes,
                len(
                    condition_levels
                ),
                len(
                    outcome_levels
                ),
            )
        )

        (
            permutation_statistic,
            _,
        ) = chi_square_statistic(
            permuted_table
        )

        if (
            permutation_statistic
            >= observed_statistic
            - 1e-12
        ):
            extreme += 1

    permutation_p_value = (
        extreme + 1
    ) / (
        n_permutations + 1
    )

    n = int(
        observed_table.sum()
    )

    denominator = min(
        len(
            condition_levels
        )
        - 1,
        len(
            outcome_levels
        )
        - 1,
    )

    cramers_v = np.sqrt(
        observed_statistic
        / (
            n
            * denominator
        )
    )

    return {
        "chi_square": (
            observed_statistic
        ),

        "degrees_of_freedom": (
            degrees_of_freedom
        ),

        "asymptotic_p_value": (
            asymptotic_p_value
        ),

        "permutation_p_value": (
            permutation_p_value
        ),

        "cramers_v": (
            cramers_v
        ),

        "minimum_expected_count": float(
            expected.min()
        ),

        "n": n,
    }


def long_run_outcome_tests(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether final outcomes depend on initial condition.

    Calculated separately for each beta.

    all_outcomes:
        includes fixed strategies and not_fixed

    fixed_only:
        includes only runs that reached fixation
    """
    rows = []

    rng = np.random.default_rng(
        PERMUTATION_SEED
    )

    for (
        run_name,
        beta,
    ), run_df in states.groupby(
        [
            "run",
            "beta",
        ],
        sort=False,
    ):

        analyses = {
            "all_outcomes": (
                run_df
            ),

            "fixed_only": (
                run_df[
                    run_df[
                        "fixed"
                    ]
                ]
            ),
        }

        for (
            analysis_name,
            subset,
        ) in analyses.items():

            result = (
                permutation_chi_square(
                    subset,
                    rng,
                )
            )

            rows.append(
                {
                    "run": run_name,
                    "beta": beta,
                    "analysis": (
                        analysis_name
                    ),
                    **result,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 3 + 7. Dominance and fixation
# ============================================================

def dominance_and_fixation_summary(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize final shares, final leaders and fixation winners.

    Results are stored per condition and for all conditions pooled.
    """
    strategies = [
        column.removeprefix(
            "final_share_"
        )
        for column
        in states.columns
        if column.startswith(
            "final_share_"
        )
    ]

    rows = []

    for (
        run_name,
        beta,
    ), run_df in states.groupby(
        [
            "run",
            "beta",
        ],
        sort=False,
    ):

        scopes = [
            (
                condition,
                run_df[
                    run_df[
                        "condition"
                    ]
                    == condition
                ],
            )
            for condition
            in run_df[
                "condition"
            ].drop_duplicates()
        ]

        scopes.append(
            (
                "ALL",
                run_df,
            )
        )

        for (
            condition,
            subset,
        ) in scopes:

            n = len(
                subset
            )

            fixed_total = int(
                subset[
                    "fixed"
                ].sum()
            )

            for strategy in strategies:

                fixation_count = int(
                    (
                        subset[
                            "winner"
                        ]
                        == strategy
                    ).sum()
                )

                leader_count = int(
                    (
                        subset[
                            "leading_strategy"
                        ]
                        == strategy
                    ).sum()
                )

                rows.append(
                    {
                        "run": run_name,
                        "beta": beta,
                        "condition": condition,
                        "strategy": strategy,

                        "replicates": n,

                        "mean_final_share": float(
                            subset[
                                f"final_share_{strategy}"
                            ].mean()
                        ),

                        "median_final_share": float(
                            subset[
                                f"final_share_{strategy}"
                            ].median()
                        ),

                        "final_leader_count": (
                            leader_count
                        ),

                        "final_leader_probability": safe_divide(
                            leader_count,
                            n,
                        ),

                        "fixation_count": (
                            fixation_count
                        ),

                        "fixation_probability_all_runs": safe_divide(
                            fixation_count,
                            n,
                        ),

                        "fixation_share_among_fixed": safe_divide(
                            fixation_count,
                            fixed_total,
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def fixation_outcomes(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count final fixation outcomes by beta and condition.
    """
    rows = []

    for (
        run_name,
        beta,
    ), run_df in states.groupby(
        [
            "run",
            "beta",
        ],
        sort=False,
    ):

        scopes = [
            (
                condition,
                run_df[
                    run_df[
                        "condition"
                    ]
                    == condition
                ],
            )
            for condition
            in run_df[
                "condition"
            ].drop_duplicates()
        ]

        scopes.append(
            (
                "ALL",
                run_df,
            )
        )

        for (
            condition,
            subset,
        ) in scopes:

            total = len(
                subset
            )

            for (
                outcome,
                count,
            ) in (
                subset[
                    "winner"
                ]
                .value_counts()
                .items()
            ):

                rows.append(
                    {
                        "run": run_name,
                        "beta": beta,
                        "condition": condition,
                        "outcome": outcome,
                        "count": int(
                            count
                        ),
                        "proportion": safe_divide(
                            count,
                            total,
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 4. Coexistence
# ============================================================

def coexistence_summary(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how often multiple strategies remain at the
    final generation.
    """
    share_columns = [
        column
        for column
        in states.columns
        if column.startswith(
            "final_share_"
        )
    ]

    rows = []

    for (
        run_name,
        beta,
    ), run_df in states.groupby(
        [
            "run",
            "beta",
        ],
        sort=False,
    ):

        scopes = [
            (
                condition,
                run_df[
                    run_df[
                        "condition"
                    ]
                    == condition
                ],
            )
            for condition
            in run_df[
                "condition"
            ].drop_duplicates()
        ]

        scopes.append(
            (
                "ALL",
                run_df,
            )
        )

        for (
            condition,
            subset,
        ) in scopes:

            surviving_counts = (
                subset[
                    share_columns
                ]
                .gt(
                    0
                )
                .sum(
                    axis=1
                )
            )

            coexisting = (
                surviving_counts
                >= 2
            )

            rows.append(
                {
                    "run": run_name,
                    "beta": beta,
                    "condition": condition,

                    "replicates": len(
                        subset
                    ),

                    "coexisting_replicates": int(
                        coexisting.sum()
                    ),

                    "coexistence_probability": float(
                        coexisting.mean()
                    ),

                    "mean_surviving_strategies": float(
                        surviving_counts.mean()
                    ),

                    "median_surviving_strategies": float(
                        surviving_counts.median()
                    ),

                    "mean_final_entropy": float(
                        subset[
                            "final_entropy"
                        ].mean()
                    ),

                    "mean_max_final_share": float(
                        subset[
                            "max_final_share"
                        ].mean()
                    ),

                    "fixation_probability": float(
                        subset[
                            "fixed"
                        ].mean()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def coexistence_pairs(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count which pairs of strategies coexist at the final generation.

    A pair is present when both strategies have a final share > 0.

    If three strategies survive in one replicate, that replicate
    contributes to all three corresponding strategy pairs.
    """
    strategies = [
        column.removeprefix(
            "final_share_"
        )
        for column
        in states.columns
        if column.startswith(
            "final_share_"
        )
    ]

    rows = []

    for (
        run_name,
        beta,
    ), run_df in states.groupby(
        [
            "run",
            "beta",
        ],
        sort=False,
    ):

        scopes = [
            (
                condition,
                run_df[
                    run_df[
                        "condition"
                    ]
                    == condition
                ],
            )
            for condition
            in run_df[
                "condition"
            ].drop_duplicates()
        ]

        scopes.append(
            (
                "ALL",
                run_df,
            )
        )

        for (
            condition,
            subset,
        ) in scopes:

            total_replicates = len(
                subset
            )

            for (
                strategy_a,
                strategy_b,
            ) in combinations(
                strategies,
                2,
            ):

                present_a = (
                    subset[
                        f"final_share_{strategy_a}"
                    ]
                    > 0
                )

                present_b = (
                    subset[
                        f"final_share_{strategy_b}"
                    ]
                    > 0
                )

                both_present = (
                    present_a
                    & present_b
                )

                count = int(
                    both_present.sum()
                )

                if count == 0:
                    continue

                rows.append(
                    {
                        "run": run_name,
                        "beta": beta,
                        "condition": condition,

                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,

                        "count": count,

                        "proportion_of_replicates": safe_divide(
                            count,
                            total_replicates,
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 5. Oscillation
# ============================================================

def oscillation_summary(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize direction reversals in strategy shares.

    Uses metrics.oscillation().
    """
    oscillation_columns = [
        column
        for column
        in states.columns
        if (
            column.startswith(
                "oscillation_"
            )
            and column
            != "oscillation_total"
        )
    ]

    rows = []

    for (
        run_name,
        beta,
    ), run_df in states.groupby(
        [
            "run",
            "beta",
        ],
        sort=False,
    ):

        scopes = [
            (
                condition,
                run_df[
                    run_df[
                        "condition"
                    ]
                    == condition
                ],
            )
            for condition
            in run_df[
                "condition"
            ].drop_duplicates()
        ]

        scopes.append(
            (
                "ALL",
                run_df,
            )
        )

        for (
            condition,
            subset,
        ) in scopes:

            row = {
                "run": run_name,
                "beta": beta,
                "condition": condition,
                "replicates": len(
                    subset
                ),

                "mean_total_direction_reversals": float(
                    subset[
                        "oscillation_total"
                    ].mean()
                ),

                "median_total_direction_reversals": float(
                    subset[
                        "oscillation_total"
                    ].median()
                ),
            }

            for column in oscillation_columns:

                strategy = (
                    column
                    .removeprefix(
                        "oscillation_"
                    )
                )

                row[
                    f"mean_{strategy}_direction_reversals"
                ] = float(
                    subset[
                        column
                    ].mean()
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 6. Directional transitions for beta = 4 only
# ============================================================

def transition_counts(
    run_name: str,
    beta: float,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize directional strategy transitions for beta = 4.

    Counts are produced per condition and across all conditions.
    """
    strategies = get_strategies(
        df
    )

    rows = []

    scopes = [
        (
            condition,
            df[
                df[
                    "condition"
                ]
                == condition
            ],
        )
        for condition
        in get_conditions(
            df
        )
    ]

    scopes.append(
        (
            "ALL",
            df,
        )
    )

    for (
        condition,
        subset,
    ) in scopes:

        totals = {}

        for old in strategies:
            for new in strategies:

                if old == new:
                    continue

                column = (
                    f"trans_{old}_{new}"
                )

                totals[
                    (
                        old,
                        new,
                    )
                ] = int(
                    subset[
                        column
                    ].sum()
                )

        total_transitions = sum(
            totals.values()
        )

        outgoing_totals = {
            old: sum(
                count
                for (
                    source,
                    destination,
                ), count
                in totals.items()
                if source == old
            )
            for old
            in strategies
        }

        for (
            old,
            new,
        ), count in totals.items():

            rows.append(
                {
                    "run": run_name,
                    "beta": beta,
                    "condition": condition,

                    "old_strategy": old,
                    "new_strategy": new,

                    "count": count,

                    "share_of_all_transitions": safe_divide(
                        count,
                        total_transitions,
                    ),

                    "share_of_outgoing_from_old": safe_divide(
                        count,
                        outgoing_totals[
                            old
                        ],
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Cross-beta comparison:
# probability of any fixation
# ============================================================

def fixation_probability_comparison(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare probability of reaching any fixation between betas.

    Two-sided Fisher exact tests are performed:
        - for each initial condition
        - for all conditions pooled

    Condition-specific tests receive BH correction.
    """
    beta_values = sorted(
        states[
            "beta"
        ].unique()
    )

    if len(
        beta_values
    ) != 2:
        raise ValueError(
            "Expected exactly two beta values."
        )

    beta_a, beta_b = (
        beta_values
    )

    conditions = list(
        states[
            "condition"
        ].drop_duplicates()
    )

    rows = []

    for condition in [
        *conditions,
        "ALL",
    ]:

        if condition == "ALL":
            subset = states

        else:
            subset = states[
                states[
                    "condition"
                ]
                == condition
            ]

        a = subset[
            subset[
                "beta"
            ]
            == beta_a
        ]

        b = subset[
            subset[
                "beta"
            ]
            == beta_b
        ]

        fixed_a = int(
            a[
                "fixed"
            ].sum()
        )

        fixed_b = int(
            b[
                "fixed"
            ].sum()
        )

        total_a = len(
            a
        )

        total_b = len(
            b
        )

        table = np.array(
            [
                [
                    fixed_a,
                    total_a
                    - fixed_a,
                ],
                [
                    fixed_b,
                    total_b
                    - fixed_b,
                ],
            ]
        )

        (
            odds_ratio,
            p_value,
        ) = fisher_exact(
            table,
            alternative="two-sided",
        )

        probability_a = safe_divide(
            fixed_a,
            total_a,
        )

        probability_b = safe_divide(
            fixed_b,
            total_b,
        )

        (
            ci_a_low,
            ci_a_high,
        ) = exact_binomial_ci(
            fixed_a,
            total_a,
        )

        (
            ci_b_low,
            ci_b_high,
        ) = exact_binomial_ci(
            fixed_b,
            total_b,
        )

        rows.append(
            {
                "condition": condition,

                "beta_a": beta_a,
                "beta_a_fixed": fixed_a,
                "beta_a_total": total_a,
                "beta_a_fixation_probability": (
                    probability_a
                ),
                "beta_a_ci95_low": (
                    ci_a_low
                ),
                "beta_a_ci95_high": (
                    ci_a_high
                ),

                "beta_b": beta_b,
                "beta_b_fixed": fixed_b,
                "beta_b_total": total_b,
                "beta_b_fixation_probability": (
                    probability_b
                ),
                "beta_b_ci95_low": (
                    ci_b_low
                ),
                "beta_b_ci95_high": (
                    ci_b_high
                ),

                "probability_difference_b_minus_a": (
                    probability_b
                    - probability_a
                ),

                "odds_ratio": (
                    odds_ratio
                ),

                "p_value": (
                    p_value
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "p_adjusted_bh"
    ] = np.nan

    condition_mask = (
        result[
            "condition"
        ]
        != "ALL"
    )

    result.loc[
        condition_mask,
        "p_adjusted_bh",
    ] = bh_adjust(
        result.loc[
            condition_mask,
            "p_value",
        ]
    )

    result[
        "significant_0_05"
    ] = (
        result[
            "p_value"
        ]
        < ALPHA
    )

    result[
        "significant_bh_0_05"
    ] = False

    result.loc[
        condition_mask,
        "significant_bh_0_05",
    ] = (
        result.loc[
            condition_mask,
            "p_adjusted_bh",
        ]
        < ALPHA
    )

    return result


# ============================================================
# Cross-beta comparison:
# fixation probability by individual strategy
# ============================================================

def strategy_fixation_probability_comparison(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare fixation probability for each strategy between betas.

    For each strategy:

                    strategy fixed    strategy did not fix
        beta A
        beta B

    Strategy did not fix includes both:
        - another strategy fixed
        - the population did not reach fixation

    Fisher exact tests are run per condition and for all
    conditions pooled.

    BH correction is applied across strategies within
    each condition.
    """
    beta_values = sorted(
        states[
            "beta"
        ].unique()
    )

    if len(
        beta_values
    ) != 2:
        raise ValueError(
            "Expected exactly two beta values."
        )

    beta_a, beta_b = (
        beta_values
    )

    strategies = [
        column.removeprefix(
            "final_share_"
        )
        for column
        in states.columns
        if column.startswith(
            "final_share_"
        )
    ]

    conditions = list(
        states[
            "condition"
        ].drop_duplicates()
    )

    rows = []

    for condition in [
        *conditions,
        "ALL",
    ]:

        if condition == "ALL":
            subset = states

        else:
            subset = states[
                states[
                    "condition"
                ]
                == condition
            ]

        a = subset[
            subset[
                "beta"
            ]
            == beta_a
        ]

        b = subset[
            subset[
                "beta"
            ]
            == beta_b
        ]

        for strategy in strategies:

            fixed_a = int(
                (
                    a[
                        "winner"
                    ]
                    == strategy
                ).sum()
            )

            fixed_b = int(
                (
                    b[
                        "winner"
                    ]
                    == strategy
                ).sum()
            )

            total_a = len(
                a
            )

            total_b = len(
                b
            )

            table = np.array(
                [
                    [
                        fixed_a,
                        total_a
                        - fixed_a,
                    ],
                    [
                        fixed_b,
                        total_b
                        - fixed_b,
                    ],
                ]
            )

            (
                odds_ratio,
                p_value,
            ) = fisher_exact(
                table,
                alternative="two-sided",
            )

            probability_a = safe_divide(
                fixed_a,
                total_a,
            )

            probability_b = safe_divide(
                fixed_b,
                total_b,
            )

            (
                ci_a_low,
                ci_a_high,
            ) = exact_binomial_ci(
                fixed_a,
                total_a,
            )

            (
                ci_b_low,
                ci_b_high,
            ) = exact_binomial_ci(
                fixed_b,
                total_b,
            )

            rows.append(
                {
                    "condition": condition,
                    "strategy": strategy,

                    "beta_a": beta_a,
                    "beta_a_fixations": (
                        fixed_a
                    ),
                    "beta_a_total": (
                        total_a
                    ),
                    "beta_a_fixation_probability": (
                        probability_a
                    ),
                    "beta_a_ci95_low": (
                        ci_a_low
                    ),
                    "beta_a_ci95_high": (
                        ci_a_high
                    ),

                    "beta_b": beta_b,
                    "beta_b_fixations": (
                        fixed_b
                    ),
                    "beta_b_total": (
                        total_b
                    ),
                    "beta_b_fixation_probability": (
                        probability_b
                    ),
                    "beta_b_ci95_low": (
                        ci_b_low
                    ),
                    "beta_b_ci95_high": (
                        ci_b_high
                    ),

                    "probability_difference_b_minus_a": (
                        probability_b
                        - probability_a
                    ),

                    "odds_ratio": (
                        odds_ratio
                    ),

                    "p_value": (
                        p_value
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    result[
        "p_adjusted_bh"
    ] = np.nan

    for (
        condition,
        indices,
    ) in result.groupby(
        "condition"
    ).groups.items():

        result.loc[
            indices,
            "p_adjusted_bh",
        ] = bh_adjust(
            result.loc[
                indices,
                "p_value",
            ]
        )

    result[
        "significant_bh_0_05"
    ] = (
        result[
            "p_adjusted_bh"
        ]
        < ALPHA
    )

    return result


# ============================================================
# Main
# ============================================================

def compare_runs(
    run_a: str,
    run_b: str,
) -> Path:
    """Run all analyses and save summary CSV files."""

    df_a, metadata_a = load_run(
        run_a,
        RESULTS_DIR,
    )

    df_b, metadata_b = load_run(
        run_b,
        RESULTS_DIR,
    )

    beta_a = run_beta(
        metadata_a
    )

    beta_b = run_beta(
        metadata_b
    )

    print(
        f"{run_a}: beta = {beta_a}"
    )

    print(
        f"{run_b}: beta = {beta_b}"
    )

    if np.isclose(
        beta_a,
        beta_b,
    ):
        raise ValueError(
            "The two runs have the same beta."
        )

    output_dir = (
        COMPARISONS_DIR
        / (
            f"{run_a}_vs_"
            f"{run_b}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Internal final-state data
    # --------------------------------------------------------

    print(
        "Building internal final-state data..."
    )

    states_a = build_final_states(
        run_a,
        beta_a,
        df_a,
    )

    states_b = build_final_states(
        run_b,
        beta_b,
        df_b,
    )

    states = pd.concat(
        [
            states_a,
            states_b,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Q1
    # --------------------------------------------------------

    print(
        "Testing final-share distributions..."
    )

    final_share_results = (
        final_share_tests(
            states
        )
    )

    final_share_results.to_csv(
        output_dir
        / "final_share_kruskal_wallis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Q2
    # --------------------------------------------------------

    print(
        "Testing long-run outcomes..."
    )

    long_run_results = (
        long_run_outcome_tests(
            states
        )
    )

    long_run_results.to_csv(
        output_dir
        / "long_run_outcome_tests.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Q3 and Q7
    # --------------------------------------------------------

    print(
        "Summarizing dominance and fixation..."
    )

    dominance = (
        dominance_and_fixation_summary(
            states
        )
    )

    dominance.to_csv(
        output_dir
        / "dominance_and_fixation.csv",
        index=False,
    )

    fixation_table = (
        fixation_outcomes(
            states
        )
    )

    fixation_table.to_csv(
        output_dir
        / "fixation_outcomes.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Q4
    # --------------------------------------------------------

    print(
        "Summarizing coexistence..."
    )

    coexistence = (
        coexistence_summary(
            states
        )
    )

    coexistence.to_csv(
        output_dir
        / "coexistence_summary.csv",
        index=False,
    )

    coexistence_pair_table = (
        coexistence_pairs(
            states
        )
    )

    coexistence_pair_table.to_csv(
        output_dir
        / "coexistence_pairs.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Q5
    # --------------------------------------------------------

    print(
        "Summarizing strategy direction reversals..."
    )

    oscillations = (
        oscillation_summary(
            states
        )
    )

    oscillations.to_csv(
        output_dir
        / "oscillation_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Q6 - beta = 4 only
    # --------------------------------------------------------

    beta4_data = None

    if np.isclose(
        beta_a,
        4.0,
    ):
        beta4_data = (
            run_a,
            beta_a,
            df_a,
        )

    elif np.isclose(
        beta_b,
        4.0,
    ):
        beta4_data = (
            run_b,
            beta_b,
            df_b,
        )

    if beta4_data is None:
        raise ValueError(
            "No beta = 4 run was found."
        )

    (
        beta4_run,
        beta4_value,
        beta4_df,
    ) = beta4_data

    print(
        "Summarizing beta = 4 strategy transitions..."
    )

    transitions = (
        transition_counts(
            beta4_run,
            beta4_value,
            beta4_df,
        )
    )

    transitions.to_csv(
        output_dir
        / "transition_counts_beta4.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Cross-beta overall fixation probability
    # --------------------------------------------------------

    print(
        "Comparing overall fixation probabilities..."
    )

    fixation_comparison = (
        fixation_probability_comparison(
            states
        )
    )

    fixation_comparison.to_csv(
        output_dir
        / "fixation_probability_beta_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Cross-beta strategy-specific fixation probability
    # --------------------------------------------------------

    print(
        "Comparing strategy-specific fixation probabilities..."
    )

    strategy_fixation_comparison = (
        strategy_fixation_probability_comparison(
            states
        )
    )

    strategy_fixation_comparison.to_csv(
        output_dir
        / (
            "strategy_fixation_probability_"
            "beta_comparison.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    comparison_metadata = {
        "run_a": run_a,
        "beta_a": beta_a,

        "run_b": run_b,
        "beta_b": beta_b,

        "alpha": ALPHA,

        "long_run_outcome_test": (
            "Pearson chi-square statistic with "
            "permutation p-value"
        ),

        "n_permutations": (
            N_PERMUTATIONS
        ),

        "permutation_seed": (
            PERMUTATION_SEED
        ),

        "fixation_beta_test": (
            "two-sided Fisher exact test"
        ),

        "strategy_fixation_beta_test": (
            "two-sided Fisher exact test"
        ),

        "multiple_testing": (
            "Benjamini-Hochberg"
        ),

        "run_a_config": (
            metadata_a.get(
                "config"
            )
        ),

        "run_b_config": (
            metadata_b.get(
                "config"
            )
        ),

        "run_a_git_commit": (
            metadata_a.get(
                "git_commit"
            )
        ),

        "run_b_git_commit": (
            metadata_b.get(
                "git_commit"
            )
        ),
    }

    (
        output_dir
        / "comparison_metadata.json"
    ).write_text(
        json.dumps(
            comparison_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Done. Results saved to: {output_dir}"
    )

    return output_dir


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(
        sys.argv
    ) != 3:

        raise SystemExit(
            "\nUsage:\n"
            "    uv run python "
            "analysis/comparisons.py "
            "<run_a> <run_b>\n\n"
            "Example:\n"
            "    uv run python "
            "analysis/comparisons.py "
            "baseline-results neutral-drift"
        )

    compare_runs(
        sys.argv[1],
        sys.argv[2],
    )