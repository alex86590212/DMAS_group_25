"""Generate plots from saved simulation results.

Run from the project root with:

    uv run python analysis/plots.py baseline-results

Creates:

    results/<run_name>/plots/
        mean_shares_<condition>.png
        entropy_<condition>.png
        fixation_outcomes.png
        fixation_generations_<condition>.png
        transition_heatmap_overall.png
        transition_heatmap_<condition>.png
        example_trajectory_<condition>_rep<replicate>.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from dmas.simulation.results import load_run

from metrics import (
    cumulative_transitions,
    entropy,
    fixation,
    share_trajectories,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# Helpers
# ============================================================

def get_strategies(df: pd.DataFrame) -> list[str]:
    """Return strategy names in stored column order."""
    shares = share_trajectories(df)

    return [
        column.removeprefix("share_")
        for column in shares.columns
    ]


def condition_order(df: pd.DataFrame) -> list[str]:
    """Return conditions in stored order."""
    return list(
        df["condition"].drop_duplicates()
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
) -> None:
    """Save and close a figure."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def mean_confidence_interval(
    values: pd.DataFrame,
    confidence: float = 0.95,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Calculate a two-sided Student's t confidence interval
    for the mean at each row.

    Columns are assumed to contain independent replicates.

    Returns:
        mean
        lower bound
        upper bound
    """
    mean = values.mean(
        axis=1
    )

    std = values.std(
        axis=1,
        ddof=1,
    )

    count = values.count(
        axis=1
    )

    se = (
        std
        / np.sqrt(count)
    )

    alpha = 1.0 - confidence

    critical = pd.Series(
        t.ppf(
            1.0 - alpha / 2.0,
            count - 1,
        ),
        index=count.index,
    )

    margin = (
        critical * se
    )

    lower = (
        mean - margin
    )

    upper = (
        mean + margin
    )

    return (
        mean,
        lower,
        upper,
    )


# ============================================================
# Mean strategy share over time
# ============================================================

def plot_mean_shares(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot mean strategy share over generations.

    One figure is created for each initial condition.

    Shaded bands show two-sided 95% Student's t
    confidence intervals for the mean.
    """
    strategies = get_strategies(
        df
    )

    for condition in condition_order(df):
        condition_df = df[
            df["condition"] == condition
        ]

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for strategy in strategies:
            column = (
                f"share_{strategy}"
            )

            trajectories = []

            for replicate, replicate_df in condition_df.groupby(
                "replicate",
                sort=True,
            ):
                replicate_df = (
                    replicate_df
                    .sort_values(
                        "generation"
                    )
                )

                trajectory = (
                    share_trajectories(
                        replicate_df
                    )[column]
                    .rename(
                        replicate
                    )
                )

                trajectories.append(
                    trajectory
                )

            shares = pd.concat(
                trajectories,
                axis=1,
            )

            mean, lower, upper = mean_confidence_interval(
                shares,
                confidence=0.95,
            )

            lower = lower.clip(
                lower=0,
                upper=1,
            )

            upper = upper.clip(
                lower=0,
                upper=1,
            )

            line = ax.plot(
                mean.index,
                mean,
                label=strategy,
            )[0]

            ax.fill_between(
                mean.index,
                lower,
                upper,
                alpha=0.2,
                color=line.get_color(),
            )

        ax.set_title(
            f"Mean strategy share - {condition}"
        )

        ax.set_xlabel(
            "Generation"
        )

        ax.set_ylabel(
            "Mean population share"
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.legend(
            title="Strategy"
        )

        ax.grid(
            alpha=0.2
        )

        save_figure(
            fig,
            output_dir
            / f"mean_shares_{condition}.png",
        )


# ============================================================
# Entropy over time
# ============================================================

def plot_entropy(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot mean Shannon entropy over generations.

    One figure is created for each initial condition.

    Shaded bands show two-sided 95% Student's t
    confidence intervals for the mean.
    """
    strategies = get_strategies(
        df
    )

    max_entropy = np.log(
        len(strategies)
    )

    for condition in condition_order(df):
        condition_df = df[
            df["condition"] == condition
        ]

        trajectories = []

        for replicate, replicate_df in condition_df.groupby(
            "replicate",
            sort=True,
        ):
            replicate_df = (
                replicate_df
                .sort_values(
                    "generation"
                )
            )

            trajectory = (
                entropy(
                    replicate_df
                )
                .rename(
                    replicate
                )
            )

            trajectories.append(
                trajectory
            )

        entropies = pd.concat(
            trajectories,
            axis=1,
        )

        mean, lower, upper = mean_confidence_interval(
            entropies,
            confidence=0.95,
        )

        lower = lower.clip(
            lower=0,
            upper=max_entropy,
        )

        upper = upper.clip(
            lower=0,
            upper=max_entropy,
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        line = ax.plot(
            mean.index,
            mean,
            label="Mean entropy",
        )[0]

        ax.fill_between(
            mean.index,
            lower,
            upper,
            alpha=0.2,
            color=line.get_color(),
        )

        ax.set_title(
            f"Population entropy - {condition}"
        )

        ax.set_xlabel(
            "Generation"
        )

        ax.set_ylabel(
            "Shannon entropy"
        )

        ax.set_ylim(
            0,
            max_entropy,
        )

        ax.grid(
            alpha=0.2
        )

        save_figure(
            fig,
            output_dir
            / f"entropy_{condition}.png",
        )


# ============================================================
# Fixation data
# ============================================================

def fixation_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one row per replicate with fixation outcome
    and fixation generation.
    """
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

        winner, generation = fixation(
            replicate_df
        )

        rows.append(
            {
                "condition": condition,
                "replicate": replicate,
                "winner": (
                    winner
                    if winner is not None
                    else "not_fixed"
                ),
                "fixation_generation": (
                    generation
                    if generation is not None
                    else -1
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Fixation outcomes
# ============================================================

def plot_fixation_outcomes(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot fixation outcomes as stacked bars."""
    fix = fixation_table(
        df
    )

    counts = (
        fix
        .groupby(
            [
                "condition",
                "winner",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    counts = counts.reindex(
        condition_order(df)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
    )

    ax.set_title(
        "Fixation outcomes by initial condition"
    )

    ax.set_xlabel(
        "Initial condition"
    )

    ax.set_ylabel(
        "Number of replicates"
    )

    ax.legend(
        title="Outcome"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    save_figure(
        fig,
        output_dir
        / "fixation_outcomes.png",
    )


# ============================================================
# Fixation generation histograms
# ============================================================

def plot_fixation_generations(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot fixation generation distributions."""
    fix = fixation_table(
        df
    )

    for condition in condition_order(df):
        generations = fix.loc[
            (
                fix["condition"]
                == condition
            )
            & (
                fix[
                    "fixation_generation"
                ]
                >= 0
            ),
            "fixation_generation",
        ]

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        if not generations.empty:
            ax.hist(
                generations,
                bins="auto",
                edgecolor="black",
                alpha=0.8,
            )

            mean_generation = (
                generations.mean()
            )

            ax.axvline(
                mean_generation,
                linestyle="--",
                label=(
                    f"Mean = "
                    f"{mean_generation:.1f}"
                ),
            )

            ax.legend()

        else:
            ax.text(
                0.5,
                0.5,
                "No replicates reached fixation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(
            f"Fixation generation - {condition}"
        )

        ax.set_xlabel(
            "Generation of fixation"
        )

        ax.set_ylabel(
            "Number of replicates"
        )

        save_figure(
            fig,
            output_dir
            / (
                "fixation_generations_"
                f"{condition}.png"
            ),
        )


# ============================================================
# Transition matrices
# ============================================================

def transition_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return total transition counts across supplied replicates.

    Rows:
        old strategy

    Columns:
        new strategy
    """
    strategies = get_strategies(
        df
    )

    matrix = pd.DataFrame(
        0,
        index=strategies,
        columns=strategies,
        dtype=int,
    )

    for _, replicate_df in df.groupby(
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

        cumulative = cumulative_transitions(
            replicate_df
        )

        if cumulative.empty:
            continue

        totals = cumulative.iloc[
            -1
        ]

        for old in strategies:
            for new in strategies:
                if old == new:
                    continue

                column = (
                    f"trans_{old}_{new}"
                )

                if column in totals.index:
                    matrix.loc[
                        old,
                        new,
                    ] += int(
                        totals[column]
                    )

    return matrix


def draw_transition_heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    """
    Draw transition heatmap.

    Rows are old strategy.
    Columns are new strategy.
    """
    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    values = matrix.to_numpy()

    image = ax.imshow(
        values
    )

    ax.set_xticks(
        range(
            len(matrix.columns)
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_xlabel(
        "New strategy"
    )

    ax.set_ylabel(
        "Old strategy"
    )

    ax.set_title(
        title
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of transitions",
    )

    maximum = (
        values.max()
        if values.size
        else 0
    )

    for i in range(
        len(matrix.index)
    ):
        for j in range(
            len(matrix.columns)
        ):
            value = matrix.iloc[
                i,
                j,
            ]

            if maximum > 0:
                text_color = (
                    "white"
                    if value > maximum / 2
                    else "black"
                )
            else:
                text_color = "black"

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    save_figure(
        fig,
        path,
    )


# ============================================================
# Overall transition heatmap
# ============================================================

def plot_overall_transition_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot transitions pooled across all conditions."""
    matrix = transition_matrix(
        df
    )

    draw_transition_heatmap(
        matrix=matrix,
        title=(
            "Strategy transitions - "
            "all conditions"
        ),
        path=(
            output_dir
            / "transition_heatmap_overall.png"
        ),
    )


# ============================================================
# Transition heatmaps per condition
# ============================================================

def plot_condition_transition_heatmaps(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot one transition heatmap per condition."""
    for condition in condition_order(df):
        condition_df = df[
            df["condition"]
            == condition
        ]

        matrix = transition_matrix(
            condition_df
        )

        draw_transition_heatmap(
            matrix=matrix,
            title=(
                "Strategy transitions - "
                f"{condition}"
            ),
            path=(
                output_dir
                / (
                    "transition_heatmap_"
                    f"{condition}.png"
                )
            ),
        )


# ============================================================
# Example single-replicate trajectories
# ============================================================

def plot_example_trajectories(
    df: pd.DataFrame,
    output_dir: Path,
    n_examples: int = 3,
    seed: int = 42,
) -> None:
    """
    Plot reproducibly selected individual trajectories
    for each condition.
    """
    strategies = get_strategies(
        df
    )

    rng = np.random.default_rng(
        seed
    )

    for condition in condition_order(df):
        condition_df = df[
            df["condition"]
            == condition
        ]

        replicates = np.array(
            sorted(
                condition_df[
                    "replicate"
                ].unique()
            )
        )

        n = min(
            n_examples,
            len(replicates),
        )

        chosen = rng.choice(
            replicates,
            size=n,
            replace=False,
        )

        for replicate in chosen:
            replicate_df = condition_df[
                condition_df[
                    "replicate"
                ]
                == replicate
            ].sort_values(
                "generation"
            )

            trajectories = share_trajectories(
                replicate_df
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            for strategy in strategies:
                column = (
                    f"share_{strategy}"
                )

                ax.plot(
                    trajectories.index,
                    trajectories[column],
                    label=strategy,
                )

            _, fixation_generation = fixation(
                replicate_df
            )

            if fixation_generation is not None:
                ax.axvline(
                    fixation_generation,
                    linestyle="--",
                    alpha=0.7,
                    label=(
                        "Fixation: generation "
                        f"{fixation_generation}"
                    ),
                )

            ax.set_title(
                f"{condition} - "
                f"replicate {replicate}"
            )

            ax.set_xlabel(
                "Generation"
            )

            ax.set_ylabel(
                "Population share"
            )

            ax.set_ylim(
                0,
                1,
            )

            ax.legend()

            ax.grid(
                alpha=0.2
            )

            save_figure(
                fig,
                output_dir
                / (
                    "example_trajectory_"
                    f"{condition}_"
                    f"rep{replicate}.png"
                ),
            )


# ============================================================
# Run all plots
# ============================================================

def make_all_plots(
    run_name: str,
    root: Path = RESULTS_DIR,
) -> None:
    """Load one saved run and generate all standard plots."""

    df, metadata = load_run(
        run_name,
        root,
    )

    output_dir = (
        root
        / run_name
        / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Run: {run_name}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        "Conditions:",
        ", ".join(
            condition_order(df)
        ),
    )

    print(
        "Strategies:",
        ", ".join(
            get_strategies(df)
        ),
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        "\nGenerating mean-share plots..."
    )

    plot_mean_shares(
        df,
        output_dir,
    )

    print(
        "Generating entropy plots..."
    )

    plot_entropy(
        df,
        output_dir,
    )

    print(
        "Generating fixation outcomes..."
    )

    plot_fixation_outcomes(
        df,
        output_dir,
    )

    print(
        "Generating fixation-generation histograms..."
    )

    plot_fixation_generations(
        df,
        output_dir,
    )

    print(
        "Generating overall transition heatmap..."
    )

    plot_overall_transition_heatmap(
        df,
        output_dir,
    )

    print(
        "Generating transition heatmaps per condition..."
    )

    plot_condition_transition_heatmaps(
        df,
        output_dir,
    )

    print(
        "Generating example trajectories..."
    )

    plot_example_trajectories(
        df,
        output_dir,
        n_examples=3,
        seed=42,
    )

    print(
        "\nDone."
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "\nUsage:\n"
            "    uv run python "
            "analysis/plots.py <run_name>\n\n"
            "Example:\n"
            "    uv run python "
            "analysis/plots.py baseline-results"
        )

    make_all_plots(
        sys.argv[1]
    )