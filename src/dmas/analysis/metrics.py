from __future__ import annotations

import numpy as np
import pandas as pd


def _share_columns(df: pd.DataFrame) -> list[str]:
    """Return share columns in the order they appear in the DataFrame"""
    return [column for column in df.columns if column.startswith("share_")]


def _transition_columns(df: pd.DataFrame) -> list[str]:
    """Return transition-count columns in the order they appear"""
    return [column for column in df.columns if column.startswith("trans_")]


def share_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Return strategy shares indexed by generation"""
    columns = _share_columns(df)
    return df.set_index("generation")[columns]


def entropy(df: pd.DataFrame) -> pd.Series:
    """Return Shannon entropy of the strategy distribution at each generation"""
    shares = share_trajectories(df)
    values = shares.to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(values > 0, values * np.log(values), 0.0)

    return pd.Series(-terms.sum(axis=1), index=shares.index, name="entropy")


def switching_rate(df: pd.DataFrame, population_size: int) -> pd.Series:
    """Return the fraction of the population that switches per generation"""
    if population_size <= 0:
        raise ValueError("population_size must be positive")

    return pd.Series(
        df["switches"].to_numpy() / population_size,
        index=df["generation"],
        name="switching_rate",
    )


def fixation(df: pd.DataFrame) -> tuple[str | None, int | None]:
    """Return the fixed strategy and fixation generation, if fixation occurred"""
    fixation_generation = int(df["fixation_generation"].iloc[0])

    if fixation_generation < 0:
        return None, None

    row = df[df["generation"] == fixation_generation].iloc[0]
    share_columns = _share_columns(df)
    strategy = row[share_columns].idxmax().removeprefix("share_")

    return strategy, fixation_generation


def cumulative_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative transition counts by generation"""
    columns = _transition_columns(df)
    return df.set_index("generation")[columns].cumsum()


def oscillation(df: pd.DataFrame) -> pd.Series:
    """Count sign changes in consecutive non-zero share differences"""
    shares = share_trajectories(df)
    result: dict[str, int] = {}

    for column in shares.columns:
        differences = shares[column].diff().to_numpy()
        differences = differences[
            np.isfinite(differences) & ~np.isclose(differences, 0.0)
        ]

        if len(differences) < 2:
            result[column.removeprefix("share_")] = 0
            continue

        signs = np.sign(differences)
        result[column.removeprefix("share_")] = int(
            np.sum(signs[1:] != signs[:-1])
        )

    return pd.Series(result, name="oscillations")