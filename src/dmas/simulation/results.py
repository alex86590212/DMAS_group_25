"""Persist simulation runs as tidy CSV plus metadata, and load them back for analysis.

Layout of results/<run_name>/:
    generations.csv  one row per (condition, replicate, generation)
    metadata.json    config, git commit and per-replicate seeds
"""

import dataclasses
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from dmas.simulation.runner import ReplicateResult, SimulationConfig, run_condition

RESULTS_DIR = Path("results")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def results_to_frame(results: Iterable[ReplicateResult]) -> pd.DataFrame:
    """Tidy frame: shares, switch count, transition counts and fixation per generation."""
    frames = []
    for r in results:
        g_count = r.switches.shape[0]
        generation = np.arange(g_count + 1)
        cols: dict[str, object] = {
            "condition": r.condition,
            "replicate": r.replicate,
            "generation": generation,
        }
        for i, s in enumerate(r.strategies):
            cols[f"share_{s}"] = r.shares[:, i]
        # Generation 0 is the initial state: no interactions have happened yet.
        cols["switches"] = np.concatenate([[0], r.switches])
        for i, old in enumerate(r.strategies):
            for j, new in enumerate(r.strategies):
                if i != j:
                    cols[f"trans_{old}_{new}"] = np.concatenate([[0], r.transitions[:, i, j]])
        fixed = np.isclose(r.shares.max(axis=1), 1.0)
        cols["fixed"] = fixed
        cols["fixation_generation"] = (
            r.fixation_generation if r.fixation_generation is not None else -1
        )
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def save_run(
    run_name: str,
    results: list[ReplicateResult],
    config: SimulationConfig,
    root: Path = RESULTS_DIR,
) -> Path:
    out = root / run_name
    out.mkdir(parents=True, exist_ok=True)
    results_to_frame(results).to_csv(out / "generations.csv", index=False)
    metadata = {
        "run_name": run_name,
        "git_commit": _git_commit(),
        "config": dataclasses.asdict(config),
        "seeds": {f"{r.condition}/{r.replicate}": list(r.seed) for r in results},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out


def load_run(run_name: str, root: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Return (generations frame, metadata dict). fixation_generation -1 means never fixed."""
    path = root / run_name
    return pd.read_csv(path / "generations.csv"), json.loads((path / "metadata.json").read_text())


def run_experiment(
    run_name: str,
    config: SimulationConfig,
    conditions: Iterable[str] | None = None,
    workers: int | None = None,
    root: Path = RESULTS_DIR,
) -> Path:
    """Run every condition (default: all in config) and save the results."""
    names = list(conditions) if conditions is not None else list(config.initial_conditions)
    results = [r for c in names for r in run_condition(c, config, workers)]
    return save_run(run_name, results, config, root)
