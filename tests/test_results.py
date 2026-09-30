import numpy as np

from dmas.simulation.results import load_run, run_experiment
from dmas.simulation.runner import SimulationConfig

CONFIG = SimulationConfig(n=20, generations=4, replicates=2, k=2, beta=16.0)


def test_run_saves_and_loads_in_one_call(tmp_path):
    run_experiment("t", CONFIG, conditions=["balanced", "tag_majority"], workers=1, root=tmp_path)
    df, meta = load_run("t", root=tmp_path)

    assert len(df) == 2 * 2 * (CONFIG.generations + 1)
    assert set(df.condition) == {"balanced", "tag_majority"}
    shares = df[[c for c in df if c.startswith("share_")]]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    trans = df[[c for c in df if c.startswith("trans_")]]
    np.testing.assert_array_equal(trans.sum(axis=1), df.switches)
    assert (df[df.generation == 0].switches == 0).all()
    assert meta["config"]["n"] == 20
    assert meta["seeds"]["balanced/0"] != meta["seeds"]["balanced/1"]
    assert "git_commit" in meta


def test_fixed_flag_matches_shares(tmp_path):
    run_experiment("t", CONFIG, conditions=["balanced"], workers=1, root=tmp_path)
    df, _ = load_run("t", root=tmp_path)
    share_max = df[[c for c in df if c.startswith("share_")]].max(axis=1)
    assert (df.fixed == np.isclose(share_max, 1.0)).all()
