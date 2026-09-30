import numpy as np

from dmas.simulation.runner import SimulationConfig, run_condition, run_replicate

CONFIG = SimulationConfig(n=20, generations=5, replicates=4, k=2, beta=16.0)
CONDITION = "balanced"


def assert_same(a, b):
    assert a.seed == b.seed
    np.testing.assert_allclose(a.shares, b.shares)
    np.testing.assert_array_equal(a.switches, b.switches)
    np.testing.assert_array_equal(a.transitions, b.transitions)
    assert a.fixation_generation == b.fixation_generation


def test_same_seed_gives_identical_trajectory():
    assert_same(run_replicate(CONDITION, 0, CONFIG), run_replicate(CONDITION, 0, CONFIG))


def test_different_replicates_differ():
    a = run_replicate(CONDITION, 0, CONFIG)
    b = run_replicate(CONDITION, 1, CONFIG)
    assert a.seed != b.seed
    assert not np.allclose(a.shares, b.shares)


def test_different_conditions_get_different_seeds():
    assert (
        run_replicate("balanced", 0, CONFIG).seed != run_replicate("tag_majority", 0, CONFIG).seed
    )


def test_results_independent_of_worker_count():
    serial = run_condition(CONDITION, CONFIG, workers=1)
    parallel = run_condition(CONDITION, CONFIG, workers=2)
    assert len(serial) == len(parallel) == CONFIG.replicates
    for a, b in zip(serial, parallel, strict=True):
        assert_same(a, b)
