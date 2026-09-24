import pytest

from dmas.strategies.equity import EQUITY_TABLE, equity


@pytest.mark.parametrize(
    ("private_rank", "public_rank", "expected"),
    [
        # J
        (0, None, 0.30),
        (0, 0, 1.0),
        (0, 1, 0.125),
        (0, 2, 0.125),

        # Q
        (1, None, 0.50),
        (1, 0, 0.125),
        (1, 1, 1.0),
        (1, 2, 0.625),

        # K
        (2, None, 0.70),
        (2, 0, 0.625),
        (2, 1, 0.625),
        (2, 2, 1.0),
    ],
)
def test_equity(private_rank, public_rank, expected):
    assert equity(private_rank, public_rank) == pytest.approx(expected)


def test_equity_table_has_all_12_states():
    assert len(EQUITY_TABLE) == 12