"""Tests for the layer class"""

import pytest
from pytest import approx
from pandas_ylt.layer import Layer


@pytest.mark.parametrize(
    "limit, agg_limit, reinst_rates, agg_loss, expected",
    [
        # 2 reinstatements at a fixed cost. 1.75 reinstatements used
        (2.0, 6.0, 0.075, 3.5, 0.2625),
        # 2.5 reinstatements at a fixed cost. 2.25 reinstatements used
        (2.0, 7.0, 0.075, 4.5, 0.3375),
    ],
)
def test_reinst_cost(limit, agg_limit, reinst_rates, agg_loss, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = Layer(limit, agg_limit=agg_limit, reinst_rate=reinst_rates)

    assert this_lyr.reinst_cost(agg_loss) == approx(expected)
