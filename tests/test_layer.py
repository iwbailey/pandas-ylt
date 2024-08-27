"""Tests for the layer class"""

import pytest
from pytest import approx
from pandas_ylt.layer import Layer


@pytest.mark.parametrize(
    "limit, agg_limit, reinst_rates, agg_loss, expected",
    [
        # 2 reinstatements at a fixed cost. 1.75 reinstatements used
        (2.0, 6.0, 0.075, 3.5, 1.75 * 2.0 * 0.075),

        # 2.5 reinstatements at a fixed cost. 2.25 reinstatements used
        (2.0, 7.0, 0.075, 4.5, 2.25 * 2.0 * 0.075),

        # 2.5 reinstatements at a fixed cost. 2.5 reinstatements used
        (2.0, 7.0, 0.075, 5.5, 2.5 * 2.0 * 0.075),

        # 2.5 reinstatements with first free. 2.5 reinstatements used
        (2.0, 7.0, [0.0, 0.075, 0.075], 5.5, 0.0 + 1.5 * 2.0 * 0.075),

        # No reinstatemnts because limit is same as agg limit
        (2.0, 2.0, 0.0, 5.5, 0.0),

        # No reinstatements where limit is more than agg limit
        (2.0, 1.5, 0.0, 5.5, 0.0),
    ],
)
def test_reinst_cost(limit, agg_limit, reinst_rates, agg_loss, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = Layer(limit, agg_limit=agg_limit, reinst_rate=reinst_rates)

    assert this_lyr.reinst_cost(agg_loss) == approx(expected)



@pytest.mark.parametrize(
        "layer_params,",
        [
            # Agg limit is 3.5 * limit, but only 2 reinstatements specified
            ({'limit': 2.0, 'agg_limit': 7.0, 'reinst_rate': [0.0, 0.075]}),

            # Agg limit is 3.5 * limit, but 4 reinstatements specified
            ({'limit': 2.0, 'agg_limit': 7.0,
              'reinst_rate': [0.0, 0.075, 0.075, 0.0755]}),
        ],
)
def test_validation_error(layer_params):
    """Test we get validation errors when using bad parameters"""

    with pytest.raises(ValueError):
        Layer(**layer_params)
