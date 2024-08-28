"""Tests for the layer class"""

import pytest
from pytest import approx
from pandas_ylt.layer import Layer, MultiLayer


@pytest.mark.parametrize(
        "layer_params,",
        [
            # limit is negative
            ({'limit': -2.0}),
        ],
)
def test_validation_error(layer_params):
    """Test we get validation errors when using bad parameters"""

    with pytest.raises(ValueError):
        Layer(**layer_params)


@pytest.mark.parametrize(
    "layer_params, agg_loss, expected",
    [
        # 2 reinstatements. 1.75 reinstatements used
        ({'limit': 2.0, 'agg_limit': 6.0, 'reinst_rate': 0.075},
         3.5, 1.75 * 2.0 * 0.075),

        # 2 reinstatements. 1.75 reinstatements used on layer with xs
        ({'limit': 2.0, 'xs': 1.0, 'agg_limit': 6.0, 'reinst_rate': 0.075},
         3.5, 1.75 * 2.0 * 0.075),

        # 2.5 reinstatements. 2.25 reinstatements used
        ({'limit': 2.0, 'agg_limit': 7.0, 'reinst_rate': 0.075},
         4.5, 2.25 * 2.0 * 0.075),

        # 2.5 reinstatements. 2.5 reinstatements used
        ({'limit': 2.0, 'agg_limit': 7.0, 'reinst_rate': 0.075},
         5.5, 2.5 * 2.0 * 0.075),

        # 3 reinstatements. 1.5 reinstatements used after agg_xs used
        ({'limit': 2.0, 'agg_limit': 6.0, 'agg_xs': 1.0, 'reinst_rate': 0.075},
         4.0, (2.0 + 1.0) * 0.075),

        # No reinstatemnts because limit is same as agg limit
        ({'limit': 2.0, 'agg_limit': 2.0, 'reinst_rate': 0.075},
         5.5, 0.0),

        # No reinstatements where limit is more than agg limit
        ({'limit': 2.0, 'agg_limit': 1.5, 'reinst_rate': 0.075},
         5.5, 0.0),

        # No reinstatements because reinst rate is zero
        ({'limit': 2.0, 'agg_limit': 4.0, 'reinst_rate': 0.0},
         5.5, 0.0),
    ],
)
def test_reinst_cost(layer_params, agg_loss, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = Layer(**layer_params)

    assert this_lyr.reinst_cost(agg_loss) == approx(expected)


@pytest.mark.parametrize(
    "limit, reinst_rates, agg_loss, expected",
    [
        # 3 reinstatements with first free. 2.75 reinstatements used
        (2.0, [0.0, 0.075, 0.075], 5.5, 2.0*0.0 + 2.0*0.075 + 1.5*0.075),
    ],
)
def test_variable_reinst_cost(limit, reinst_rates, agg_loss, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = MultiLayer.from_variable_reinst_lyr_params(
        limit, reinst_rates=reinst_rates)

    assert this_lyr.reinst_cost(agg_loss) == approx(expected)
