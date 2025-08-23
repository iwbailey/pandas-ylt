"""Tests for the layer class"""

import pytest
from pytest import approx
import numpy as np
from pandas_ylt.layer import apply_layer, Layer, variable_reinst_layer


@pytest.mark.parametrize(
    "layer_params, loss, expected",
    [
        ({'limit': 5.0}, 4.0, 4.0),
        ({'limit': 5.0}, 5.0, 5.0),
        ({'limit': 5.0}, 7.0, 5.0),
        ({'limit': 5.0, 'xs': 8}, 4.0, 0.0),
        ({'limit': 5.0, 'xs': 8}, 8.0, 0.0),
        ({'limit': 5.0, 'xs': 8}, 10.0, 2.0),
        ({'limit': 5.0, 'xs': 8}, 14.0, 5.0),
        ({'limit': 5.0, 'xs': 8, 'is_franchise': True}, 14.0, 5.0),
        ({'limit': 10.0, 'xs': 8, 'is_franchise': True}, 9.0, 9.0),
        ({'limit': 10.0, 'xs': 8, 'is_franchise': True}, 14.0, 10.0),
        ({'limit': 5.0, 'xs': 8, 'share': 0.5}, 10.0, 1.0),
        ({'limit': 5.0, 'xs': 8, 'share': 0.2}, 14.0, 1.0),
        ({'limit': 13.0, 'xs': 8, 'share': 0.5, 'is_franchise': True}, 12.0, 6.0),
        ({'limit': 13.0, 'xs': 8, 'share': 0.5, 'is_step': True}, 12.0, 0.5),
        ({'is_step': True}, 12.0, 1.0),
        ({'xs': 10, 'share': 100, 'is_step': True}, 12.0, 100.0),
        ({'xs': 10, 'share': 100, 'is_step': True}, 9.0, 0.0),
    ])
def test_apply_layer(layer_params, loss, expected):
    """Test a layer is applied correctly"""

    assert apply_layer(loss, **layer_params) == expected


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
        ({'limit': 2.0, 'agg_limit': 6.0, 'premium': 0.15, 'reinst_at': 1.0},
         3.5, 1.75 * 0.15),

        # 2 reinstatements. 1.75 reinstatements used on layer with xs
        ({'limit': 2.0, 'xs': 1.0, 'agg_limit': 6.0, 'premium': 0.15, 'reinst_at': 1.0},
         3.5, 1.75 * 0.15),

        # 2.5 reinstatements. 2.25 reinstatements used
        ({'limit': 2.0, 'agg_limit': 7.0, 'premium': 0.15, 'reinst_at': 1.0},
         4.5, 2.25 * 2.0 * 0.075),

        # 2.5 reinstatements. 2.5 reinstatements used
        ({'limit': 2.0, 'agg_limit': 7.0, 'premium': 0.15, 'reinst_at': 1.0},
         5.5, 2.5 * 2.0 * 0.075),

        # 3 reinstatements. 1.5 reinstatements used after agg_xs used
        ({'limit': 2.0, 'agg_limit': 6.0, 'agg_xs': 1.0, 'premium': 0.15, 'reinst_at': 1.0},
         4.0, (2.0 + 1.0) * 0.075),

        # No reinstatemnts because limit is same as agg limit
        ({'limit': 2.0, 'agg_limit': 2.0, 'premium': 0.15, 'reinst_at': 1.0},
         5.5, 0.0),

        # No reinstatements where limit is more than agg limit
        ({'limit': 2.0, 'agg_limit': 1.5, 'premium': 0.15, 'reinst_at': 1.0},
         5.5, 0.0),

        # No reinstatements because reinst rate is zero
        ({'limit': 2.0, 'agg_limit': 4.0, 'reinst_at': 0.0},
         5.5, 0.0),
    ],
)
def test_reinst_cost(layer_params, agg_loss, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = Layer(**layer_params)

    assert this_lyr.reinst_cost(agg_loss) == approx(expected)


@pytest.mark.parametrize(
    "layer_params, event_loss, expected",
    [
        # Simple single limit
        ({'limit': 2.0}, [3.5], 2.0),

        # Limit and xs
        ({'limit': 2.0, 'xs': 1.0}, [2.5], 1.5),

        # Limit and xs and share
        ({'limit': 2.0, 'xs': 1.0, 'share': 0.5}, [2.5], 0.75),

        # XS and no limit
        ({'xs': 1.0}, [100], 99.0),

        # Share only
        ({'share': 0.35}, [100], 35.0),

        # Agg limit and xs, plus occ limit
        ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, [2.5], 0.5),

        # Agg limit and xs, plus occ limit with prior agg loss reducing agg limit
        ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, [2.0, 3.0], 2.0),

        # Agg limit and xs, plus occ limit with prior agg loss eroding agg xs
        ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, [1.5, 3.0], 1.5),

        # Agg limit already used up
        ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, [4.0, 3.0], 2.0),

        # Agg limit and agg xs with no layer limit, single event uses all loss
        ({'agg_limit': 2.0, 'agg_xs': 5.0}, [7.0], 2.0),
    ],
)
def test_layer_loss(layer_params, event_loss, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = Layer(**layer_params)

    assert this_lyr.ceded_loss_in_year(event_loss) == approx(expected)


@pytest.mark.parametrize(
    "layer_params, event_loss, expected",
    [

        # Agg limit and xs, plus occ limit with prior agg loss reducing agg limit
        ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, [2.0, 3.0], [0.0, 2.0]),

        # Agg limit and xs, plus occ limit with prior agg loss eroding agg xs
        ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, [1.5, 3.0], [0.0, 1.5]),

        # Agg limit already used up
        ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, [4.0, 3.0], [2.0, 0.0]),

        # Agg limit and agg xs with no layer limit, single event uses all loss
        ({'agg_limit': 2.0, 'agg_xs': 5.0}, [7.0, 6.0], [2.0, 0.0]),
    ],
)
def test_layer_event_loss(layer_params, event_loss, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = Layer(**layer_params)

    assert this_lyr.ceded_event_losses_in_year(event_loss) == approx(np.array(expected))


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
