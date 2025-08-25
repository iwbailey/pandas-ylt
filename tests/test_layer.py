"""Tests for the layer class"""

import pytest
from pytest import approx
import pandas as pd
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
        "layer_params, event_loss, expected",
        [
            # Simple single limit
            ({'limit': 2.0}, 3.5, 2.0),

            # Limit and xs
            ({'limit': 2.0, 'xs': 1.0}, 2.5, 1.5),

            # Limit and xs and share
            ({'limit': 2.0, 'xs': 1.0, 'share': 0.5}, 2.5, 0.75),

            # XS and no limit
            ({'xs': 1.0}, 100, 99.0),

            # Share only
            ({'share': 0.35}, 100, 35.0),

            # Agg limit and xs, plus occ limit
            ({'agg_limit': 2.0, 'agg_xs': 1.0, 'xs': 1.0}, 2.5, 0.5),

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
    """Test we can calculate annual loss to a layer from a set of input event losses"""

    this_lyr = Layer(**layer_params)

    assert this_lyr.ceded_loss_in_year(event_loss) == approx(expected)

    # Test version that takes YELT as input
    yelt = (pd.Series(event_loss, name='Loss').rename_axis('EventId')
            .to_frame().assign(Year=1).set_index('Year', append=True)['Loss']
    )
    yelt.attrs['n_yrs'] = 1

    assert this_lyr.ceded_ylt(yelt).iloc[0] == approx(expected)


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

    event_losses = this_lyr.ceded_event_losses_in_year(event_loss)

    assert  event_losses == approx(np.array(expected))

    # Test version that takes YELT as input
    yelt = (pd.Series(event_loss, name='Loss').rename_axis('EventId')
            .to_frame().assign(Year=1).set_index('Year', append=True)['Loss']
    )
    yelt.attrs['n_yrs'] = 1

    assert this_lyr.ceded_yelt(yelt).values == approx(np.array(expected))

@pytest.mark.parametrize(
    "layer_params, event_losses, expected",
    [
        # 2 reinstatements. 1.75 reinstatements used
        ({'limit': 2.0, 'agg_limit': 6.0, 'reinst_at': 1.0},
         [2.0, 1.5], 1.75),

        # 2 reinstatements. 1.75 reinstatements used, 50% of premium
        ({'limit': 2.0, 'agg_limit': 6.0, 'reinst_at': 0.5},
         [2.0, 1.5], 1.75 * 0.5),

        # 2 reinstatements. 1.75 reinstatements used on layer with xs
        ({'limit': 2.0, 'xs': 1.0, 'agg_limit': 6.0, 'reinst_at': 1.0},
         [3.5, 2.5], 1.75),

        # 2.5 reinstatements. 2.25 reinstatements used
        ({'limit': 2.0, 'agg_limit': 7.0, 'reinst_at': 1.0},
         [2.0, 2.0, 0.5], 2.25),

        # 2.5 reinstatements. 2.5 reinstatements used
        ({'limit': 2.0, 'agg_limit': 7.0, 'reinst_at': 1.0},
         [2.0, 2.0, 1.5], 2.5),

        # 3 reinstatements. 1.5 reinstatements used after agg_xs used
        ({'limit': 2.0, 'agg_limit': 6.0, 'agg_xs': 1.0, 'reinst_at': 1.0},
         [2.0, 2.0], (0.5 + 1.0)),

        # No reinstatemnts because limit is same as agg limit
        ({'limit': 2.0, 'agg_limit': 2.0, 'reinst_at': 1.0},
         [5.5], 0.0),

        # No reinstatements where limit is more than agg limit
        ({'limit': 2.0, 'agg_limit': 1.5, 'reinst_at': 1.0},
         [5.5], 0.0),

        # No reinstatements because reinst rate is zero
        ({'limit': 2.0, 'agg_limit': 4.0, 'reinst_at': 0.0},
         [5.5], 0.0),
    ],
)
def test_reinstatements(layer_params, event_losses, expected):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    this_lyr = Layer(**layer_params)

    reinstated_limit = this_lyr.reinstated_limit_in_year(event_losses)

    assert this_lyr.paid_reinstatements(reinstated_limit) == approx(np.array(expected))

    # Test version that takes YELT as input
    yelt = (pd.Series(event_losses, name='Loss').rename_axis('EventId')
            .to_frame().assign(Year=1).set_index('Year', append=True)['Loss']
    )
    yelt.attrs['n_yrs'] = 1

    reinstated_limit_v2 = this_lyr.ceded_ylt(yelt, only_reinstated=True)
    assert this_lyr.paid_reinstatements(reinstated_limit_v2).iloc[0] == approx(expected)


@pytest.mark.parametrize(
    "layer_params, reinst_rates, event_loss, expected_loss, expected_reinst",
    [
        # 3 reinstatements with first free. 1.75 reinstatements used
        ({'limit': 2.0, 'xs': 1.0, }, [0.0, 1.0, 0.5], [3.0, 2.5], 3.5, 0.75),
        # 3 reinstatements, first free, 3 event losses using partial reinstatements
        ({'limit': 2.0, 'xs': 1.0, }, [0.0, 1.0, 1.0], [3.0, 2.5, 2.0, 1.0], 4.5, 1.25),
        # 3 reinstatements, first free, 3 event losses using all reinstatements
        ({'limit': 1.0, 'xs': 1.0, }, [0.0, 1.0, 1.0], [3.0, 2.5, 2.0, 2.0], 4.0, 2.0),
        # 2 reinstatements, second free. 3 event losses using all reinst
        ({'limit': 2.0, 'xs': 1.0, }, [1.0, 0.0], [3.0, 2.5, 2.0], 4.5, 1.0),
    ],
)
def test_variable_reinst_cost(layer_params, reinst_rates, event_loss, expected_loss,
                              expected_reinst):
    """Test we can calculate the coorect reinstatement costs for agg loss to a layer"""

    layers = variable_reinst_layer(**layer_params, reinst_at=reinst_rates)

    reinstated_losses = [l.reinstated_limit_in_year(event_loss) for l in layers]

    # Avoid double counting by taking only the reinstated part from the first layers
    losses = reinstated_losses[:-1] + [layers[-1].ceded_loss_in_year(event_loss)]



    paid_reinst = [l.paid_reinstatements(x) for l, x in zip(layers, reinstated_losses)]

    # print({'losses': losses})
    # print({'reinst losses': reinstated_losses})
    # print({'paid_reinst': paid_reinst})

    assert sum(losses) == approx(expected_loss)
    assert sum(paid_reinst) == approx(expected_reinst)


    # reinstatements = [l.paid_reinstatements for l in layers]


@pytest.mark.parametrize(
        "layer_params, event_loss, expected_loss",
        [
            # Case 1: unused limit in top layer drops down for second loss in lower layer
            ([{'limit': 1.0, 'xs': 1.0, 'agg_limit': 2.0},
              {'limit': 1.0, 'xs': 10.0, 'agg_limit': 1.0}], [11.0, 2.0], [2.0, 0.0]),
            ([{'limit': 1.0, 'xs': 1.0, 'agg_limit': 2.0},
              {'limit': 1.0, 'xs': 10.0, 'agg_limit': 1.0}], [5.0, 2.0], [1.0, 1.0]),
            ([{'limit': 1.0, 'xs': 1.0, 'agg_limit': 2.0},
              {'limit': 1.0, 'xs': 10.0, 'agg_limit': 1.0}], [10.5, 2.0], [1.5, 0.5]),
            ([{'limit': 1.0, 'xs': 1.0, 'agg_limit': 2.0},
              {'limit': 1.0, 'xs': 10.0, 'agg_limit': 1.0}], [5.0, 2.0, 11.0],
             [1.0, 1.0, 0.0]),
        ],
)
def test_top_and_drop(layer_params, event_loss, expected_loss):
    """Test drop down structures"""

    layer1 = Layer(**layer_params[0])
    layer2 = Layer(**layer_params[1])

    lyr_losses1 = layer1.ceded_event_losses_in_year(event_loss)
    lyr_losses2 = layer2.ceded_event_losses_in_year(event_loss)

    # Apply the agg limit to the combined loss
    layer3 = Layer(agg_limit=layer_params[0]['agg_limit'])
    lyr_losses3 = layer3.ceded_event_losses_in_year(lyr_losses1 + lyr_losses2)

    assert lyr_losses3 == approx(np.array(expected_loss))


@pytest.mark.parametrize(
        "layer_params, xs_dropdown, event_loss, expected_loss",
        [
            # xs reduces for the second loss but exit stays the same
            ({'limit': 1.0, 'xs': 2.0, 'agg_limit': 3.0}, 1.0,
             [3.0, 3.0], [1.0, 2.0]),
            ({'limit': 1.0, 'xs': 2.0, 'agg_limit': 3.0}, 1.0,
             [2.5, 3.0], [0.5, 2.0]),
            ({'limit': 1.0, 'xs': 2.0, 'agg_limit': 3.0}, 1.0,
             [1.5, 1.5, 2.1], [0.0, 0.0, 0.1]),
            ({'limit': 1.0, 'xs': 2.0, 'agg_limit': 3.0}, 1.0,
             [1.5, 2.1, 2.0, 2.0, 2.0], [0.0, 0.1, 1.0, 1.0, 0.9]),
        ],
)
def test_dropdown(layer_params, xs_dropdown, event_loss, expected_loss):
    """Test drop down structures"""

    # Pre-drop down layer
    layer1 = Layer(**layer_params)
    lyr_losses1 = layer1.ceded_event_losses_in_year(event_loss)

    # After drop down, before agg limit
    dropdown_params = {'xs': xs_dropdown,
                       'limit': layer_params['limit'] + (layer_params['xs'] - xs_dropdown)}

    layer2 = Layer(**dropdown_params)
    lyr_losses2 = layer2.ceded_event_losses_in_year(event_loss)

    # Take 1st pre-drop down loss and all others from the post-drop down layer
    i = np.nonzero(lyr_losses1)[0][0] + 1

    if i <= len(lyr_losses1):
        lyr_losses2[:i] = 0.0
        lyr_losses1[i:] = 0.0

    lyr_losses3 = lyr_losses1 + lyr_losses2

    # Apply the agg limit
    layer3 = Layer(agg_limit=layer_params['agg_limit'])
    lyr_losses = layer3.ceded_event_losses_in_year(lyr_losses3)

    assert lyr_losses == approx(np.array(expected_loss))
