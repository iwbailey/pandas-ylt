"""Test the loss series under base class"""
import pandas as pd
import pytest  # noqa # pylint: disable=unused-import
import pandas_ylt.base_classes  # noqa # pylint: disable=unused-import


def test_set_n_yrs():
    """Test that we can add n_yrs to a series in line"""
    ds = pd.Series([1, 1, 1, 1], index=[1, 2, 3, 4]).ls.set_n_yrs(5)

    assert ds.ls.n_yrs == 5


def test_set_n_yrs_attrs():
    """Test that setting n_yrs will add an attr called n_yrs"""
    ds = pd.Series([1, 1, 1, 1], index=[1, 2, 3, 4]).ls.set_n_yrs(5)
    assert ds.attrs['n_yrs'] == 5


def test_aal():
    """Test that we can calculate the correct AAL"""
    ds = pd.Series([1, 1, 1, 1], index=[1, 2, 3, 4]).ls.set_n_yrs(5)

    assert ds.ls.aal == 4 / 5


def test_col_loss():
    """test we can get the name of the loss column"""
    ds = pd.Series([1, 1, 1, 1], index=[1, 2, 3, 4], name='loss'
                   ).ls.set_n_yrs(5)

    assert ds.ls.col_loss == 'loss'


def test_empty_col_loss():
    """test we can get the name of the loss column"""
    ds = pd.Series([1, 1, 1, 1], index=[1, 2, 3, 4]
                   ).ls.set_n_yrs(5)

    assert ds.ls.col_loss is None


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
        ({'limit': 5.0, 'xs': 8, 'is_franchise': True}, 14.0, 13.0),
        ({'limit': 5.0, 'xs': 8, 'share': 0.5}, 10.0, 1.0),
        ({'limit': 5.0, 'xs': 8, 'share': 0.2}, 14.0, 1.0),
        ({'limit': 5.0, 'xs': 8, 'share': 0.5, 'is_franchise': True}, 12.0, 6.0),
        ({'limit': 5.0, 'xs': 8, 'share': 0.5, 'is_step': True}, 12.0, 0.5),
        ({'is_step': True}, 12.0, 1.0),
        ({'xs': 10, 'share': 100, 'is_step': True}, 12.0, 100.0),
        ({'xs': 10, 'share': 100, 'is_step': True}, 9.0, 0.0),
    ])
def test_apply_layer(layer_params, loss, expected):
    """Test a layer is applied correctly"""

    # Create a yelt
    this_yelt = pd.Series([loss], name='loss').ls.set_n_yrs(5)

    assert this_yelt.ls.apply_layer(**layer_params).iloc[0] == expected
