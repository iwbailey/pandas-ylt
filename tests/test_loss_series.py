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
