"""Test the ylt module is working as expected"""
import unittest
import pytest  # noqa # pylint: disable=unused-import

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from pandas_ylt import yearloss as ylt


def test_initialise_min_requiremnts():
    """Test set up from a simple pandas series """
    ds = pd.Series([1, 1, 1, 1], index=[1, 2, 3, 4]).yl.set_n_yrs(5)

    assert ds.yl.is_valid

class TestYLT(unittest.TestCase):
    """Tests on the YearLossTable"""
    def setUp(self) -> None:
        """Define a basic ylt for testing"""
        self.ylt_in = pd.DataFrame({
            'Year': [1, 2, 3, 5, 7],
            'Loss': [5.0, 2.5, 7.5, 10.0, 10.0],
        })
        self.n_years = 10

    def get_default_ylt(self):
        """Return the default YLT"""
        ylt_series = ylt.from_cols(
            year=self.ylt_in['Year'].values,
            loss=self.ylt_in['Loss'].values,
            n_yrs=self.n_years,
        )
        return ylt_series

    def test_from_cols(self):
        """Create a ylt"""
        ylt_series = self.get_default_ylt()

        # Check we got a series back
        self.assertIsInstance(ylt_series, pd.Series, msg="Expected series")

        # Check we stored the years as an attribute
        self.assertIn('n_yrs', ylt_series.attrs.keys(),
                      msg="Expected num years in attrs")

        # Check we pass the validation checks
        tmp = ylt.YearLossTable(ylt_series)
        self.assertIsInstance(tmp, ylt.YearLossTable)

    def test_calc_aal(self):
        """Test calculation of AAL"""
        ylt_series = self.get_default_ylt()

        # check we get the expected value for the AAL
        self.assertAlmostEqual(ylt_series.yl.aal,
                               self.ylt_in['Loss'].sum() / self.n_years,
                               delta=1e-12)

    def test_calc_std(self):
        """Test calculation of standard deviation"""

        # Reference with the zero loss years included
        test_ref = pd.Series([1, 2, 0, 4, 0, 6, 7, 0, 9, 10])

        # Make an equivalent ylt with zero loss years not included
        test_ylt = test_ref.copy()
        test_ylt.index = pd.Index(range(1, 11), name='Year')
        test_ylt.name = 'Loss'
        test_ylt.attrs['n_yrs'] = 10
        test_ylt = test_ylt.loc[test_ylt> 0]

        # Check we get the same standard deviation back
        self.assertEqual(test_ref.std(), test_ylt.yl.std())

    def test_prob_of_a_loss_default(self):
        """Test we calculate the right prob of a loss"""

        # Test we get expected prob for default example
        ylt_series = self.get_default_ylt()
        self.assertAlmostEqual(ylt_series.yl.prob_of_a_loss,
                               1 - (self.ylt_in.Loss > 0).sum() / self.n_years)

    def test_prob_of_loss_with_negative(self):
        """Check we get prob only of the positive losses when negative and zeros
        are present
        """
        # Test a more complex example with negatives and zeros
        ylt2 = ylt.from_cols(year=[1, 2, 3, 4, 5], loss=[-1, 0, 0, 2, 3],
                             n_yrs=6)
        self.assertAlmostEqual(ylt2.yl.prob_of_a_loss, 2 / 6)

    def test_cprob(self):
        """Test calculation of cumulative distribution"""
        ylt_series = self.get_default_ylt()
        cprobs = ylt_series.yl.cprob()

        # Check no change in series length
        self.assertEqual(len(ylt_series), len(cprobs),
                         msg="Expected series length to remain unchanged")

        # Check all > 0
        self.assertTrue((cprobs > 0).all(),
                        msg="Expected all probabilities to be >0")

        # Check it goes up to 1.0
        self.assertAlmostEqual(cprobs.max(), 1.0, delta=1e-8,
                               msg="Expected max cumulative prob to be 1.0")

        # Check it is aligned with the losses
        diffprob = (pd.concat([ylt_series, cprobs], axis=1)
                    .sort_values('Loss')['CProb']
                    .diff()
                    .iloc[1:]
                    )
        self.assertTrue((diffprob >= 0.0).all(),
                        msg="Cumul probs don't increase as loss increases")

    def test_calc_ecdf(self):
        """Test calculation of the empirical cdf"""
        ylt_series = self.get_default_ylt()

        # Check the columns are there
        ecdf = ylt_series.yl.to_ecdf()
        self.assertTrue('Loss' in ecdf.columns, msg="Expected 'Loss' column")
        self.assertTrue('CProb' in ecdf.columns, msg="Expected 'CProb' column")

        # Check the cprobs are aligned with the series
        cprobs = ylt_series.yl.cprob()
        self.assertTrue(all((c in ecdf['CProb'].values for c in cprobs)),
                        'Expected all calculated cprobs to be in ecdf')

        # Check monotonically increasing
        self.assertTrue(ecdf['Loss'].is_monotonic_increasing &
                        ecdf['CProb'].is_monotonic_increasing)

    def test_ecdf_neg_losses(self):
        """Check a case with negative losses"""
        ylt2 = ylt.from_cols(year=[1, 2, 3, 4, 5], loss=[-1, 0, 0, 2, 3],
                             n_yrs=6)
        ecdf = ylt2.yl.to_ecdf()

        # Check monotonically increasing
        self.assertTrue(ecdf['Loss'].is_monotonic_increasing &
                        ecdf['CProb'].is_monotonic_increasing)

    def test_exprob(self):
        """Test calculation of exceedance prob"""
        ylt_series = self.get_default_ylt()

        exprobs = ylt_series.yl.exprob()

        # Check they are the same length
        self.assertEqual(len(ylt_series), len(exprobs))

        # Check all indices are matching
        self.assertTrue(ylt_series.index.equals(exprobs.index))

        # Check the probabilities are all within range
        self.assertTrue((exprobs > 0).all() & (exprobs <= 1.0).all())

        # Check the exprobs are decreasing as losses increase
        diffprob = (pd.concat([ylt_series, exprobs], axis=1)
                    .sort_values('Loss')['ExProb']
                    .diff()
                    .iloc[1:]
                    )
        self.assertTrue((diffprob <= 0.0).all())

    def test_exprob_with_dup_losses(self):
        """Test we pick out the largest exceedance prob for duplicate losses"""

        this_ylt = ylt.from_cols(year=[1, 2, 3, 4, 5], loss=[1, 1, 1, 2, 2],
                                 n_yrs=8)

        # 3 years should get 5/8 and 2 years should get 2/8
        expected = pd.Series([5, 5, 5, 2, 2], index=[1, 2, 3, 4, 5]) / 8
        self.assertTrue(expected.equals(this_ylt.yl.exprob()))

        # Ex curve should get rid of the duplicates
        expected = pd.DataFrame({'Loss': [2., 1., 0.], 'ExProb': np.array([2, 5, 8]) / 8})
        self.assertTrue(expected.equals(this_ylt.yl.to_ep_curve()))

    def test_ep_curve(self):
        """Check the EP curve calculation"""
        ylt_series = self.get_default_ylt()

        # Get the EP curve
        loss_ep = ylt_series.yl.to_ep_curve()

        # Check Exprob increases as Loss increases
        self.assertTrue((loss_ep['Loss'].is_monotonic_decreasing &
                         loss_ep['ExProb'].is_monotonic_increasing),
                        msg="Expecting loss to decrease as Exprob increases")

        # Check index starts at zero and is unique
        self.assertIsInstance(loss_ep.index, pd.RangeIndex,
                              msg="Expecting a range index for EP curve")

    def test_loss_at_exprobs(self):
        """Check getting the loss for specific exceedance probabilities"""
        ylt_series = self.get_default_ylt()

        # Check the max loss is at the max return period
        self.assertEqual(ylt_series.yl.loss_at_exprobs([1 / self.n_years]),
                         ylt_series.max())

        # Check we can do multiple return periods including outside of range
        exprobs = 1 / np.arange(1, 13, 1)
        losses = ylt_series.yl.loss_at_exprobs(exprobs)

        expected = np.array([0., 2.5, 5., 7.5, 10., 10., 10., 10., 10., 10., np.nan, np.nan])

        assert_array_equal(losses, expected)


if __name__ == '__main__':
    unittest.main()
