"""Test the ylt module is working as expected"""
import unittest
from cattbl import ylt
import pandas as pd
import numpy as np


class TestYLT(unittest.TestCase):
    def setUp(self) -> None:
        """Define a basic ylt for testing"""
        self.ylt_in = pd.DataFrame({
            'Year': [1, 2, 3, 5, 7],
            'Loss': [5.0, 2.5, 7.5, 10.0, 10.0],
        })
        self.n_years = 10

    def get_default_ylt(self):
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
        self.assertTrue(ylt_series.ylt.is_valid)

    def test_calc_aal(self):
        """Test calculation of AAL"""
        ylt_series = self.get_default_ylt()

        # check we get the expected value for the AAL
        self.assertAlmostEqual(ylt_series.ylt.aal,
                               self.ylt_in['Loss'].sum() / self.n_years,
                               delta=1e-12)

    def test_prob_of_a_loss_default(self):
        """Test we calculate the right prob of a loss"""

        # Test we get expected prob for default example
        ylt_series = self.get_default_ylt()
        self.assertAlmostEqual(ylt_series.ylt.prob_of_a_loss,
                               1 - (self.ylt_in.Loss > 0).sum() / self.n_years)

    def test_prob_of_loss_with_negative(self):
        """Check we get prob only of the positive losses when negative and zeros
        are present
        """
        # Test a more complex example with negatives and zeros
        ylt2 = ylt.from_cols(year=[1, 2, 3, 4, 5], loss=[-1, 0, 0, 2, 3],
                             n_yrs=6)
        self.assertAlmostEqual(ylt2.ylt.prob_of_a_loss, 2 / 6)

    def test_cprob(self):
        """Test calculation of cumulative distribution"""
        ylt_series = self.get_default_ylt()
        cprobs = ylt_series.ylt.cprob()

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
        ecdf = ylt_series.ylt.to_ecdf()
        self.assertTrue('Loss' in ecdf.columns, msg="Expected 'Loss' column")
        self.assertTrue('CProb' in ecdf.columns, msg="Expected 'CProb' column")

        # Check the cprobs are aligned with the series
        cprobs = ylt_series.ylt.cprob()
        self.assertTrue(all([c in ecdf['CProb'].values for c in cprobs]),
                        'Expected all calculated cprobs to be in ecdf')

        # Check monotonically increasing
        self.assertTrue(ecdf['Loss'].is_monotonic_increasing &
                        ecdf['CProb'].is_monotonic_increasing)

    def test_ecdf_neg_losses(self):
        # Check a case with negative losses
        ylt2 = ylt.from_cols(year=[1, 2, 3, 4, 5], loss=[-1, 0, 0, 2, 3],
                             n_yrs=6)
        ecdf = ylt2.ylt.to_ecdf()

        # Check monotonically increasing
        self.assertTrue(ecdf['Loss'].is_monotonic_increasing &
                        ecdf['CProb'].is_monotonic_increasing)

    def test_ecdf_keep_years(self):
        """Test the ECDF returns the years when requested"""
        ylt_series = self.get_default_ylt()
        ecdf = ylt_series.ylt.to_ecdf(keep_years=False)
        self.assertFalse('Year' in ecdf.columns)

        ecdf = ylt_series.ylt.to_ecdf(keep_years=True)
        self.assertTrue('Year' in ecdf.columns)

    def test_exprob(self):
        """Test calculation of exceedance prob"""
        ylt_series = self.get_default_ylt()

        exprobs = ylt_series.ylt.exprob()

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

    def test_ep_curve(self, keep_years=False):
        """Check the EP curve calculation"""
        ylt_series = self.get_default_ylt()

        # Get the EP curve
        loss_ep = ylt_series.ylt.to_ep_curve(keep_years)

        # Check Exprob increases as Loss increases
        self.assertTrue((loss_ep['Loss'].is_monotonic_decreasing &
                         loss_ep['ExProb'].is_monotonic_increasing),
                        msg="Expecting loss to decrease as Exprob increases")

        # Check index starts at zero and is unique
        self.assertIsInstance(loss_ep.index, pd.RangeIndex,
                              msg="Expecting a range index for EP curve")

    def test_ep_curve_with_years(self):
        # Check with years that the other columns are not changed
        self.test_ep_curve(False)
        ylt_series = self.get_default_ylt()

        loss_ep = ylt_series.ylt.to_ep_curve(keep_years=False)
        loss_ep_v2 = ylt_series.ylt.to_ep_curve(keep_years=True)

        self.assertTrue(loss_ep.reset_index(drop=True).equals(
                (loss_ep_v2[loss_ep.columns]
                 .drop_duplicates()
                 .reset_index(drop=True)
                 )
        ))

        # Check the year loss combinations are the same as input
        self.assertTrue((loss_ep_v2.set_index('Year')['Loss']
                         .subtract(ylt_series, fill_value=0.0) < 1e-8).all())

    def test_rp_loss(self):
        """Check return period interpolation"""
        ylt_series = self.get_default_ylt()

        # Check the max loss is at the max return period
        self.assertEqual(ylt_series.ylt.loss_at_rp(self.n_years),
                         ylt_series.max())

        # Check we can do multiple return periods including outside of range
        retpers = range(12)
        losses = ylt_series.ylt.loss_at_rp(retpers)

        # Check same number of values returned
        self.assertEqual(len(losses), len(retpers))

        # Check no losses greater than the maximum
        self.assertEqual(np.nanmax(losses), ylt_series.max())

        # Check only the return periods less than 1 get a nan loss
        self.assertTrue(all([np.array(retpers)[np.isnan(losses)] < 1]))


if __name__ == '__main__':
    unittest.main()
