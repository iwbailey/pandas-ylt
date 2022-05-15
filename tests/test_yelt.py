import unittest
import os
import pandas as pd

import cattbl.yeareventloss
from cattbl import yeareventloss as yelt
from cattbl.yearloss import YearLossTable  # Import for the decorator


IFILE_TEST_YELT = os.path.join(os.path.dirname(__file__),
                               "_data",
                               "example_pareto_poisson_yelt.csv")
TEST_YELT_N_YEARS = 1e5


class TestIdentifyIndices(unittest.TestCase):
    """Test functions that identify indices"""
    def test_identify_year_col(self):
        """Test we pick out the preferred column"""
        index_names = ['Year', 'EventID']
        self.assertEqual(yelt.identify_year_col(index_names), 'Year')

    def test_identify_year_col_ambig(self):
        """Test we pick out the preferred column"""
        index_names = ['Year', 'Period', 'EventID']
        self.assertEqual(yelt.identify_year_col(index_names), 'Year')

        index_names = ['Period', 'YearIdx', 'EventID']
        self.assertEqual(yelt.identify_year_col(index_names), 'Period')

    def test_year_lower_case(self):
        """Test we can get it when lower case"""
        index_names = ['year', 'EventID']
        i = yelt.identify_year_col(index_names)
        self.assertEqual(i, 'year')


class TestCreateYELT(unittest.TestCase):
    """Test we can create a YELT from various starting points"""
    def setUp(self) -> None:

        # Example Data Frame
        self.df = pd.DataFrame({
            'Year': [1, 2, 4, 5],
            'EventID': [1, 2, 3, 4],
            'DayOfYear': [25, 60, 200, 143],
            'Loss': [10.0, 1.0, 2.0, 3.0]
        })
        self.n_yrs = 5

    def test_manually_created(self):
        """Test we can create a YELT from a series we create"""
        ds = self.df.set_index(['Year', 'EventID', 'DayOfYear'])['Loss']
        ds.attrs['n_yrs'] = self.n_yrs

        self.assertIsInstance(yelt.YearEventLossTable(ds),
                              cattbl.yeareventloss.YearEventLossTable)

    def test_from_df(self):
        """Test creation from the from_df function"""

        my_yelt = yelt.from_df(self.df, n_yrs=self.n_yrs)

        self.assertEqual(my_yelt.yel.n_yrs, self.n_yrs)

    def test_invalid_key(self):
        """Check raise an error with duplicate keys"""
        with self.assertRaises(ValueError):
            yelt.from_df(
                    pd.DataFrame({'year': [4, 4],
                                  'eventid': [3, 3],
                                  'dayofyear': [200, 200],
                                  'loss': [2.0, 3.0]}),
                                 n_yrs=5,
                                 colname_loss='loss')

    def test_from_df_with_years(self):
        # See if we can create using n_yrs as existing attribute
        df = self.df.copy()
        df.attrs['n_yrs'] = self.n_yrs
        my_yelt = yelt.from_df(df)
        self.assertEqual(my_yelt.yel.n_yrs, self.n_yrs)

    def test_from_csv(self):
        """Test we can create a YELT from the example file"""

        my_yelt = yelt.from_csv(IFILE_TEST_YELT,
                                n_yrs=TEST_YELT_N_YEARS)

        self.assertEqual(my_yelt.yel.n_yrs, TEST_YELT_N_YEARS)


class TestYELTprops(unittest.TestCase):
    """Test we can access various properties of the YELT"""
    def setUp(self) -> None:
        """Initialise test variables"""

        # Read the YELT from file
        self.test_yelt = yelt.from_csv(IFILE_TEST_YELT,
                                       n_yrs=TEST_YELT_N_YEARS)

    def test_n_yrs(self):
        """Test the number of years is okay"""
        self.assertEqual(self.test_yelt.yel.n_yrs, TEST_YELT_N_YEARS)

    def test_aal(self):
        """Test we can calculate an AAL"""

        aal = self.test_yelt.yel.aal
        self.assertGreater(aal, 0.0)
        self.assertAlmostEqual(aal,
                               self.test_yelt.sum() / TEST_YELT_N_YEARS)

    def test_freq(self):
        """Test we can calculate the frequency of a loss"""
        f0 = self.test_yelt.yel.freq0

        self.assertGreater(f0, 0.0)
        self.assertAlmostEqual(f0,
                               (self.test_yelt > 0).sum() / TEST_YELT_N_YEARS)


class TestYELTmethods(unittest.TestCase):
    """Test the various methods that act on a YELT via the accessor"""
    def setUp(self) -> None:
        """Initialize test variables"""

        # Read the YELT from file
        self.test_yelt = yelt.from_csv(IFILE_TEST_YELT,
                                       n_yrs=TEST_YELT_N_YEARS)

    def test_to_ylt(self):
        """Test we can convert to a ylt"""

        this_ylt = self.test_yelt.yel.to_ylt()

        # Check the AAL are equal
        self.assertAlmostEqual(self.test_yelt.yel.aal,
                               this_ylt.yl.aal)

    def test_to_occ_ylt(self):
        """Test we can convert to a year occurrence loss table"""

        this_ylt = self.test_yelt.yel.to_ylt(is_occurrence=True)

        # Check all values are less or equal than the annual
        agg_ylt = self.test_yelt.yel.to_ylt()
        diff_ylt = agg_ylt.subtract(this_ylt)
        self.assertGreaterEqual(diff_ylt.min(), 0.0)
        self.assertGreater(diff_ylt.max(), 0.0)

    def test_exceedance_freqs(self):
        """Test we can calculate an EEF curve"""

        eef = self.test_yelt.yel.exfreq()

        # Test the same length
        self.assertEqual(len(eef), len(self.test_yelt))

        # Test the max frequency is the same as the freq of loss
        self.assertAlmostEqual(eef.max(), self.test_yelt.yel.freq0)

        # Check all indices are matching
        self.assertTrue(self.test_yelt.index.equals(eef.index))

        # Check the probabilities are all within range
        self.assertTrue((eef > 0).all())

        # Check the frequencies are decreasing as losses increase
        diffprob = (pd.concat([self.test_yelt, eef], axis=1)
                    .sort_values('Loss')['ExFreq']
                    .diff()
                    .iloc[1:]
                    )
        self.assertTrue((diffprob <= 0.0).all())

    def test_cprob(self):
        """Test we can calculate an EEF curve"""

        cprob = self.test_yelt.yel.cprob()

        # Test the same length
        self.assertEqual(len(cprob), len(self.test_yelt))

        # Test the max prob is 1
        self.assertAlmostEqual(cprob.max(), 1.0)

        # Check all indices are matching
        self.assertTrue(self.test_yelt.index.equals(cprob.index))

        # Check the probabilities are all within range
        self.assertTrue((cprob > 0).all())

        # Check the frequencies are decreasing as losses increase
        diffprob = (pd.concat([self.test_yelt, cprob], axis=1)
                    .sort_values('Loss')['CProb']
                    .diff()
                    .iloc[1:]
                    )
        self.assertTrue((diffprob >= 0.0).all())

    def test_apply_layer(self):
        """Test a layer is applied correctly"""

        # Create a yelt
        y = yelt.from_df(pd.DataFrame({
            'year': [1, 1, 1, 1],
            'EventID': range(4),
            'dayofyear': range(1, 5),
            'loss':[5, 7, 8, 10]}),
                n_yrs=1, colname_loss='loss')

        # Test an upper limit
        self.assertTrue((y.yel.apply_layer(limit=5) == 5.0).all())

        # Test a lower threshold
        tmp = y.yel.apply_layer(xs=8)

        # Check we only get one non-zero value back
        self.assertEqual((tmp > 0.0).sum(), 1)

        # Check the loss of 10 is changed to 2
        self.assertEqual(tmp.xs(3, level='EventID').iloc[0], 2.0)

        # Check the occurrence cuts of other events
        self.assertEqual((y.yel.apply_layer(n_loss=1) > 0).sum(), 1)

        # Check all three combined
        tmp = y.yel.apply_layer(limit=2, xs=6, n_loss=2)

        # Should get only two losses
        self.assertEqual((tmp > 0).sum(), 2)

        # First loss should be 1
        self.assertEqual(tmp.xs(1, level='EventID').iloc[0], 1.0)

        # Second loss should be 2
        self.assertEqual(tmp.xs(2, level='EventID').iloc[0], 2.0)

    def test_layer_aal(self):
        """Test we can calculate the loss in range"""

        # Test calculating the loss in the full range is the same as AAL
        self.assertEqual(self.test_yelt.yel.layer_aal(),
                         self.test_yelt.yel.aal)

        # Test an upper layer reduces the loss
        loss1 = 2.0 * self.test_yelt.min()
        loss2 = 0.9 * self.test_yelt.max() - loss1
        aal_upper = self.test_yelt.yel.layer_aal(limit=loss2)
        self.assertLess(aal_upper, self.test_yelt.yel.aal)

        # Test a lower layer reduces the loss
        aal_lower = self.test_yelt.yel.layer_aal(xs=loss1)
        self.assertLess(aal_lower, self.test_yelt.yel.aal)

        # Test both lower and upper is lowest of all
        aal_mid = self.test_yelt.yel.layer_aal(xs=loss1, limit=loss2)
        self.assertLess(aal_mid, aal_upper)
        self.assertLess(aal_mid, aal_lower)

        # Test reducing the number of losses reduces the aal
        aal_mid_1loss = self.test_yelt.yel.layer_aal(xs=loss1, limit=loss2,
                                                      n_loss=1)

        self.assertLess(aal_mid_1loss, aal_mid)

    def test_severity_curve(self):
        """Test we can calculate a severity curve"""

        sevcurve = self.test_yelt.yel.to_severity_curve()

        # Max prob should be 1
        self.assertAlmostEqual(sevcurve['CProb'].max(), 1.0)

        # Min prob should be 1 / num_losses
        self.assertAlmostEqual(sevcurve['CProb'].min(), 1 / len(self.test_yelt))

        # cumul prob should always increase as loss increases
        self.assertTrue((sevcurve['Loss'].is_monotonic_increasing &
                         sevcurve['CProb'].is_monotonic_increasing),
                        msg="Expecting loss to increase as CProb increases")

    def test_ef_curve(self, keep_index=False):
        """Check the EF curve calculation"""

        # Get the EP curve
        loss_ef = self.test_yelt.yel.to_ef_curve(keep_index)

        # Check Exprob increases as Loss increases
        self.assertTrue((loss_ef['Loss'].is_monotonic_decreasing &
                         loss_ef['ExFreq'].is_monotonic_increasing),
                        msg="Expecting loss to decrease as ExpFreq increases")

        # Check index starts at zero and is unique
        self.assertIsInstance(loss_ef.index, pd.RangeIndex,
                              msg="Expecting a range index for EF curve")

    def test_max_ev_loss(self):
        """Test we can get the max event loss for any year"""

        yelt2 = self.test_yelt.yel.to_maxloss_yelt()

        # Check indices are preserved
        self.assertCountEqual(yelt2.index.names, self.test_yelt.index.names)

        # Check same values as if we compute the YLT on occurrence basis
        ylt = self.test_yelt.yel.to_ylt(is_occurrence=True)

        cmp = yelt2.rename('Loss1').to_frame().join(ylt.rename('Loss2'),
                                                    how='outer').fillna(0.0)
        vs = (cmp['Loss1'] - cmp['Loss2']).abs()
        self.assertTrue(vs.max() < 1e-8)

# TODO: Test we can handle an EEF curve with negative loss


if __name__ == '__main__':
    unittest.main()
