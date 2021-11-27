import unittest
import os
import pandas as pd

from cattbl import yelt
from cattbl.ylt import YearLossTable

IFILE_TEST_YELT = os.path.join(os.path.dirname(__file__),
                               "_data",
                               "example_pareto_poisson_yelt.csv")
TEST_YELT_N_YEARS = 1e5


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

        self.assertTrue(ds.yelt.is_valid)

    def test_from_cols(self):
        """Test creation from the from_cols function"""
        df = yelt.from_cols(
            year=self.df.Year.values,
            eventid=self.df.EventID.values,
            dayofyear=self.df.DayOfYear.values,
            loss=self.df.Loss.values,
            n_yrs=self.n_yrs
        )

        self.assertTrue(df.yelt.is_valid)

    def test_invalid_key(self):
        """Check raise an error with duplicate keys"""
        with self.assertRaises(ValueError):
            yelt.from_cols(year=[4, 4], eventid=[3, 3], dayofyear=[200, 200],
                           loss=[2.0, 3.0], n_yrs=5)

    def test_from_df(self):
        """Test creation from the from_df function"""

        my_yelt = yelt.from_df(self.df, n_yrs=self.n_yrs)

        self.assertTrue(my_yelt.yelt.is_valid)

    def test_from_df_with_years(self):
        # See if we can create using n_yrs as existing attribute
        df = self.df.copy()
        df.attrs['n_yrs'] = self.n_yrs
        my_yelt = yelt.from_df(df)
        self.assertTrue(my_yelt.yelt.is_valid)

    def test_from_csv(self):
        """Test we can create a YELT from the example file"""

        my_yelt = yelt.from_csv(IFILE_TEST_YELT,
                                n_yrs=TEST_YELT_N_YEARS)

        self.assertTrue(my_yelt.yelt.is_valid)


class TestYELTprops(unittest.TestCase):
    """Test we can access various properties of the YELT"""
    def setUp(self) -> None:
        """Initialise test variables"""

        # Read the YELT from file
        self.test_yelt = yelt.from_csv(IFILE_TEST_YELT,
                                       n_yrs=TEST_YELT_N_YEARS)

    def test_n_yrs(self):
        """Test the number of years is okay"""
        self.assertEqual(self.test_yelt.yelt.n_yrs, TEST_YELT_N_YEARS)

    def test_aal(self):
        """Test we can calculate an AAL"""

        aal = self.test_yelt.yelt.aal
        self.assertGreater(aal, 0.0)
        self.assertAlmostEqual(aal,
                               self.test_yelt.sum() / TEST_YELT_N_YEARS)

    def test_freq(self):
        """Test we can calculate the frequency of a loss"""
        f0 = self.test_yelt.yelt.freq0

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

        this_ylt = self.test_yelt.yelt.to_ylt()

        # Check it is a valid YLT
        self.assertTrue(this_ylt.ylt.is_valid)

        # Check the AAL are equal
        self.assertAlmostEqual(self.test_yelt.yelt.aal,
                               this_ylt.ylt.aal)

    def test_to_occ_ylt(self):
        """Test we can convert to a year occurrence loss table"""

        this_ylt = self.test_yelt.yelt.to_ylt(is_occurrence=True)

        # Check it is a valid YLT
        self.assertTrue(this_ylt.ylt.is_valid)

        # Check all values are less or equal than the annual
        agg_ylt = self.test_yelt.yelt.to_ylt()
        diff_ylt = agg_ylt.subtract(this_ylt)
        self.assertGreaterEqual(diff_ylt.min(), 0.0)
        self.assertGreater(diff_ylt.max(), 0.0)

    def test_exceedance_freqs(self):
        """Test we can calculate an EEF curve"""

        eef = self.test_yelt.yelt.exfreq()

        # Test the same length
        self.assertEqual(len(eef), len(self.test_yelt))

        # Test the max frequency is the same as the freq of loss
        self.assertAlmostEqual(eef.max(), self.test_yelt.yelt.freq0)

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

    def test_apply_layer(self):
        """Test a layer is applied correctly"""

        # Create a yelt
        y = yelt.from_cols(year=[1, 1, 1, 1], eventid=range(4),
                           dayofyear=range(1, 5), loss=[5, 7, 8, 10], n_yrs=1)

        # Test an upper limit
        self.assertTrue((y.yelt.apply_layer(limit=5) == 5.0).all())

        # Test a lower threshold
        tmp = y.yelt.apply_layer(xs=8)

        # Check we only get one non-zero value back
        self.assertEqual((tmp > 0.0).sum(), 1)

        # Check the loss of 10 is changed to 2
        self.assertEqual(tmp.xs(3, level='EventID').iloc[0], 2.0)

        # Check the occurrence cuts of other events
        self.assertEqual((y.yelt.apply_layer(n_loss=1) > 0).sum(), 1)

        # Check all three combined
        tmp = y.yelt.apply_layer(limit=2, xs=6, n_loss=2)

        # Should get only two losses
        self.assertEqual((tmp > 0).sum(), 2)

        # First loss should be 1
        self.assertEqual(tmp.xs(1, level='EventID').iloc[0], 1.0)

        # Second loss should be 2
        self.assertEqual(tmp.xs(2, level='EventID').iloc[0], 2.0)

    def test_layer_aal(self):
        """Test we can calculate the loss in range"""

        # Test calculating the loss in the full range is the same as AAL
        self.assertEqual(self.test_yelt.yelt.layer_aal(),
                         self.test_yelt.yelt.aal)

        # Test an upper layer reduces the loss
        loss1 = 2.0 * self.test_yelt.min()
        loss2 = 0.9 * self.test_yelt.max() - loss1
        aal_upper = self.test_yelt.yelt.layer_aal(limit=loss2)
        self.assertLess(aal_upper, self.test_yelt.yelt.aal)

        # Test a lower layer reduces the loss
        aal_lower = self.test_yelt.yelt.layer_aal(xs=loss1)
        self.assertLess(aal_lower, self.test_yelt.yelt.aal)

        # Test both lower and upper is lowest of all
        aal_mid = self.test_yelt.yelt.layer_aal(xs=loss1, limit=loss2)
        self.assertLess(aal_mid, aal_upper)
        self.assertLess(aal_mid, aal_lower)

        # Test reducing the number of losses reduces the aal
        aal_mid_1loss = self.test_yelt.yelt.layer_aal(xs=loss1, limit=loss2,
                                                      n_loss=1)

        self.assertLess(aal_mid_1loss, aal_mid)

    def test_severity_curve(self):
        """Test we can calculate a severity curve"""

        # Max prob should be 1

        # Min prob should be 1 / num_losses

        # cumul prob should always increase as loss increases

        pass

    def test_ef_curve(self, keep_index=False):
        """Check the EF curve calculation"""

        # Get the EP curve
        loss_ef = self.test_yelt.yelt.to_ef_curve(keep_index)

        # Check Exprob increases as Loss increases
        self.assertTrue((loss_ef['Loss'].is_monotonic_decreasing &
                         loss_ef['ExFreq'].is_monotonic_increasing),
                        msg="Expecting loss to decrease as ExpFreq increases")

        # Check index starts at zero and is unique
        self.assertIsInstance(loss_ef.index, pd.RangeIndex,
                              msg="Expecting a range index for EF curve")


# TODO: Test we can handle an EEF curve with negative loss

# TODO: Test severity


if __name__ == '__main__':
    unittest.main()
