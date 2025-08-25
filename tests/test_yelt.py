"""Tests for the year event loss table."""
import unittest
import os
import pandas as pd

from pandas_ylt import yeareventloss as yelt
from pandas_ylt.yearloss import YearLossTable  # Import for the decorator


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
        """Set up the dataframe used in tests """
        # Example Data Frame
        self.example_yelt = pd.DataFrame({
            'Year': [1, 2, 4, 5],
            'EventID': [1, 2, 3, 4],
            'DayOfYear': [25, 60, 200, 143],
            'Loss': [10.0, 1.0, 2.0, 3.0]
        })
        self.n_yrs = 5

    def test_manually_created(self):
        """Test we can create a YELT from a series we create"""
        yelt_as_series = self.example_yelt.set_index(['Year', 'EventID', 'DayOfYear'])
        yelt_as_series = yelt_as_series['Loss']
        yelt_as_series.attrs['n_yrs'] = self.n_yrs

        self.assertIsInstance(yelt.YearEventLossTable(yelt_as_series),
                              yelt.YearEventLossTable)

    def test_invalid_key(self):
        """Check raise an error with duplicate keys"""
        with self.assertRaises(ValueError):
            idx = pd.Index({'year': [4, 4],
                                  'eventid': [3, 3],
                                  'dayofyear': [200, 200]})
            ds = pd.Series([2.0, 3.0], name='loss', index=idx)
            ds.attrs['n_yrs'] = 5

            assert ds.yel.is_valid


class TestYELTprops(unittest.TestCase):
    """Test we can access various properties of the YELT"""
    def setUp(self) -> None:
        """Initialise test variables"""

        # Read the YELT from file
        ds = pd.read_csv(IFILE_TEST_YELT)
        ds = ds.set_index(['Year', 'EventID', 'DayOfYear'])['Loss']
        ds.attrs['n_yrs'] = TEST_YELT_N_YEARS

        self.test_yelt = ds

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
        freq0 = self.test_yelt.yel.freq0

        self.assertGreater(freq0, 0.0)
        self.assertAlmostEqual(freq0,
                               (self.test_yelt > 0).sum() / TEST_YELT_N_YEARS)


class TestYELTmethods(unittest.TestCase):
    """Test the various methods that act on a YELT via the accessor"""
    def setUp(self) -> None:
        """Initialize test variables"""

        # Read the YELT from file
        ds = pd.read_csv(IFILE_TEST_YELT)
        ds = ds.set_index(['Year', 'EventID', 'DayOfYear'])['Loss']
        ds.attrs['n_yrs'] = TEST_YELT_N_YEARS
        self.test_yelt = ds

    def test_to_ylt(self):
        """Test we can convert to a ylt"""

        this_ylt = self.test_yelt.yel.to_ylt()

        # Check the AAL are equal
        self.assertAlmostEqual(self.test_yelt.yel.aal,
                               this_ylt.yl.aal)

        self.assertIsInstance(YearLossTable(this_ylt), YearLossTable)

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

    def test_ef_curve(self):
        """Check the EF curve calculation"""

        # Get the EP curve
        loss_ef = self.test_yelt.yel.to_ef_curve()

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
        loss_diff = (cmp['Loss1'] - cmp['Loss2']).abs()
        self.assertTrue(loss_diff.max() < 1e-8)

    def test_negative_losses(self):
        """Test negative losses don't lead to an error"""
        this_yelt = pd.Series([-1, 0, 0, 2, 3],
                              index=pd.MultiIndex.from_tuples([
                                  (1, 1), (2, 2), (2, 3), (4, 4), (5, 5)],
                                      names=('Year', 'EventID')))
        this_yelt.attrs['n_yrs'] = 6

        self.assertAlmostEqual(this_yelt.yel.freq0, 2 / 6)


if __name__ == '__main__':
    unittest.main()
