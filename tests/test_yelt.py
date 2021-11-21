import unittest
import os
import pandas as pd

from cattbl import yelt

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
        # TODO: Read the YELT from file
        pass

# TODO: Test the number of years is okay

# TODO: Test we can calculate an AAL

# TODO: Test we can get

# TODO: Test we can convert to a ylt

# TODO: Test we get the EEF curve

# TODO: Test we can handle an EEF curve with negative loss


if __name__ == '__main__':
    unittest.main()
