import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from cattbl import yelt

IFILE_TEST_YELT = None
N_YEARS = 1e5

class TestCreateYELT(unittest.TestCase):
    """Test we can create a YELT from various starting points"""
    def test_from_cols(self):
        # TODO: Test we can create a yelt
        df = yelt.from_cols(
            Year=[1, 2, 4, 5],
            EventId=[1, 2, 3, 4],
            DayOfYear=[25, 60, 200, 143],
            Loss=[10.0, 1.0, 2.0, 3.0],
            n_yrs=5
        )

        self.assertTrue(df.yelt.is_valid)

    def test_invalid_key(self):
        """Check raise an error with duplicate keys"""
        with self.assertRaises(ValueError) as cm:
            yelt.from_cols(Year=[4, 4], EventId=[3, 3], DayOfYear=[200, 200],
                           Loss=[ 2.0, 3.0], n_yrs=5)

    def test_from_df(self):
        """Test we can create a YELT from a dataframe"""
        df = pd.Series()

        self.assertTrue(df.yelt.is_valid)


    def test_from_file(self):
        """Test we can create a YELT from the example file"""
        # TODO: Test we can create a yelt from the test file
        pass


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

