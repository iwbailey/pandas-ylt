"""Tests for the year event allocated loss table"""
import unittest
import os
import pandas as pd

from cattbl.yeareventallocloss import YearEventAllocLossTable  # Import for the decorator


IFILE_TEST_YELT = os.path.join(os.path.dirname(__file__),
                               "_data",
                               "example_allocated_loss.csv")
TEST_YELT_N_YEARS = 1e5


class TestCreateYEALT(unittest.TestCase):
    def setUp(self) -> None:
        """Read the example yealt"""
        df = pd.read_csv(IFILE_TEST_YELT)
        df = df.set_index([c for c in df.columns if c != 'Loss'])['Loss']
        df.attrs['n_yrs'] = int(TEST_YELT_N_YEARS)
        # df.attrs['col_year'] = 'Year'
        df.attrs['col_event'] =['ModelID', 'EventID', 'DayOfYear']
        self.example_yealt = df

    def test_validate_example(self):
        """Check if the example is a valid yealt"""
        self.assertTrue(self.example_yealt.yeal.is_valid)

    def test_col_year(self):
        self.assertEqual(self.example_yealt.yeal.col_year, 'Year')

    def test_col_event(self):
        print(self.example_yealt.yeal.col_event)

if __name__ == '__main__':
    unittest.main()
