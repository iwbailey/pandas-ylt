"""Tests for the year event allocated loss table"""
import unittest
import os
import pandas as pd

from cattbl.yeareventallocloss import YearEventAllocLossTable  # Import for the decorator


IFILE_TEST_YELT = os.path.join(os.path.dirname(__file__),
                               "_data",
                               "example_allocated_loss.csv")
TEST_YELT_N_YEARS = 1e5


class TestYEALT(unittest.TestCase):
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
        """Check if the class can find the correct year index"""
        self.assertEqual(self.example_yealt.yeal.col_year, 'Year')

    def test_col_event(self):
        """Check that the event columns are picked up"""
        self.assertCountEqual(self.example_yealt.yeal.col_event,
                              ['ModelID', 'EventID', 'DayOfYear'])

    def test_subset(self):
        """Test we can extract a subset of the table"""
        yealt2 = self.example_yealt.yeal.to_subset(ModelID='Model1',
                                                   RegionID=(1, 2),
                                                   LossSourceID=1)

        yealt2 = yealt2.reset_index()
        self.assertCountEqual(yealt2.ModelID.unique(), ['Model1'])
        self.assertCountEqual(yealt2.RegionID.unique(), [1, 2])
        self.assertCountEqual(yealt2.LossSourceID.unique(), [1])

        # Check the original YELT is unchanged
        self.assertGreater(len(self.example_yealt), len(yealt2))

    def test_ylt(self):
        """Test we can extract a YLT"""
        ylt = self.example_yealt.yeal.to_ylt()

        print(ylt.sum(), self.example_yealt.sum())

if __name__ == '__main__':
    unittest.main()
