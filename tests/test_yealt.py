"""Tests for the year event allocated loss table"""
import unittest
import os
import pandas as pd

# Import for the decorator
from pandas_ylt.yeareventallocloss import YearEventAllocLossTable  # pylint: disable=unused-import


IFILE_TEST_YELT = os.path.join(os.path.dirname(__file__),
                               "_data",
                               "example_allocated_loss.csv")
TEST_YELT_N_YEARS = 1e5


class TestYEALT(unittest.TestCase):
    """Tests for the Year Event Allocated Loss Table"""
    def setUp(self) -> None:
        """Read the example yealt"""
        example_yealt = pd.read_csv(IFILE_TEST_YELT)
        example_yealt = example_yealt.set_index([c for c in example_yealt.columns
                                                 if c != 'Loss'])['Loss']
        example_yealt.attrs['n_yrs'] = int(TEST_YELT_N_YEARS)
        # example_yealt.attrs['col_year'] = 'Year'
        example_yealt.attrs['col_event'] = ['ModelID', 'EventID', 'DayOfYear']
        self.example_yealt = example_yealt

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

        self.assertLessEqual(len(ylt), self.example_yealt.yeal.n_yrs)
        self.assertAlmostEqual(ylt.sum(), self.example_yealt.sum(), places=6)

    def test_ylt_maxocc(self):
        """Test getting the max event loss per year"""
        ylt_max = self.example_yealt.yeal.to_ylt(is_occurrence=True)
        ylt = self.example_yealt.yeal.to_ylt()

        self.assertCountEqual(ylt_max.index, ylt.index)
        self.assertTrue((ylt >= ylt_max).all())
        self.assertTrue(ylt.sum() != ylt_max.sum())

    def test_yalt_aep(self):
        """Test year allocated loss table"""
        yalt = self.example_yealt.yeal.to_yalt()

        ylt = self.example_yealt.yeal.to_ylt()

        check_allocation = (yalt.rename('Alloc').to_frame()
                            .join(ylt)
                            .assign(ppn=lambda df: df['Alloc'] / df['Loss']))

        self.assertLess((check_allocation.groupby('Year')['ppn'].sum() - 1)
                        .abs().max(),
                        1e-12)

    def test_yalt_oep(self):
        """Test year allocated loss table"""
        yalt = self.example_yealt.yeal.to_yalt(is_occurrence=True)

        ylt = self.example_yealt.yeal.to_ylt(is_occurrence=True)

        check_allocation = (yalt.rename('Alloc').to_frame()
                            .join(ylt)
                            .assign(ppn=lambda df: df['Alloc'] / df['Loss']))

        self.assertLess((check_allocation.groupby('Year')['ppn'].sum() - 1)
                        .abs().max(),
                        1e-12)


if __name__ == '__main__':
    unittest.main()
