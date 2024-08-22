"""Module for working with year event loss table, where event loss is allocated.
"""
from collections.abc import Iterable
import pandas as pd
from pandas_ylt.yearloss import VALID_YEAR_COLNAMES_LC
from pandas_ylt.base_classes import LossSeries


@pd.api.extensions.register_series_accessor("yeal")
class YearEventAllocLossTable(LossSeries):
    """A more granular version of a YELT where the event loss is allocated.

    The series should have an attribute called 'col_event' that contains a list
    of which index columns define a single event.

    The series can have an attribute called 'col_year' that defines which is the
    index column for the year. Otherwise, the year column is guessed based on
    candidate names
    """
    def __init__(self, pandas_obj):
        """Initialise"""
        super().__init__(pandas_obj)
        self._validate(pandas_obj)

        self.col_year = self._init_col_year(pandas_obj)
        self._validate_years(pandas_obj, self.col_year)

    @staticmethod
    def _validate(obj):
        """Check key requirements for this to work"""

        if 'col_event' not in obj.attrs.keys():
            raise AttributeError("Must have 'col_event' in the series attrs " +
                                 "to specify which index columns define a " +
                                 "unique event.")

        if (isinstance(obj.attrs['col_event'], str)
                or not isinstance(obj.attrs['col_event'], Iterable)):
            raise TypeError("attrs 'col_event' should be an iterable")

        # All event columns are in the multi-index
        if not all(c in obj.index.names for c in obj.attrs['col_event']):
            raise AttributeError("Not all specified event columns are in the " +
                                 "multi-index")

    @staticmethod
    def _init_col_year(obj):
        """Return the index column name for the year"""
        if 'col_year' in obj.attrs.keys():
            col_year = obj.attrs['col_year']
        else:
            col_year = next((c for c in obj.index.names
                             if c.lower() in VALID_YEAR_COLNAMES_LC), None)
            if col_year is None:
                raise AttributeError("No valid year column in " +
                                     f"{obj.index.names}")

        return col_year

    @staticmethod
    def _validate_years(obj, col_year):
        """Check the years make sense"""

        # Check the years are within range 1, n_yrs
        if obj.index.get_level_values(col_year).min() < 1 or \
                obj.index.get_level_values(col_year).max() > obj.attrs['n_yrs']:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def col_event(self):
        """Return the index column names for defining a unique event"""
        return self._obj.attrs['col_event']

    def to_subset(self, **kwargs):
        """Get a version of the YEALT, filtered to certain index levels"""
        this_yealt = self._obj
        for k, val in kwargs.items():
            if (not isinstance(val, Iterable)
                    or isinstance(val, str)):
                # Filtering on a single value
                this_yealt = this_yealt.xs(val, level=k, drop_level=False)
            else:
                # Filtering on a list of values
                this_yealt = this_yealt.loc[
                    this_yealt.index.get_level_values(k).isin(val), :]

        return this_yealt

    def to_yelt(self, **kwargs):
        """Output as a year event loss table

        kwargs can be used to specify the subset of allocation indices to use.
        """

        # Calculate the subset of the yealt for each specified index
        filtered_yealt = self.to_subset(**kwargs)

        # Group and sum
        yelt = filtered_yealt.groupby([self.col_year] + self.col_event).sum()

        return yelt

    def to_ylt(self, is_occurrence=False, **kwargs):
        """Output as a year loss table

        kwargs can be used to specify the subset of allocation indices to use.
        """

        # Group and sum
        if is_occurrence:
            ylt = (self.to_yelt(**kwargs).groupby(self.col_year).max())
        else:
            # Calculate the subset of the yealt for each specified index
            filtered_yealt = self.to_subset(**kwargs)

            ylt = filtered_yealt.groupby(self.col_year).sum()

        return ylt

    def to_yalt(self, is_occurrence=False, **kwargs):
        """Collapse the events to get a year loss table allocated among the
        other indices"""

        # Calculate the subset of the yealt for each specified index
        filtered_yealt = self.to_subset(**kwargs)

        # Group by everything excep the events
        groupcols = [c for c in self._obj.index.names if
                     c not in self.col_event]

        if not is_occurrence:
            # Group and sum
            yalt = filtered_yealt.groupby(groupcols, observed=True).sum()

        else:
            # Identify the max event loss per year, then keep only those events
            filtered_yelt = filtered_yealt.yeal.to_yelt()
            filtered_yelt = (filtered_yelt
                             .sort_values(ascending=False)
                             .reset_index()
                             .groupby('Year').first()
                             .set_index(self.col_event, append=True))

            # Return only those events
            yalt = (filtered_yealt.to_frame()
                    .join(filtered_yelt[self._obj.name].rename('loss2'),
                          how='inner')
                    .reset_index()
                    .set_index(groupcols)[self._obj.name])

        return yalt

    # def to_ep_contrib(self, is_occurrence=False, filterby=None, groupby=None):
    #     """Return the contributors to each year of an EP curve"""
    #
    #     # Calculate the subset of the yealt for each specified index
    #     filtered_yealt = self.to_subset(**filterby)
    #
    #     # Group to the allocation columns
    #     this_yealt = filtered_yealt.groupby([self.col_year] + self.col_event +
    #                                         groupby, observed=True).sum()
    #
    #     # Get the year allocation table
    #
    #     # Calculate exceedence probabilities on the full curve
    #     exprobs = this_yealt.yeal.to_ylt(is_occurrence).yl.exprobs()
    #
