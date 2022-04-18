"""Module for working with year event loss table, where event loss is allocated.
"""
import pandas as pd
from .yearloss import VALID_YEAR_COLNAMES_LC


@pd.api.extensions.register_series_accessor("yeal")
class YearEventAllocLossTable:
    """A more granular version of a YELT where the event loss is allocated.

    The series should have an attribute called 'colEvent' that contains a list
    of which index columns define a single event.

    The series can have an attribute called 'colYear' that defines which is the
    index column for the year
    """
    def __init__(self, pandas_obj):
        """Initialise"""
        self._validate(pandas_obj)
        self._obj = pandas_obj

        self.col_year = self._init_col_year(pandas_obj)
        self._validate_years(pandas_obj, self.col_year)

    @staticmethod
    def _validate(obj):
        """Check key requirments for this to work"""

        # Check it is a numeric series
        if not pd.api.types.is_numeric_dtype(obj):
            raise TypeError(f"Series should be numeric. It is {obj.dtype}")

        if 'n_yrs' not in obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the series attrs")

        if 'col_event' not in obj.attrs.keys():
            raise AttributeError("Must have 'col_event' in the series attrs " +
                                 "to specify which index columns define a " +
                                 "unique event.")

        # TODO: Check col_event is a list

        # All event columns are in the multi-index
        if not all(c in obj.index.names for c in obj.attrs['col_event']):
            raise AttributeError("Not all specified event columns are in the " +
                                 "multi-index")


        if not obj.index.is_unique:
            raise AttributeError("Index is not unique")

        # TODO: check indices

        pass

    @staticmethod
    def _init_col_year(obj):
        """Return the index column name for the year"""
        if 'col_year' in obj.attrs.keys():
            col_year = obj.attrs['col_year']
        else:
            col_year = next((c for c in obj.index.names
                             if c.lower() in VALID_YEAR_COLNAMES_LC), None)
            if col_year is None:
                raise AttributeError("No valid year column in {}".format(
                        obj.index.names))

        return col_year

    @staticmethod
    def _validate_years(obj, col_year):
        """Check the years make sense"""

        # Check the years are within range 1, n_yrs
        if obj.index.get_level_values(col_year).min() < 1 or \
                obj.index.get_level_values(col_year).max() > obj.attrs['n_yrs']:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def is_valid(self):
        """Dummy function to run the validation check"""
        return True

    @property
    def col_event(self):
        """Return the index column names for defining a unique event"""
        return self._obj.attrs['col_event']

    def to_subset(self, **kwargs):
        """Get a version of the YEAL, filtered to certain index levels"""
        this_yealt = self.obj
        for k in kwargs:
            this_yealt = this_yealt.loc[
                this_yealt.index.get_level_values(k).isin(kwargs[k]), :
            ]
        return this_yealt

    def to_yelt(self, **kwargs):
        """Output as a year event loss table

        kwargs can be used to specify the subset of allocation indices to use.
        """

        # Calculate the subset of the yealt for each specified index
        filtered_yealt = self.to_subset(**kwargs)

        # Group and sum
        yelt = filtered_yealt.groupby([self.colYear] + self.colEvent).sum()

        return yelt

    def to_ylt(self, is_occurrence=False, **kwargs):
        """Output as a year loss table

        kwargs can be used to specify the subset of allocation indices to use.
        """

        # Group and sum
        if is_occurrence:
            ylt = (self.to_yelt(**kwargs).groupby(self.colYear).max())
        else:
            # Calculate the subset of the yealt for each specified index
            filtered_yealt = self.to_subset(**kwargs)

            ylt = filtered_yealt.groupby(self.colYear).sum()

        return ylt

    def to_yalt(self, is_occurrence=False, **kwargs):
        """Return a year loss table allocated"""

        # Group and sum
        if is_occurrence:
            raise Exception("is_occurrence not yet implemented")

            filtered_yealt = self.to_subset(**kwargs)

            # Isolate the events
            ylt = self.to_ylt(is_occurrence=True, **kwargs)

            # Return only those events

            yalt = ylt.join(filtered_yealt, how='inner')

        else:
            # Calculate the subset of the yealt for each specified index
            filtered_yealt = self.to_subset(**kwargs)

            groupcols = [c for c in self.obj.index.names if
                         c not in self.colEvent]
            yalt = filtered_yealt.groupby(groupcols).sum()

        return yalt


    def to_ep_contrib(self, is_occurrence=False, filterby=None, groupby=None):
        """Return the contributors to each year of an EP curve"""

        # Calculate the subset of the yealt for each specified index
        filtered_yealt = self.to_subset(**filterby)

        # Group to the allocation columns
        this_yealt = filtered_yealt.groupby([self.colYear] + self.colEvent +
                                            groupby).sum()

        # Get the year allocation table

        # Calculate exceedence probabilities on the full curve
        exprobs = this_yealt.yeal.to_ylt(is_occurrence).yl.exprobs()



        pass

        return