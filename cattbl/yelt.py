"""Module for working with a period event loss table"""
import warnings
import pandas as pd
import numpy as np

from cattbl import ylt


@pd.api.extensions.register_series_accessor("yelt")
class YearEventLossTable:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(self):
        """Check it is a valid YELT series"""

        # TODO Check the index names

        # TODO Check data types

        # TODO: Check unique index

        # TODO: Check number of years is there

        # TODO: Check index years are within range
        pass

    @property
    def is_valid(self):
        """TODO"""
        return False


def from_cols(Year, EventId, DayOfYear, Loss, n_yrs):
    """Create a pandas DataFrame with our format period event loss table from
    input args

    :returns: (pandas.DataFrame) with unique multi-index ('PeriodIndex', 'EventId',
    'RepNum') and columns ('EventDate', 'Loss')
    """

    new_yelt = pd.DataFrame({'Year': Year,
                             'EventId': EventId,
                             'DayOfYear': DayOfYear,
                             'Loss': Loss})
    try:
        new_yelt.set_index(['Year', 'EventId', 'DayOfYear'],
                           verify_integrity=True, inplace=True)
    except ValueError:
        raise ValueError("You cannot have duplicate combinations of Year, " +
                         "EventID, DayOfYear")

    new_yelt.attrs['n_yrs'] = n_yrs

    return new_yelt['Loss']


def from_df(df, n_yrs=None):
    """Standardise the PELT from a dataframe"""
    
    if n_yrs is None:
        n_yrs = df.attrs['n_yrs']

    # Convert existing columns to lower case so we handle different input
    # options for column case
    df.columns = [c.lower() for c in df.columns]

    yelt = from_cols(Year=df['year'],
                     EventId=df['eventid'],
                     DayOfYear=df['dayofyear'],
                     Loss=df['loss'],
                     n_yrs=n_yrs)
    return yelt

def from_csv(ifile, n_yrs):
    """Read from a csv file"""
    # TODO
    pass