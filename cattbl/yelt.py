"""Module for working with a period event loss table"""
import warnings
import pandas as pd
import numpy as np


COL_YEAR = 'Year'
COL_EVENT = 'EventID'
COL_DAY = 'DayOfYear'
COL_LOSS = 'Loss'
INDEX_NAMES = [COL_YEAR, COL_EVENT, COL_DAY]


@pd.api.extensions.register_series_accessor("yelt")
class YearEventLossTable:
    """Accessor for a Year event loss table as a series.

    The pandas series should have a MultiIndex with unique combinations of
    Year, EventID, DayOfYear (all of int type) and its values contain the
    losses. There should be an attribute called 'n_yrs'
    """
    def __init__(self, pandas_obj):
        """Validate the series for use with accessor"""
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Check it is a valid YELT series"""

        # Check the index names
        if len(obj.index.names) != 3:
            raise AttributeError("Expecting 3 index levels")

        if not all([c in obj.index.names for c in INDEX_NAMES]):
            raise AttributeError(f"Expecting index names {INDEX_NAMES}")

        # Check indices are all integer types
        if not all([pd.api.types.is_integer_dtype(c)
                    for c in obj.index.levels]):
            raise TypeError("Indices must all be integer types. Currently: " +
                            f"{[c.dtype for c in obj.index.levels]}")

        # Check the series is numeric
        if not pd.api.types.is_numeric_dtype(obj):
            raise TypeError(f"Series should be numeric. It is {obj.dtype}")

        # Check n_yrs stored in attributes
        if 'n_yrs' not in obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the series attrs")

        # Check unique
        if not obj.index.is_unique:
            raise AttributeError(f"Combinations of {INDEX_NAMES} not unique")

        # Check the years are within range 1, n_yrs
        years = obj.index.get_level_values(COL_YEAR)
        if years.min() < 1 or years.max() > obj.attrs['n_yrs']:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def is_valid(self):
        """Check we can pass the initialisation checks"""
        return True


def from_cols(year, eventid, dayofyear, loss, n_yrs):
    """Create a pandas Series YELT (Year Event Loss Table) from its columns

    :param year: a list or array af integer year numbers starting at 1

    :param eventid: a list or array of integer event IDs

    :param dayofyear: a list or array of integer days within a year

    :param loss: a list or array of numeric losses

    :param n_yrs: [int] the total number of years in the table

    All column inputs should be the same length. The year, eventid, dayofyear
    combinations should all be unique.

    :returns: (pandas.Series) compatible with the accessor yelt
    """

    new_yelt = pd.DataFrame({COL_YEAR: year,
                             COL_EVENT: eventid,
                             COL_DAY: dayofyear,
                             COL_LOSS: loss})
    try:
        new_yelt.set_index([COL_YEAR, COL_EVENT, COL_DAY],
                           verify_integrity=True, inplace=True)
    except ValueError:
        raise ValueError("You cannot have duplicate combinations of " +
                         f"{INDEX_NAMES}")

    new_yelt.attrs['n_yrs'] = n_yrs

    return new_yelt[COL_LOSS]


def from_df(df, n_yrs=None):
    """Create a pandas Series YELT (Year Event Loss Table) from a DataFrame

    :param df: [pandas.DataFrame] see from_cols for details of column names

    :param n_yrs: [int] the total number of years in the table

    :returns: (pandas.Series) compatible with the accessor yelt
    """

    if n_yrs is None:
        n_yrs = df.attrs['n_yrs']

    # Reset index in case
    df = df.reset_index(drop=False)

    # Convert existing columns to lower case so we handle different input
    # options for column case
    df.columns = [c.lower() for c in df.columns]

    # Convert the types if necessary
    for col in [c for c in INDEX_NAMES]:
        col2 = col.lower()
        if not pd.api.types.is_integer_dtype(df[col2]):
            warnings.warn(f"{col} is {df[col2].dtype} " +
                          "and will be forced to int type")
            df[col2] = df[col2].astype(np.int64)

    # Set up the YELT
    yelt = from_cols(year=df[COL_YEAR.lower()],
                     eventid=df[COL_EVENT.lower()],
                     dayofyear=df[COL_DAY.lower()],
                     loss=df[COL_LOSS.lower()],
                     n_yrs=n_yrs)
    return yelt


def from_csv(ifile, n_yrs):
    """Create a pandas Series YELT (Year Event Loss Table) from a csv file

    :param ifile: [str] file passed to pandas.read_csv see from_cols for details
     of expected column names

    :param n_yrs: [int] the total number of years in the table

    :returns: (pandas.Series) compatible with the accessor yelt
    """

    df = pd.read_csv(ifile, usecols=INDEX_NAMES + [COL_LOSS])

    return from_df(df, n_yrs)
