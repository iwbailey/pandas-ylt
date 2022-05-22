"""Module for working with a year loss table
"""
import warnings
import pandas as pd
import numpy as np
from cattbl.base_classes import LossSeries


# List of valid index names for the year column in order of preference
VALID_YEAR_COLNAMES_LC = ['year', 'period', 'yearidx',
                     'periodidx', 'year_idx', 'period_idx',
                     'yearnumber', 'periodnumber', 'yearno', 'periodno',
                     'yearnum', 'periodnum', 'index', 'idx']


@pd.api.extensions.register_series_accessor("yl")
class YearLossTable(LossSeries):
    """A year loss table as a pandas series accessor

    The series must have an index 'Year', a name 'Loss', and attribute 'n_yrs'
    (stored in attrs)

    Years go from 1 to n_yrs. Missing years are assumed to have zero loss.
    """
    def __init__(self, pandas_obj):
        super().__init__(pandas_obj)
        self._validate(pandas_obj)
        self._obj = pandas_obj

        # Define the column names
        self.col_year = self._obj.index.name
        if self.col_year.lower() not in VALID_YEAR_COLNAMES_LC:
            warnings.warn(f"Index col {self.col_year} is not from expected list " +
                          f"{VALID_YEAR_COLNAMES_LC}")

        self.col_loss = self._obj.name
        if self.col_loss is None:
            self.col_loss = 'Loss'

    @staticmethod
    def _validate(obj):
        """Verify the name is Loss, index is Year, and attribute n_yrs"""

        # Check the years are within range 1, n_yrs
        if obj.index.min() < 1 or obj.index.max() > obj.attrs['n_yrs']:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def std(self):
        """Return the standard deviation of annual loss"""
        return self.to_ylt_filled().std()

    @property
    def prob_of_a_loss(self):
        """Empirical probability of a positive loss year"""
        return (self._obj > 0).sum() / self.n_yrs

    def cprob(self, **kwargs):
        """Calculate the empiric cumulative probability of each loss per year

        CProb = Prob(X<=x) where X is the annual loss
        """
        return (self._obj.rank(ascending=True, method='max', **kwargs)
                .add(self.n_yrs - len(self._obj))
                .divide(self.n_yrs)
                .rename('CProb')
                )

    def to_ylt_filled(self, fill_value=0.0):
        """Get a YLT with all years in the index, missing years filled value"""

        filled_ylt = self._obj.reindex(range(1, int(self.n_yrs) + 1),
                                       fill_value=fill_value)

        return filled_ylt

    def to_ecdf(self, keep_years=False, **kwargs):
        """Return the empirical cumulative loss distribution function

        :returns: [pandas.DataFrame] with columns 'Loss' and 'CProb' ordered by
        Loss, CProb and Year, respectively. The index is a range index named
        'Order'

        If keep_years=True, then the 'Years' of the original YLT are retained.

        kwargs are passed to ylt.cprob
        """

        # Get a YLT filled in with zero losses
        with_zeros = (self.to_ylt_filled(fill_value=0.0)
                      .rename(self.col_loss))

        # Get loss vs cumulative prop
        ecdf = pd.concat([with_zeros, with_zeros.yl.cprob(**kwargs)], axis=1)

        # Sort with loss ascending
        ecdf = ecdf.reset_index().sort_values([self.col_loss, 'CProb',
                                               self.col_year])

        if not keep_years:
            ecdf = ecdf.drop(self.col_year, axis=1).drop_duplicates()

        # Reset index
        ecdf = ecdf.reset_index(drop=True)
        ecdf.index.name = 'Order'

        return ecdf

    def exprob(self, method='max', **kwargs):
        """Calculate the empiric annual exceedance probability for each loss

        The exceedance prob is defined here as P(Loss >= x)

        :returns: [pandas.Series] of probabilities with same index
        """

        return (self._obj.rank(ascending=False, method=method, **kwargs)
                .divide(self.n_yrs)
                .rename('ExProb')
                )

    def to_ep_curve(self, keep_years=False, **kwargs):
        """Get the full loss-exprob curve

        :returns: [pandas.DataFrame] with columns 'Loss', and 'ExProb', index is
        ordered loss from largest to smallest.
        """

        # Get a YLT filled in with zero losses
        with_zeros = (self._obj.copy()
                      .rename(self.col_loss)
                      .reindex(range(1, int(self.n_yrs) + 1), fill_value=0.0))

        # Create the dataframe by combining loss with exprob
        ep_curve = pd.concat([with_zeros, with_zeros.yl.exprob(**kwargs)],
                             axis=1)

        # Sort from largest to smallest loss
        ep_curve = ep_curve.reset_index().sort_values(
            by=[self.col_loss, 'ExProb', self.col_year],
                ascending=(False, True, False))

        if not keep_years:
            ep_curve = ep_curve.drop(self.col_year, axis=1).drop_duplicates()

        # Reset the index
        ep_curve = ep_curve.reset_index(drop=True)
        ep_curve.index.name = 'Order'

        return ep_curve

    def loss_at_rp(self, return_periods, **kwargs):
        """Interpolate the year loss table for losses at specific return periods

        :param return_periods: [numpy.array] should be ordered from largest to
        smallest. A list will also work.

        :returns: [numpy.array] losses at the corresponding return periods

        The interpolation is done on exceedance probability.
        Values below the smallest exceedance probability get the max loss
        Values above the largest exceedance probability get zero
        Invalid exceedance return periods get NaN
        """

        # Get the full EP curve
        ep_curve = self.to_ep_curve(method='first', **kwargs)

        # Get the max loss for the high return periods
        max_loss = ep_curve[self.col_loss].iloc[0]

        # Replace invalid return periods with NaN
        return_periods = np.array(return_periods).astype(float)
        return_periods[return_periods < 1.0] = np.nan

        # Interpolate between the return periods
        losses = np.interp(1 / return_periods,
                           ep_curve['ExProb'],
                           ep_curve[self.col_loss],
                           left=max_loss, right=0.0)

        return losses

    def to_ep_summary(self, return_periods, **kwargs):
        """Get loss at summary return periods and return a pandas Series

        :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
        of those return periods
        """

        return pd.Series(self.loss_at_rp(return_periods, **kwargs),
                         index=pd.Index(return_periods, name='ReturnPeriod'),
                         name='Loss')


def from_cols(year, loss, n_yrs):
    """Create a panadas Series  with year loss table from input args

    :param year: [numpy.Array] an array of integer years

    :param loss: [numpy.Array]

    :param n_yrs: [int]

    :returns: (pandas.DataFrame) with ...
      index
        'Year' [int]
      columns
        'Loss': [float] total period loss
      optional columns
        'MaxLoss': [float] maximum event loss
    """

    ylt = pd.Series(loss, name='Loss', index=pd.Index(year, name='Year'))

    # Store the number of years as meta-data
    ylt.attrs['n_yrs'] = n_yrs

    _ = ylt.yl.is_valid

    return ylt


def from_yelt(yelt):
    """Convert from a year event loss table to year loss table
    """
    ylt = (yelt
           .groupby('Year')
           .sum()
           .sort_index()
           )

    ylt.attrs['n_yrs'] = yelt.attrs['n_yrs']

    # Validate
    _ = ylt.yl.is_valid

    return ylt
