"""Module for working with a year loss table
"""
import pandas as pd
import numpy as np


@pd.api.extensions.register_series_accessor("ylt")
class YearLossTable:
    """A year loss table as a pandas series accessor

    The series must have an index 'Year', a name 'Loss', and attribute 'n_yrs'
    (stored in attrs)

    Years go from 1 to n_yrs. Missing years are assumed to have zero loss.
    """
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Verify the name is Loss, index is Year, and attribute n_yrs"""

        if obj.index.name != 'Year':
            raise AttributeError("Must have 'Year' as index")

        if not pd.api.types.is_integer_dtype(obj.index):
            raise TypeError(f"Year must be integer. It is {obj.index.dtype}")

        if not pd.api.types.is_numeric_dtype(obj):
            raise TypeError(f"Series should be numeric. It is {obj.dtype}")

        if 'n_yrs' not in obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the series attrs")

        if not obj.index.is_unique:
            raise AttributeError("Index years are not unique")

        # Check the years are within range 1, n_yrs
        if obj.index.min() < 1 or obj.index.max() > obj.attrs['n_yrs']:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def is_valid(self):
        """Dummy function to run the validation check"""
        return True

    @property
    def n_yrs(self):
        """Return the number of years for the ylt"""
        return self._obj.attrs['n_yrs']

    @property
    def aal(self):
        """Return the average annual loss"""
        return self._obj.sum() / self.n_yrs

    @property
    def is_all_positive(self):
        """Returns true if all loss values are positive"""
        return (self._obj > 0.0).all()

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

    def to_ecdf(self, keep_years=False, **kwargs):
        """Return the empirical cumulative loss distribution function

        :returns: [pandas.DataFrame] with columns 'Loss' and 'CProb' ordered by
        Loss, CProb and Year, respectively. The index is a range index named
        'Order'

        If keep_years=True, then the 'Years' of the original YLT are retained.

        kwargs are passed to ylt.cprob
        """

        # Get a YLT filled in with zero losses
        with_zeros = (self._obj.copy()
                      .reindex(range(1, self.n_yrs + 1), fill_value=0.0))

        # Get loss vs cumulative prop
        ecdf = pd.concat([with_zeros, with_zeros.ylt.cprob(**kwargs)], axis=1)

        # # If we know there is zero prob of zero loss, add it
        # if len(self._obj) == self.n_yrs and self._obj.min() > 0.0:
        #     ecdf = ecdf.append(
        #         pd.DataFrame({'Loss': 0.0, 'CProb': 0.0},
        #                      index=pd.Index([-1], name='Year'))
        #     )

        # Sort with loss ascending
        ecdf = ecdf.reset_index().sort_values(['Loss', 'CProb', 'Year'])

        if not keep_years:
            ecdf = ecdf.drop('Year', axis=1).drop_duplicates()

        # Reset index
        ecdf = ecdf.reset_index(drop=True)
        ecdf.index.name = 'Order'

        return ecdf

    def exprob(self, **kwargs):
        """Calculate the empiric annual exceedance probability for each loss

        The exceedance prob is defined here as P(Loss >= x)

        :returns: [pandas.Series] of probabilities with same index
        """
        return (self._obj.rank(ascending=False, method='min', **kwargs)
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
                      .reindex(range(1, self.n_yrs + 1), fill_value=0.0))

        # Create the dataframe by combining loss with exprob
        ep_curve = pd.concat([with_zeros, with_zeros.ylt.exprob(**kwargs)],
                             axis=1)

        # Sort from largest to smallest loss
        ep_curve = ep_curve.reset_index().sort_values(
            by=['Loss', 'ExProb', 'Year'], ascending=(False, True, False))

        if not keep_years:
            ep_curve = ep_curve.drop('Year', axis=1).drop_duplicates()

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
        ep_curve = self.to_ep_curve(**kwargs)

        # Get the max loss for the high return periods
        max_loss = ep_curve['Loss'].iloc[0]

        # Remove invalid return periods
        return_periods = np.array(return_periods).astype(float)
        return_periods[return_periods < 1.0] = np.nan

        losses = np.interp(1 / return_periods,
                           ep_curve['ExProb'],
                           ep_curve['Loss'],
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

    _ = ylt.ylt.is_valid

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
    _ = ylt.ylt.is_valid

    return ylt
