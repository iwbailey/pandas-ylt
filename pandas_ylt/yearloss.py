"""Module for working with a year loss table
"""

import pandas as pd
import numpy as np
from pandas_ylt.base_classes import LossSeries
from pandas_ylt.lossexceedance import LossExceedanceCurve


# List of valid index names for the year column in order of preference
VALID_YEAR_COLNAMES_LC = [
    "year",
    "period",
    "yearidx",
    "periodidx",
    "year_idx",
    "period_idx",
    "periodid",
    "yearnumber",
    "periodnumber",
    "yearno",
    "periodno",
    "yearnum",
    "periodnum",
    "index",
    "idx",
    "modelyear",
]


@pd.api.extensions.register_series_accessor("yl")
class YearLossTable(LossSeries):
    """A year loss table as a pandas series accessor

    The series must have an index 'Year', a name 'Loss', and attribute 'n_yrs'
    (stored in attrs)

    Years go from 1 to n_yrs. Missing years are assumed to have zero loss.
    """

    def __init__(self, pandas_obj, n_yrs=None):
        if n_yrs is not None:
            pandas_obj.attrs["n_yrs"] = n_yrs

        super().__init__(pandas_obj)
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Verify the name is Loss, index is Year, and attribute n_yrs"""

        # Check the years are within range 1, n_yrs
        if obj.index.min() < 1:
            raise AttributeError("Years less than 1 are present")

        if "n_yrs" in obj.attrs and obj.index.max() > obj.attrs["n_yrs"]:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    def std(self, *args, **kwargs):
        """Return the standard deviation of annual loss"""
        return self.to_ylt_filled().std(*args, **kwargs)

    def summary_stats(self, std_dev_args=None):
        """Get the AAL and std deviation in a dict"""

        if std_dev_args is None:
            std_dev_args = {}

        return {'AAL': self.aal, 'STD': self.std(**std_dev_args)}

    def to_summary_stats_series(self, std_dev_args=None):
        """Get the AAL and std deviation in a pandas Series"""

        if std_dev_args is None:
            std_dev_args = {}

        result = pd.Series(self.summary_stats(std_dev_args)
                           ).rename(self.col_loss)

        result.index.name = 'Metric'

        return result

    @property
    def prob_of_a_loss(self):
        """Empirical probability of a positive loss year"""
        return (self._obj > 0).sum() / self.n_yrs

    def cprob(self, **kwargs):
        """Calculate the empiric cumulative probability of each loss per year

        CProb = Prob(X<=x) where X is the annual loss
        """
        return (
            self._obj.rank(ascending=True, method="max", **kwargs)
            .add(self.n_yrs - len(self._obj))
            .divide(self.n_yrs)
            .rename("CProb")
        )

    def to_ylt_filled(self, fill_value=0.0):
        """Get a YLT with all years in the index, missing years filled value"""

        filled_ylt = self._obj.reindex(
            range(1, int(self.n_yrs) + 1), fill_value=fill_value
        )

        return filled_ylt

    def to_ecdf(self, **kwargs):
        """Return the empirical cumulative loss distribution function

        :returns: [pandas.DataFrame] with columns 'Loss' and 'CProb' ordered by
        Loss, CProb and Year, respectively. The index is a range index named
        'Order'

        kwargs are passed to ylt.cprob
        """

        # Get a YLT filled in with zero losses
        with_zeros_sorted = self.to_ylt_filled(0.0).sort_values(ascending=True)

        # Create the dataframe by combining loss with exprob
        ecdf = (
            pd.concat([with_zeros_sorted,
                       with_zeros_sorted.yl.cprob(**kwargs)], axis=1)
            .drop_duplicates()
        )

        # Reset index
        ecdf = ecdf.reset_index(drop=True)
        ecdf.index.name = "Order"

        return ecdf

    def exprob(self, method="max", **kwargs):
        """Calculate the empiric annual exceedance probability for each loss

        The exceedance prob is defined here as P(Loss >= x)

        :returns: [pandas.Series] of probabilities with same index
        """

        return (
            self._obj.rank(ascending=False, method=method, **kwargs)
            .divide(self.n_yrs)
            .rename("ExProb")
        )

    def loss_exprob(self, x):
        """Get the exceedance prob for a specific single loss value"""
        return float((self._obj >= x).sum()) / self.n_yrs

    def loss_exprobs(self, losses):
        """Get the exceedance probabilities for speocific loss levels"""

        try:
            return np.array([self.loss_exprob(x) for x in losses])
        except TypeError:
            # Case where input is a single value
            return self.loss_exprob(losses)

    def to_loss_excurve(self, **kwargs):
        """Get the full loss-exprob curve

        :returns: [pandas_ylt.LossExceedanceCurve] 
        """

        # Get a YLT filled in with zero losses
        with_zeros_sorted = self.to_ylt_filled(0.0).sort_values(ascending=False)

        # Create the dataframe by combining loss with exprob
        ep_curve = (
            pd.concat([with_zeros_sorted,
                       with_zeros_sorted.yl.exprob(**kwargs)], axis=1)
            .drop_duplicates()
        )

        return LossExceedanceCurve(ep_curve[self.col_loss].values,
                                   ep_curve['ExProb'].values,
                                   self.n_yrs)

    def to_ep_curve(self, **kwargs):
        """Get the full loss-exprob curve

        :returns: [pandas.DataFrame] with columns 'Loss', and 'ExProb', index is
        ordered loss from largest to smallest.
        """

        ep_curve = self.to_loss_excurve(**kwargs).frame

        # Rename the index
        ep_curve.index.name = "Order"

        # Rename the columns
        ep_curve = ep_curve.rename(columns={'loss': self.col_loss, 'exfreq': 'ExProb'})

        return ep_curve

    def loss_at_exprobs(self, exprobs, **kwargs):
        """Get the largest loss(es) exceeded at specified exceedance prob(s)"""

        ep_curve = self.to_loss_excurve(**kwargs)

        return ep_curve.loss_at_exceedance(exprobs)

    def to_rp_summary(self, return_periods, **kwargs):
        """Get loss at summary return periods and return a pandas Series

        :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
        of those return periods
        """

        ep_curve = self.to_loss_excurve(**kwargs)

        return ep_curve.rp_summary(return_periods)



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

    ylt = pd.Series(loss, name="Loss", index=pd.Index(year, name="Year"))

    # Store the number of years as meta-data
    ylt.attrs["n_yrs"] = n_yrs

    _ = ylt.yl.is_valid

    return ylt
