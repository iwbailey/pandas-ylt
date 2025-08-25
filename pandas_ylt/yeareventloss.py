"""Module for working with a year event loss table"""

import pandas as pd
import numpy as np

from pandas_ylt.base_classes import LossSeries
from pandas_ylt.lossexceedance import LossExceedanceCurve
from pandas_ylt.yearloss import VALID_YEAR_COLNAMES_LC

# Default column names
DEFAULT_COLNAME_YEAR = "Year"
COL_EVENT = "EventID"
COL_DAY = "DayOfYear"
DEFAULT_COLNAME_LOSS = "Loss"
INDEX_NAMES = [DEFAULT_COLNAME_YEAR, COL_DAY, COL_EVENT]


def identify_year_col(index_names, valid_yearcol_names=None):
    """Identify which index column corresponds to the year, return the"""

    # Check the default column name as priority
    if DEFAULT_COLNAME_YEAR in index_names:
        result = DEFAULT_COLNAME_YEAR
    elif DEFAULT_COLNAME_YEAR.lower() in index_names:
        result = DEFAULT_COLNAME_YEAR.lower()
    else:
        if valid_yearcol_names is None:
            valid_yearcol_names = VALID_YEAR_COLNAMES_LC

        # Find the first match in a lowercase comparison
        result = next(col for col in index_names if col.lower() in valid_yearcol_names)

    return result


@pd.api.extensions.register_series_accessor("yel")
class YearEventLossTable(LossSeries):
    """Accessor for a Year Event Loss Table as a series.

    The pandas series should have a MultiIndex with one index defining the year
    and remaining indices defining an event. The value of the series should
    represent the loss. There should be an attribute called 'n_yrs'
    """

    def __init__(self, pandas_obj, n_yrs=None):
        """Validate the series for use with accessor"""
        if n_yrs is not None:
            pandas_obj.attrs["n_yrs"] = n_yrs

        super().__init__(pandas_obj)

        self._validate(pandas_obj)

    @staticmethod
    def _validate(obj):
        """Check it is a valid YELT series"""

        # Check the index
        if len(obj.index.names) < 2:
            raise AttributeError(
                "Need at least 2 index levels to define year" + "/events"
            )

        # Check the years are within range 1, n_yrs
        icol = identify_year_col(obj.index.names)
        years = obj.index.get_level_values(icol)
        if years.min() < 1:
            raise AttributeError("Years in index are out of range 1,n_yrs")

        if "n_yrs" in obj.attrs and years.max() > obj.attrs["n_yrs"]:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def col_year(self):
        """The name of the column which stores the year"""
        return identify_year_col(self._obj.index.names)

    @property
    def event_index_names(self):
        """Return the list of all index names in order without the year"""
        return [n for n in self._obj.index.names if n != self.col_year]

    @property
    def freq0(self):
        """Frequency of a loss greater than zero"""
        return (self._obj > 0).sum() / self.n_yrs

    def to_ylt(self, is_occurrence=False):
        """Convert to a YLT

        If is_occurrence return the max loss in a year. Otherwise, return the
        summed loss in a year.
        """

        yrgroup = self._obj.groupby(self.col_year)

        if is_occurrence:
            return yrgroup.max()

        return yrgroup.sum()

    def to_ylt_partitioned(self, splitby=None, is_occurrence=False):
        """Convert to YLT but split the loss into columns based on one index"""

        if splitby is None:
            return self.to_ylt(is_occurrence)

        yrgroup = self._obj.unstack(level=splitby, fill_value=0.0).groupby(self.col_year)

        if is_occurrence:
            ylts = yrgroup.max()
        else:
            ylts = yrgroup.sum()

        ylts.attrs["n_yrs"] = self.n_yrs

        return ylts

    def exfreq(self, **kwargs):
        """For each loss calculate the frequency >= loss

        :returns: [pandas.Series] named 'ExFreq' with the frequency of >= loss
        in the source series. The index is not changed

        **kwargs are passed to pandas.Series.rank . However, arguments are
        reserved: ascending=False, method='min'.
        """
        return (
            self._obj.rank(ascending=False, method="min", **kwargs)
            .divide(self.n_yrs)
            .rename("ExFreq")
        )

    def loss_exfreqs(self, losses):
        """Get the exceedance frequencies for specified loss levels"""

        return np.array([(self._obj >= x).sum() / self.n_yrs for x in losses])

    def cprob(self, **kwargs):
        """Calculate the empiric conditional cumulative probability of loss size

        CProb = Prob(X<=x|Loss has occurred) where X is the event loss, given a
        loss has occurred.
        """
        return (
            self._obj.rank(ascending=True, method="max", **kwargs)
            .divide(len(self._obj))
            .rename("CProb")
        )

    def to_maxloss_yelt(self):
        """Return a YELT with only the maximum event loss in a year"""

        return self._obj.sort_values(ascending=False).groupby(self.col_year).head(1)

    def to_aggloss_in_year(self):
        """Return a YELT with aggregate loss in the year for each event"""
        agg_loss = self._obj.sort_index(level="DayOfYear").groupby("Year").cumsum()

        return agg_loss

    def to_loss_excurve(self, **kwargs):
        """Get the full loss exceedance frequency curve"""

        ef_curve = (
            self._obj.to_frame()
            .assign(ExFreq=self.exfreq(**kwargs))
            .drop_duplicates()
            .sort_values(self.col_loss, ascending=False)
        )

        return LossExceedanceCurve(
            ef_curve[self.col_loss].values, ef_curve["ExFreq"].values, self.n_yrs
        )

    def to_ef_curve(self, col_exfreq="ExFreq", new_index_name="Order", **kwargs):
        """Return an Exceedance frequency curve

        :returns: [pandas.DataFrame] the frequency (/year) of >= each loss
        in the YELT. Column name for loss is retained.
        """

        ef_curve = self.to_loss_excurve(**kwargs).frame

        ef_curve.index.name = new_index_name
        ef_curve = ef_curve.rename(columns={"loss": self.col_loss, "exfreq": col_exfreq})

        return ef_curve

    def loss_at_exfreqs(self, exfreqs, **kwargs):
        """Return the largest losses exceeded at specified frequency"""
        ef_curve = self.to_ef_curve(**kwargs)

        return ef_curve.loss_at_exceedance(exfreqs)

    def to_rp_summary(self, return_periods, is_ep=True, is_occurrence=False, **kwargs):
        """Get loss at summary return periods and return a pandas Series

        :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
        of those return periods
        """

        if is_ep:
            ex_curve = self.to_ylt(is_occurrence).yl.to_loss_excurve(**kwargs)
        else:
            ex_curve = self.to_loss_excurve(**kwargs)

        return ex_curve.rp_summary(return_periods)

    def to_rp_summaries(
        self, return_periods, is_aep=True, is_oep=True, is_ef=True, **kwargs
    ):
        """Return a series with multiple EP curves concatenated"""

        if not is_aep and not is_oep and not is_ef:
            raise ValueError("Must specify one of is_aep, is_oep, is_eef")

        combined = []
        keys = []

        # Calculate the AEP summary
        if is_aep:
            aep = self.to_rp_summary(
                return_periods, is_ep=True, is_occurrence=False, **kwargs
            )

            if "colname_aep" in kwargs:
                keys.append(kwargs.get("colname_aep"))
            else:
                keys.append("AEP")

            combined.append(aep)

        # Calculate the OEP summary
        if is_oep:
            oep = self.to_rp_summary(
                return_periods, is_ep=True, is_occurrence=True, **kwargs
            )

            if "colname_oep" in kwargs:
                keys.append(kwargs.get("colname_oep"))
            else:
                keys.append("OEP")

            combined.append(oep)

        # Calculate the EEF summary
        if is_ef:
            eef = self.to_rp_summary(return_periods, is_ep=False, **kwargs)

            if "colname_eef" in kwargs:
                keys.append(kwargs.get("colname_eef"))
            else:
                keys.append("EF")

            combined.append(eef)

        # Join them all together
        combined = pd.concat(combined, axis=0, keys=keys, names=["Metric"])

        return combined

    def to_severity_curve(
        self, keep_index=False, col_cprob="CProb", new_index_name="Order", **kwargs
    ):
        """Return a severity curve. Cumulative prob of loss size."""

        # Create the dataframe by combining loss with cumulative probability
        sev_curve = pd.concat(
            [
                self._obj.copy().rename(self.col_loss),
                self._obj.yel.cprob(**kwargs).rename(col_cprob),
            ],
            axis=1,
        )

        # Sort from largest to smallest loss
        sev_curve = sev_curve.reset_index().sort_values(
            by=[self.col_loss, col_cprob, self.col_year] + self.event_index_names,
            ascending=[True, True, True] + [True] * len(self.event_index_names),
        )

        if not keep_index:
            sev_curve = sev_curve[[self.col_loss, col_cprob]].drop_duplicates()

        # Reset the index
        sev_curve = sev_curve.reset_index(drop=True)
        sev_curve.index.name = new_index_name

        return sev_curve

    def to_summary_stats_series(self, std_dev_args=None):
        """Return a AAL and std deviation in a pandas series"""

        return self.to_ylt().yl.to_summary_stats_series(std_dev_args)
