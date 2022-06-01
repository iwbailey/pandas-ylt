"""Module for working with a year event loss table"""
import warnings
import pandas as pd
import numpy as np

from cattbl.base_classes import LossSeries
from cattbl.yearloss import VALID_YEAR_COLNAMES_LC

# Default column names
DEFAULT_COLNAME_YEAR = 'Year'
COL_EVENT = 'EventID'
COL_DAY = 'DayOfYear'
DEFAULT_COLNAME_LOSS = 'Loss'
INDEX_NAMES = [DEFAULT_COLNAME_YEAR, COL_DAY, COL_EVENT]


def identify_year_col(index_names, valid_yearcol_names=None):
    """Identify which index column corresponds to the year, return the """

    # Check the default column name as priority
    if DEFAULT_COLNAME_YEAR in index_names:
        result = DEFAULT_COLNAME_YEAR
    elif DEFAULT_COLNAME_YEAR.lower() in index_names:
        result = DEFAULT_COLNAME_YEAR.lower()
    else:
        if valid_yearcol_names is None:
            valid_yearcol_names = VALID_YEAR_COLNAMES_LC

        # Find the first match in a lowercase comparison
        result = next(col for col in index_names
                      if col.lower() in valid_yearcol_names)

    return result


@pd.api.extensions.register_series_accessor("yel")
class YearEventLossTable(LossSeries):
    """Accessor for a Year Event Loss Table as a series.

    The pandas series should have a MultiIndex with one index defining the year
    and remaining indices defining an event. The value of the series should
    represent the loss. There should be an attribute called 'n_yrs'
    """
    def __init__(self, pandas_obj):
        """Validate the series for use with accessor"""
        super().__init__(pandas_obj)
        self._validate(pandas_obj)

    @staticmethod
    def _validate(obj):
        """Check it is a valid YELT series"""

        # Check the index
        if len(obj.index.names) < 2:
            raise AttributeError("Need at least 2 index levels to define year" +
                                 "/events")

        # Check the years are within range 1, n_yrs
        icol = identify_year_col(obj.index.names)
        years = obj.index.get_level_values(icol)
        if years.min() < 1 or years.max() > obj.attrs['n_yrs']:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def col_year(self):
        """The name of the column which stores the year"""
        return identify_year_col(self._obj.index.names)

    @property
    def col_loss(self):
        """Return the name of the loss column based on series name or default"""
        if self._obj.name is None:
            loss_col = DEFAULT_COLNAME_LOSS
        else:
            loss_col = self._obj.name

        return loss_col

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

        yrgroup = (self._obj
                   .unstack(level=splitby, fill_value=0.0)
                   .groupby(self.col_year)
                   )

        if is_occurrence:
            ylts = yrgroup.max()
        else:
            ylts = yrgroup.sum()

        ylts.attrs['n_yrs'] = self.n_yrs

        return ylts

    def exfreq(self, **kwargs):
        """For each loss calculate the frequency >= loss

        :returns: [pandas.Series] named 'ExFreq' with the frequency of >= loss
        in the source series. The index is not changed

        **kwargs are passed to pandas.Series.rank . However, arguments are
        reserved: ascending=False, method='min'.
        """
        return (self._obj.rank(ascending=False, method='min', **kwargs)
                .divide(self.n_yrs)
                .rename('ExFreq')
                )

    def cprob(self, **kwargs):
        """Calculate the empiric conditional cumulative probability of loss size

        CProb = Prob(X<=x|Loss has occurred) where X is the event loss, given a
        loss has occurred.
        """
        return (self._obj.rank(ascending=True, method='max', **kwargs)
                .divide(len(self._obj))
                .rename('CProb')
                )

    def to_maxloss_yelt(self):
        """Return a YELT with only the maximum event loss in a year"""

        return (self._obj
                .sort_values(ascending=False)
                .groupby(self.col_year)
                .head(1))

    def to_ef_curve(self, keep_index=False, col_exfreq='ExFreq',
                    new_index_name='Order', **kwargs):
        """Return an Exceedance frequency curve

        :returns: [pandas.DataFrame] the frequency (/year) of >= each loss
        in the YELT. Column name for loss is retained.

        If keep_index=False, duplicate loss
        """

        # Create the dataframe by combining loss with exceedance frequency
        ef_curve = pd.concat([self._obj.copy()
                             .rename(self.col_loss),
                              self._obj.yel.exfreq(**kwargs)
                             .rename(col_exfreq)],
                             axis=1)

        # Sort from largest to smallest loss
        ef_curve = (ef_curve
                    .reset_index()
                    .sort_values(by=[self.col_loss, col_exfreq, self.col_year] +
                                 self.event_index_names,
                                 ascending=[False, True, False] +
                                           [False] * len(self.event_index_names)
                                 )
                    )

        if not keep_index:
            ef_curve = ef_curve[[self.col_loss, col_exfreq]].drop_duplicates()

        # Reset the index
        ef_curve = ef_curve.reset_index(drop=True)
        ef_curve.index.name = new_index_name

        return ef_curve

    def to_severity_curve(self, keep_index=False, col_cprob='CProb',
                          new_index_name='Order', **kwargs):
        """Return a severity curve. Cumulative prob of loss size."""

        # Create the dataframe by combining loss with cumulative probability
        sev_curve = pd.concat([self._obj.copy()
                              .rename(self.col_loss),
                               self._obj.yel.cprob(**kwargs)
                              .rename(col_cprob)],
                              axis=1)

        # Sort from largest to smallest loss
        sev_curve = (sev_curve
                     .reset_index()
                     .sort_values(by=[self.col_loss, col_cprob, self.col_year] +
                                  self.event_index_names,
                                  ascending=[True, True, True] +
                                            [True] * len(self.event_index_names)
                                  ))

        if not keep_index:
            sev_curve = sev_curve[[self.col_loss, col_cprob]].drop_duplicates()

        # Reset the index
        sev_curve = sev_curve.reset_index(drop=True)
        sev_curve.index.name = new_index_name

        return sev_curve

    def loss_at_rp(self, return_periods, **kwargs):
        """Interpolate the year loss table for losses at specific return periods

        :param return_periods: [numpy.array] An array of return periods, which
        should be ordered from largest to smallest. A list will also work.

        :returns: [numpy.array] losses at the corresponding return periods.

        The interpolation is done on exceedance frequency.
        Values below the smallest exceedance frequency get the max loss
        Values above the largest exceedance frequency get zero
        Invalid exceedance return periods get NaN
        """

        # Get the full EP curve
        ef_curve = self.to_ef_curve(**kwargs)

        # Get the max loss for the high return periods
        max_loss = ef_curve[self.col_loss].iloc[0]

        # Remove invalid return periods
        return_periods = np.array(return_periods).astype(float)
        return_periods[return_periods <= 0.0] = np.nan

        losses = np.interp(1 / return_periods,
                           ef_curve['ExFreq'],
                           ef_curve[self.col_loss],
                           left=max_loss, right=0.0)

        return losses

    def apply_layer(self, limit=None, attach=0.0, n_loss=None,
                    is_franchise=False):
        """Calculate the loss to a layer for each event"""

        assert attach >= 0, "Lower loss must be >= 0"

        # Apply layer attachment and limit
        layer_losses = (self._obj
                        .subtract(attach).clip(lower=0.0)
                        .clip(upper=limit)
                        )

        # Keep only non-zero losses to make the next steps quicker
        layer_losses = layer_losses.loc[layer_losses > 0]

        if is_franchise:
            layer_losses += attach

        # Apply occurrence limit
        if n_loss is not None:
            layer_losses = (layer_losses
                            .sort_index(level=[self.col_year] +
                                        self.event_index_names)
                            .groupby(self.col_year).head(n_loss)
                            )

        return layer_losses

    def layer_aal(self, **kwargs):
        """Calculate the AAL within a layer.

        :param kwargs: passed to apply_layer.
        """

        layer_losses = self.apply_layer(**kwargs)

        return layer_losses.sum() / self.n_yrs

    def to_ep_summary(self, return_periods, is_occurrence=False, **kwargs):
        """Get loss at summary return periods and return a pandas Series

        :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
        of those return periods
        """

        ylt = self.to_ylt(is_occurrence)

        return pd.Series(ylt.yl.loss_at_rp(return_periods, **kwargs),
                         index=pd.Index(return_periods, name='ReturnPeriod'),
                         name='Loss')

    def to_ef_summary(self, return_periods, **kwargs):
        """Get loss at summary return periods and return a pandas Series

        For an EF curve, the return period is 1 / rate of event occurrence

        :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
        of those return periods
        """

        return pd.Series(self.loss_at_rp(return_periods, **kwargs),
                         index=pd.Index(return_periods, name='ReturnPeriod'),
                         name='Loss')

    def to_ep_summaries(self, return_periods, is_aep=True, is_oep=True,
                        is_eef=False, **kwargs):
        """Return a dataframe with multiple EP curves side by side"""

        if not is_aep and not is_oep and not is_eef:
            raise Exception("Must specify one of is_aep, is_oep, is_eef")

        combined = []
        keys = []

        # Calculate the AEP summary
        if is_aep:
            aep = self.to_ep_summary(return_periods, is_occurrence=False,
                                     **kwargs)

            if 'colname_aep' in kwargs:
                keys.append(kwargs.get('colname_aep'))
            else:
                keys.append('YearLoss')

            combined.append(aep)

        # Calculate the OEP summary
        if is_oep:
            oep = self.to_ep_summary(return_periods, is_occurrence=True,
                                     **kwargs)

            if 'colname_oep' in kwargs:
                keys.append(kwargs.get('colname_oep'))
            else:
                keys.append('MaxEventLoss')

            combined.append(oep)

        # Calculate the EEF summary
        if is_eef:
            eef = self.to_ef_summary(return_periods, **kwargs)

            if 'colname_eef' in kwargs:
                keys.append(kwargs.get('colname_eef'))
            else:
                keys.append('EventLoss')

            combined.append(eef)

        # Join them all together
        combined = pd.concat(combined, axis=1, keys=keys, names=['CurveType'])

        return combined

    def to_subset_ep_summaries(self, return_periods, splitby=None, **kwargs):
        """Create side-by-side EP summaries for subsets of the YELT indices."""

        if splitby is None:
            raise Exception("Must specify what to split the summaries by.")

        # Loop through each value of the 'splitby' field
        ep_curves = []
        keys = []
        for split_id in self._obj.index.get_level_values(splitby).unique():
            # Call this function on the subset of the YELT
            yelt_sub = self._obj.xs(split_id, level=splitby)
            ep_curves.append(yelt_sub.yel.to_ep_summaries(return_periods,
                                                          **kwargs))
            keys.append(split_id)

        # Make sure the splitby is a list
        if isinstance(splitby, str):
            splitby = [splitby]

        return (pd.concat(ep_curves, keys=keys, axis=1,
                          names=splitby)
                .swaplevel(0, -1, axis=1)
                )

    def to_aal_series(self, is_std=True, aal_name='AAL', std_name='STD'):
        """Return a pandas series with the AAL """

        result = pd.Series([self.aal], index=[aal_name], name='')

        if is_std:
            stddev = self.to_ylt().yl.std
            result = pd.concat([result,
                                pd.Series([stddev], index=[std_name], name='')])

        return result


def from_df(dataframe, n_yrs=None, colname_loss=None):
    """Create a pandas Series YELT (Year Event Loss Table) from a DataFrame

    :param dataframe: [pandas.DataFrame] see from_cols for details of column
    names

    :param n_yrs: [int] the total number of years in the table

    :param colname_loss: [str] the name of the loss column

    :returns: (pandas.Series) compatible with the accessor yelt
    """
    if colname_loss is None:
        colname_loss = DEFAULT_COLNAME_LOSS

    if colname_loss not in dataframe.columns:
        raise KeyError(f'No column named {colname_loss}')

    if n_yrs is not None:
        dataframe.attrs['n_yrs'] = np.int64(n_yrs)

    # Reset index in case
    if dataframe.index.name is not None:
        dataframe = dataframe.reset_index(drop=False)

    # Identify the year column
    year_col = identify_year_col(dataframe.columns)

    # Identify the event columns
    event_cols = [c for c in dataframe.columns
                  if c not in (year_col, colname_loss, None)]

    # Convert the types if necessary
    index_names = [year_col] + event_cols
    for col in index_names:
        if not pd.api.types.is_integer_dtype(dataframe[col]):
            warnings.warn(f"{col} is {dataframe[col].dtype} " +
                          "and will be forced to int type")
            dataframe[col] = dataframe[col].astype(np.int64)

    yelt = dataframe.set_index(index_names, verify_integrity=True)[colname_loss]

    return yelt


def from_csv(ifile, n_yrs):
    """Create a pandas Series YELT (Year Event Loss Table) from a csv file

    :param ifile: [str] file passed to pandas.read_csv see from_cols for details
     of expected column names

    :param n_yrs: [int] the total number of years in the table

    :returns: (pandas.Series) compatible with the accessor yelt
    """

    dataframe = pd.read_csv(ifile, usecols=INDEX_NAMES + [DEFAULT_COLNAME_LOSS])

    return from_df(dataframe, n_yrs)


@pd.api.extensions.register_dataframe_accessor("yel")
class YearEventLossTables:
    """Year event loss tables sharing the same index"""
    def __init__(self, pandas_obj):
        """Validate the dataframe for use with accessor"""

        self._validate(pandas_obj)
        self._obj = pandas_obj

        # Define the column names
        self.col_year = identify_year_col(self._obj.index.names)

    @staticmethod
    def _validate(obj):
        """Check it is a valid YELT series"""

        # Check the dataframe is numeric
        for col in obj.columns:
            if not pd.api.types.is_numeric_dtype(obj[col]):
                raise TypeError("All series should be numeric. " +
                                f"{col} is {obj[col].dtype}")

        # Validate the first series
        assert obj[obj.columns[0]].yel.n_yrs > 0

    @property
    def n_yrs(self):
        """Return the number of years for the ylt"""
        return self._obj.attrs['n_yrs']

    @property
    def event_index_names(self):
        """Return the list of all index names in order without the year"""
        return [n for n in self._obj.index.names if n != self.col_year]

    @property
    def aal(self):
        """Return the average annual loss"""
        return self._obj.sum() / self.n_yrs

    @property
    def freq0(self):
        """Frequency of a loss greater than zero"""
        return (self._obj > 0).sum() / self.n_yrs

    def to_ylt(self, is_occurrence=False):
        """Convert to a YLT

        If is_occurrence return the max loss in a year. Otherwise return the
        summed loss in a year.
        """

        yrgroup = self._obj.groupby(self.col_year)

        if is_occurrence:
            return yrgroup.max()

        return yrgroup.sum()

    def to_ef_curves(self, col_exfreq='ExFreq', **kwargs):
        """Return exceedance frequency curves for each column

        :returns: [pandas.DataFrame] the frequency (/year) of >= each loss
        in the YELT. Column name for loss is retained.

        """

        ef_curves = [self._obj[c].yel.to_ef_curve(keep_index=False,
                                                  col_exfreq=col_exfreq,
                                                  **kwargs)
                     for c in self._obj.columns]

        ef_curves = pd.concat(ef_curves, axis=1)

        ef_curves = ef_curves.fillna(0.0)

        ef_curves = ef_curves.drop(col_exfreq, axis=1)
        ef_curves[col_exfreq] = 1.0 / ef_curves.index

        return ef_curves

    def to_ep_summary(self, return_periods, **kwargs):
        """Get loss at summary return periods and return a pandas Series

        :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
        of those return periods
        """

        losses = pd.concat([self._obj[c].yel.to_ep_summary(return_periods,
                                                           **kwargs)
                            for c in self._obj.columns],
                           axis=1)

        losses.columns = self._obj.columns

        return losses

    def to_ep_summaries(self, return_periods, **kwargs):
        """Get multiple EP curves on the same return period scale"""
        losses = pd.concat([self._obj[c].yel.to_ep_summaries(return_periods,
                                                             **kwargs)
                            for c in self._obj.columns],
                           axis=1, keys=self._obj.columns,
                           names=['LossPerspective'])

        return losses

    def to_subset_ep_summaries(self, return_periods, **kwargs):
        """Get multiple EP curves on the same return period scale"""
        losses = pd.concat([self._obj[c].yel.to_subset_ep_summaries(
                            return_periods, **kwargs)
                            for c in self._obj.columns],
                           axis=1, keys=self._obj.columns,
                           names=['LossPerspective'])

        return losses


    def to_aal_df(self, is_std=True, hide_columns=False, aal_name='AAL',
                  std_name='STD'):
        """Return a pandas series with the AAL """

        result = pd.Series(self.aal, name=aal_name).to_frame().T

        if is_std:
            stddev = [self._obj[c].yel.to_ylt().yl.std
                      for c in self._obj.columns]

            result = result.append(
                     pd.Series(stddev, name=std_name, index=self._obj.columns))

        if hide_columns:
            result.columns = [''] * self._obj.shape[1]

        return result
