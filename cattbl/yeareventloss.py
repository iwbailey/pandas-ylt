"""Module for working with a period event loss table"""
import warnings
import pandas as pd
import numpy as np

from cattbl.yearloss import VALID_YEAR_COLNAMES_LC

COL_YEAR = 'Year'
COL_EVENT = 'EventID'
COL_DAY = 'DayOfYear'
COL_LOSS = 'Loss'
INDEX_NAMES = [COL_YEAR, COL_DAY, COL_EVENT]


def identify_year_col(index_names, valid_yearcol_names=VALID_YEAR_COLNAMES_LC):
    """Identify which index corresponds to the year"""

    # Check we can find the year index
    yrcol_match = [i for i, n in enumerate(index_names)
                   if n is not None and n.lower() in valid_yearcol_names]

    if len(yrcol_match) == 0:
        warnings.warn("No valid year column name amongst index names." +
                      f"\nIndex names: {index_names}" +
                      f"\nValid names (case insensitive): {valid_yearcol_names}")
        icol = 0

    else:
        icol = yrcol_match[0]

    return icol


@pd.api.extensions.register_series_accessor("yel")
class YearEventLossTable:
    """Accessor for a Year Event Loss Table as a series.

    The pandas series should have a MultiIndex with one index defining the year
    and remaining indices defining an event. The value of the series should
    represent the loss. There should be an attribute called 'n_yrs'
    """
    def __init__(self, pandas_obj):
        """Validate the series for use with accessor"""

        self._validate(pandas_obj)
        self._obj = pandas_obj

        # Define the column names
        self.col_year = self._obj.index.names[
            identify_year_col(self._obj.index.names)]

    @staticmethod
    def _validate(obj):
        """Check it is a valid YELT series"""

        # Check the series is numeric
        if not pd.api.types.is_numeric_dtype(obj):
            raise TypeError(f"Series should be numeric. It is {obj.dtype}")

        # Check the index
        if len(obj.index.names) < 2:
            raise AttributeError("Need at least 2 index levels to define year" +
                                 "/events")

        # Check indices can be unique and sortable
        if any([pd.api.types.is_float_dtype(c) for c in obj.index.levels]):
            warnings.warn("Float indices found which might cause errors: " +
                            f"{[c.dtype for c in obj.index.levels]}")

        # Check unique
        if not obj.index.is_unique:
            raise AttributeError(f"Index not unique")

        # Check n_yrs stored in attributes
        if 'n_yrs' not in obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the series attrs")

        # Check the years are within range 1, n_yrs
        icol = identify_year_col(obj.index.names)
        years = obj.index.get_level_values(icol)
        if years.min() < 1 or years.max() > obj.attrs['n_yrs']:
            raise AttributeError("Years in index are out of range 1,n_yrs")

    @property
    def is_valid(self):
        """Check we can pass the initialisation checks"""
        return True

    @property
    def n_yrs(self):
        """Return the number of years for the ylt"""
        return self._obj.attrs['n_yrs']

    @property
    def col_loss(self):
        """Return the name of the loss column based on series name or default"""
        if self._obj.name is None:
            return 'Loss'
        else:
            return self._obj.name

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
        else:
            return yrgroup.sum()

    def to_ylt_subsets(self, splitby=None, is_occurrence=False):
        """Partition into YLTs based on one of the indices"""

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

    def to_ef_curve(self, keep_index=False, col_exfreq='ExFreq',
                    new_index_name='Order', **kwargs):
        """Return an Exceedance frequency curve

        :returns: [pandas.DataFrame] the frequency (/year) of >= each loss
        in the YELT. Column name for loss is retained.

        If keep_index=False, duplicate loss
        """

        # Create the dataframe by combining loss with exfreq
        # TODO: is the copy necessary here?
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
                                 [False] * len(self.event_index_names))
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
        # TODO: is the copy necessary here?
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
                                            [True] * len(self.event_index_names)))

        if not keep_index:
            sev_curve = sev_curve[[self.col_loss, col_cprob]].drop_duplicates()

        # Reset the index
        sev_curve = sev_curve.reset_index(drop=True)
        sev_curve.index.name = new_index_name

        return sev_curve

    def loss_at_rp(self, return_periods, **kwargs):
        """Interpolate the year loss table for losses at specific return periods

        :param return_periods: [numpy.array] should be ordered from largest to
        smallest. A list will also work.

        :returns: [numpy.array] losses at the corresponding return periods

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

    def apply_layer(self, limit=None, xs=0.0, n_loss=None, is_franchise=False):
        """Calculate the loss to a layer for each event"""

        assert xs >= 0, "Lower loss must be >= 0"

        # Apply layer attachment and limit
        layer_losses = (self._obj
                        .subtract(xs).clip(lower=0.0)
                        .clip(upper=limit)
                        )

        # Keep only non-zero losses to make the next steps quicker
        layer_losses = layer_losses.loc[layer_losses > 0]

        if is_franchise:
            layer_losses += xs

        # Apply occurrence limit
        if n_loss is not None:
            layer_losses = (layer_losses
                            .sort_index(level=[self.col_year] +
                                               self.event_index_names)
                            .groupby(self.col_year).head(n_loss)
                            )

        return layer_losses

    def layer_aal(self, **kwargs):
        """Calculate the AAL within a layer

        :param kwargs: passed to .apply_layer
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
                        is_eef=False, splitby=None, colname_aep='YearLoss',
                        colname_oep='MaxEventLoss', colname_eef='EventLoss',
                        **kwargs):
        """Return a dataframe with multiple EP curves side by side"""

        if not is_aep and not is_oep and not is_eef:
            raise Exception("Must specify one of is_aep, is_oep, is_eef")

        # Optionally split based in a specific index value
        if splitby is not None:

            # Loop through each value of the splitby field
            ep_curves = []
            keys = []
            for split_id in self._obj.index.get_level_values(splitby).unique():

                # Call this function on the subset of the YELT
                yelt_sub = self._obj.xs(split_id, level=splitby)
                ep_curves.append(yelt_sub.yel.to_ep_summaries(return_periods,
                                                              is_aep, is_oep,
                                                              is_eef, None,
                                                              colname_aep,
                                                              colname_oep,
                                                              colname_eef,
                                                              **kwargs))
                keys.append(split_id)
            if isinstance(splitby, str):
                splitby = [splitby]
            return (pd.concat(ep_curves, keys=keys, axis=1,
                              names=splitby)
                    .swaplevel(0, -1, axis=1)
                    )

        combined = []
        keys = []
        if is_aep:
            aep = self.to_ep_summary(return_periods, is_occurrence=False,
                                     **kwargs)
            keys.append(colname_aep)
            combined.append(aep)

        if is_oep:
            oep = self.to_ep_summary(return_periods, is_occurrence=True,
                                     **kwargs)
            keys.append(colname_oep)
            combined.append(oep)

        if is_eef:
            eef = self.to_ef_summary(return_periods, **kwargs)
            keys.append(colname_eef)
            combined.append(eef)

        combined = pd.concat(combined, axis=1, keys=keys, names=['CurveType'])

        return combined

    def to_aal_series(self, is_std=True, aal_name='AAL', std_name='STD'):
        """Return a pandas series with the AAL """

        result = pd.Series([self.aal], index=[aal_name], name='')

        if is_std:
            stddev = self.to_ylt().yl.std
            result = pd.concat([result,
                                pd.Series([stddev], index=[std_name], name='')])

        return result


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
        new_yelt.set_index(INDEX_NAMES,
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


@pd.api.extensions.register_dataframe_accessor("yel")
class YearEventLossTables:
    """Year event loss tables sharing the same index"""
    def __init__(self, pandas_obj):
        """Validate the dataframe for use with accessor"""

        self._validate(pandas_obj)
        self._obj = pandas_obj

        # Define the column names
        self.col_year = self._obj.index.names[
            identify_year_col(self._obj.index.names)]

    @staticmethod
    def _validate(obj):
        """Check it is a valid YELT series"""

        # Check the dataframe is numeric
        for c in obj.columns:
            if not pd.api.types.is_numeric_dtype(obj[c]):
                raise TypeError(f"All series should be numeric. {c} is {obj[c].dtype}")

        # Validate the first series
        assert obj[obj.columns[0]].yel.is_valid

    @property
    def is_valid(self):
        """Check we can pass the initialisation checks"""
        return True

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
        else:
            return yrgroup.sum()

    def to_ef_curves(self, col_exfreq='ExFreq', **kwargs):
        """Return exceedance frequency curves for each column

        :returns: [pandas.DataFrame] the frequency (/year) of >= each loss
        in the YELT. Column name for loss is retained.

        """

        ef_curves = pd.concat([self._obj[c].yel.to_ef_curve(keep_index=False,
                                                            col_exfreq=col_exfreq,
                                                            **kwargs)
                               for c in self._obj.columns],
                              axis=1)

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

    # def to_ef_summary(self, return_periods, **kwargs):
    #     """Get loss at summary return periods and return a pandas Series
    #
    #     For an EF curve, the return period is 1 / rate of event occurrence
    #
    #     :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
    #     of those return periods
    #     """
    #
    #     return pd.Series(self.loss_at_rp(return_periods, **kwargs),
    #                      index=pd.Index(return_periods, name='ReturnPeriod'),
    #                      name='Loss')
    #
    # def to_ep_summaries(self, return_periods, is_eef=True,
    #                     colname_aep='YearLoss', colname_oep='MaxEventLoss',
    #                     colname_eef='EventLoss', **kwargs):
    #     """Return a dataframe with multiple EP curves side by side"""
    #
    #     aep = self.to_ep_summary(return_periods, is_occurrence=False, **kwargs)
    #     oep = self.to_ep_summary(return_periods, is_occurrence=True, **kwargs)
    #
    #     aep = aep.rename(colname_aep)
    #     oep = oep.rename(colname_oep)
    #
    #     if is_eef:
    #         eef = self.to_ef_summary(return_periods, **kwargs)
    #         eef = eef.rename(colname_eef)
    #         combined = pd.concat([aep, oep, eef], axis=1)
    #     else:
    #         combined = pd.concat([aep, oep], axis=1)
    #
    #     return combined

    def to_aal_df(self, is_std=True, hide_columns=False, aal_name='AAL', std_name='STD'):
        """Return a pandas series with the AAL """

        result = pd.Series(self.aal, name=aal_name).to_frame().T

        if is_std:
            stddev = [self._obj[c].yel.to_ylt().yl.std for c in self._obj.columns]

            result = result.append(
                     pd.Series(stddev, name=std_name, index=self._obj.columns))

        if hide_columns:
            result.columns = [''] * self._obj.shape[1]

        return result
