"""Module for an event loss table where each row defines a beta distribution of
loss"""
import numpy as np
from scipy.stats import beta as sp_beta


def append_beta_params(elt, eps=1e-12, sd_lower=0.0):
    """Append the alpha and beta parameters of beta distribution to the ELT

    :param elt: (pandas.DataFrame) an event loss table with following columns...
    'MeanLoss', 'StdDev', 'MaxLoss'

    :param eps: (float) minimum value allowed for normalized mean and std dev.
    If either are below this, the alpha and beta are returned as NaN.

    :param sd_lower: (float) apply a lower clip so normalized standard deviation
    is not allowed to be lower than this.

    :returns: dataframe with additional columns 'alpha' and 'beta'

    If the mean or standard deviation are less than eps, then the returned values are
    NaN
    """

    beta_mu = elt['MeanLoss'] / elt['MaxLoss']
    beta_std = (elt['StdDev']  / elt['MaxLoss']).clip(lower=sd_lower)

    elt['alpha'] = beta_mu * ((beta_mu * (1 - beta_mu) / beta_std ** 2) - 1)
    elt['beta'] = elt['alpha'] * (1 - beta_mu) / beta_mu

    # Beta parameters aren't valid when beta_mu or beta_std are less than threshold
    elt.loc[(beta_std < eps) | (beta_mu < eps), ['alpha', 'beta']] = np.nan

    return elt


def expand_elt_to_years(elt, yeqt):
    """Do an inner join between elt and peqt"""

    if elt.index.name != 'EventID':
        elt.set_index('EventID', inplace=True)

    yelt = (elt[['MeanLoss', 'MaxLoss', 'alpha', 'beta']]
            .merge(yeqt, how='inner', left_index=True, right_on='EventID')
            )
    print(f"\t...{len(yelt):,} rows")
    print(f"\t...{yelt.EventId.nunique():,} events are used")

    return yelt


def calculcate_yelt_loss_samples(yelt, quantile_upper=1.0):
    """Apply inverse beta to each row at the quantile using alpha beta

    :param yelt: (pandas.DataFrame) year event mean-loss table with beta parameters
    Needed columns 'MeanLoss', 'LossQuantile', 'alpha', 'beta', 'MaxLoss'

    :param quantile_upper: (float) upper bound for the loss quantile used.

    :returns: the input yelt with

    """

    # Default for invalid numbers is the mean
    is_valid = (~yelt.alpha.isna()) & (~yelt.beta.isna())
    yelt['Loss'] = yelt['MeanLoss']

    # Calculate the loss from inverse beta
    print("Sampling beta distribution at quantiles...")
    yelt.loc[is_valid, 'Loss'] = sp_beta.ppf(
        yelt.loc[is_valid, 'LossQuantile'].clip(upper=quantile_upper),
        a=yelt.loc[is_valid, 'alpha'],
        b=yelt.loc[is_valid, 'beta'],
        scale=yelt.loc[is_valid, 'MaxLoss'])

    return yelt


def elt_to_yelt(elt, yeqt, quantile_upper=1.0):
    """Create a new year Event Loss Table from event loss table

    :param elt: (pandas.DataFrame) event loss table with unique index named
    'EventID' and columns 'MeanLoss', 'StdDev', 'MaxLoss'

    :param yeqt: (pandas.DataFrame) year event quantile table with columns...
    'Year', 'EventID', 'DayOfYear', 'LossQuantile' and attrs['n_yrs']

    :param quantile_upper: [float] The maximum beta quantile to consider before
    using the maximum loss from the ELT

    :returns: (pandas.DataFrame) year event loss table in our format (see pelt)

    """

    elt = append_beta_params(elt)
    print(f"\t...{elt.alpha.count():,} valid alpha and {elt.beta.count():,} " +
          "valid beta")

    # Expand to the year event table
    yelt = expand_elt_to_years(elt, yeqt)

    # Sample the beta distribution at the quantiles
    yelt = calculcate_yelt_loss_samples(yelt, quantile_upper)

    # Clean up
    yelt = yelt.set_index(['Year', 'DayOfYear', 'EventID'])['Loss']
    yelt.attrs['n_yrs'] = yeqt.attrs['n_yrs']

    return yelt
