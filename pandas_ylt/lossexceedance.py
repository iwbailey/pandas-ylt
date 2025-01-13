"""Loss exceedance curve"""
import pandas as pd
import numpy as np


class LossExceedanceCurve:
    """Generic class for exceedance curves"""
    def __init__(self, losses, exfreqs, n_yrs):
        """Initialise the class"""
        self._obj = pd.DataFrame({'loss': losses, 'exfreq': exfreqs})
        self._obj.attrs['n_yrs'] = n_yrs
        self._obj = self._obj.sort_values('loss', ascending=False)
        self._obj.index.name = 'order'

        self._validate(self._obj)

    @staticmethod
    def _validate(obj):
        """Check this is an exceedance curve"""

        # Losses are all decreasing
        if not obj['loss'].is_monotonic_decreasing:
            raise AttributeError("losses should all be decreasing")

        # Rates are all increasing
        if not obj['exfreq'].is_monotonic_increasing:
            raise AttributeError("exceedance frequencies should all be increasing")

    @property
    def frame(self):
        """Return the underlying dataframe"""
        return self._obj

    @property
    def loss(self):
        """Numpy array of the losses"""
        return self._obj.loss.values

    @property
    def exfreq(self):
        """Numpy array of the exceedance rates"""
        return self._obj.exfreq.values

    @property
    def n_yrs(self):
        """Return the number of years for the ylt"""

        # Check n_yrs stored in attributes
        if 'n_yrs' not in self._obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the frame attrs")

        return self._obj.attrs['n_yrs']

    @property
    def min_exfreq(self):
        """The minimum exceedance probability of a loss"""
        return 1 / self.n_yrs

    def max_loss_exceeded(self, x):
        """Get the loss exceeded at a specific rate or probability"""
        is_exceeded = (x >= self.min_exfreq) & (self.exfreq >= x)
        if is_exceeded.any():
            return self.loss[is_exceeded.argmax()]

        return np.nan

    def loss_at_exceedance(self, exfreqs):
        """Get the largest loss(es) exceeded at specified exceedance prob(s)"""

        try:
            return np.array([self.max_loss_exceeded(x) for x in exfreqs])
        except TypeError:
            return self.max_loss_exceeded(exfreqs)

    def rp_summary(self, return_periods):
        """Get loss at summary return periods and return a pandas Series

        :returns: [pands.Series] with index 'ReturnPeriod' and Losses at each
        of those return periods
        """

        return pd.Series(
            self.loss_at_exceedance([1 / r for r in return_periods]),
            index=pd.Index(return_periods, name="ReturnPeriod"),
            name="Loss",
        )
