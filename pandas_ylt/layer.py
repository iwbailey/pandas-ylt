"""Class to define a generic policy layer"""
from typing import List
import warnings
import pandas as pd
import numpy as np


class Layer:
    """A policy layer"""

    def __init__(
            self,
            limit: float = None,
            xs: float = 0.0,
            share: float = 1.0,
            **kwargs
    ):
        """Define the layer properties"""

        if limit is None:
            limit = np.inf

        # Defaults
        other_layer_params = {
            'agg_limit': np.inf,
            'agg_xs': 0.0,
            'reinst_at': 0.0,
            'premium': 0.0,
        }

        # Override defaults with inputs
        for k in other_layer_params:
            if k in kwargs and kwargs[k] is not None:
                other_layer_params[k] = kwargs[k]

        self._occ_limit = limit
        self._xs = xs
        self._share = share
        self._agg_limit = other_layer_params['agg_limit']
        self._agg_xs = other_layer_params['agg_xs']
        self._reinst_at = other_layer_params['reinst_at']
        self._premium = other_layer_params['premium']

        if 'reinst_rate' in kwargs and self._reinst_at == 0.0:
            warnings.warn(
                    ("Use of reinst_rate is deprecated and will be removed " +
                    "in a future release. Use reinst_at and premium instead."),
                    DeprecationWarning,
                    stacklevel=2)

            if self._premium != 0.0:
                self._reinst_at = kwargs['reinst_rate'] * self._occ_limit / self._premium
            else:
                self._premium = kwargs['reinst_rate'] * self._occ_limit
                self._reinst_at = 1.0

        self._validate(self)

    @staticmethod
    def _validate(obj):
        """Validate parameters"""
        if obj.limit <= 0.0:
            raise ValueError("The limit must be greater than zero")

    @property
    def limit(self):
        """Get the layer occurrence limit"""
        return self._occ_limit

    @property
    def notional_limit(self):
        """The share of the occurrence limit"""
        return self._occ_limit * self._share

    @property
    def agg_limit(self):
        """The aggregate limit for the layer"""
        return self._agg_limit

    @property
    def premium(self):
        """Premium for the layer"""
        return self._premium

    @property
    def rate_on_line(self):
        """The rate-on-line for the layer"""
        return self._premium / self.notional_limit

    @property
    def reinst_at(self):
        """Get the proportion of premium to reinstate the limit"""
        return self._reinst_at

    @property
    def reinst_rate(self):
        """Get the reinstatement rate-on-line"""
        return self._reinst_at * self.rate_on_line

    @property
    def max_reinstated_limit(self) -> float:
        """The maximum amount of full limit that can be reinstated in the term"""

        if self._agg_limit == np.inf:
            return np.inf

        return max(self._agg_limit - self._occ_limit, 0.0)

    def reinst_cost(self, agg_loss):
        """Calculate the reinstatement cost for a given annual loss

        Assumes agg xs and share has not yet been applied
        """

        reinstated_limit = min(max(agg_loss - self._agg_xs, 0.0),
                               self.max_reinstated_limit)

        return reinstated_limit * self.reinst_rate * self._share

    def ceded_ylt(self, yelt_in, only_reinstated=False):
        """Get the YLT for losses to the layer from an input year-event loss table"""

        if only_reinstated:
            agg_limit = self.max_reinstated_limit
        else:
            agg_limit = self._agg_limit

        year_loss = (yelt_in
                     .yel.apply_layer(limit=self._occ_limit, xs=self._xs)
                     .yel.to_ylt()
                     .yl.apply_layer(limit=agg_limit, xs=self._agg_xs)
                     )

        return year_loss * self._share

    def ceded_loss_in_year(self, event_losses):
        """Return the total ceded loss for a set of event losses. """

        event_losses = pd.DataFrame({'Year': 1, 'Loss': event_losses})
        event_losses = event_losses.set_index('Year', append=True)['Loss']
        event_losses.attrs['n_yrs'] = 1

        return self.ceded_ylt(event_losses).iloc[0]

    def ceded_yelt(self, yelt_in, only_reinstated=False, net_reinst=False):
        """Get the YELT for losses to the layer

        Aggregate limit and excess are calculated according to the order of events in
        the input YELT. So if you want to apply consecutively, sort the YELT by day of
        year. If you want to apply in order of event size, sort by loss, descending.
        """

        if only_reinstated:
            agg_limit = self.max_reinstated_limit
        else:
            agg_limit = self._agg_limit


        cumul_loss = (yelt_in
                      # Apply occurrence conditions
                      .yel.apply_layer(limit=self._occ_limit, xs=self._xs)
                      # Calculate cumulative loss in year and apply agg conditions
                      .groupby(yelt_in.yel.col_year).cumsum()
                      .yel.apply_layer(limit=agg_limit, xs=self._agg_xs)
                      )

        # Convert back into the occurrence loss
        lyr_loss = cumul_loss.groupby(yelt_in.yel.col_year).diff().fillna(cumul_loss)
        lyr_loss.attrs['n_yrs'] = yelt_in.yel.n_yrs

        if net_reinst:
            reinst_closs = np.minimum(cumul_loss, self.max_reinstated_limit)
            reinst_lyr_loss = reinst_closs.groupby('Year').diff().fillna(reinst_closs)
            reinst_costs = reinst_lyr_loss * self.reinst_rate

            # Reinstatements offset the loss of the layer, so subtract
            lyr_loss = lyr_loss - reinst_costs

        return lyr_loss * self._share

    def ceded_event_losses_in_year(self, event_losses):
        """Return the ceded loss per event for a set of event losses. """

        event_losses = pd.DataFrame({'Year': 1, 'Loss': event_losses})
        event_losses = event_losses.set_index('Year', append=True)['Loss']
        event_losses.attrs['n_yrs'] = 1

        return self.ceded_yelt(event_losses).values



class MultiLayer:
    """Class for a series of layers that acts as a single layer"""

    def __init__(self, layers: List[Layer]  = None):
        self._layers = layers

    @classmethod
    def from_variable_reinst_lyr_params(
            cls,
            limit,
            reinst_rates: List[float],
            **kwargs
    ):
        """Initialise a multilayer to represent a single layer with variable
        reinstatement costs"""

        n_reinst = len(reinst_rates)

        if 'agg_xs' not in kwargs:
            agg_xs = 0
        else:
            agg_xs = kwargs['agg_xs']

        other_layer_params = {k: v for k, v in kwargs.items()
                              if k not in ('limit', 'agg_xs', 'agg_limit', 'reinst_rate')}

        layers = []
        for i in range(n_reinst):
            this_agg_xs = agg_xs + i * limit
            layers.append(
                Layer(limit,
                    agg_limit=limit*2,
                    agg_xs=this_agg_xs,
                    reinst_rate=reinst_rates[i],
                      **other_layer_params
                )
            )

        layers.append(
            Layer(limit, agg_limit=limit,
                  agg_xs=agg_xs + n_reinst * limit,
                  reinst_rate=0.0, **other_layer_params)
        )

        return cls(layers)

    @property
    def layers(self):
        """Return the list of layers"""
        return self._layers


    def reinst_cost(self, agg_loss):
        """Calculate the reinstatement cost for a given annual loss"""

        return sum((lyr.reinst_cost(agg_loss) for lyr in self.layers))

    def loss(self, event_losses):
        """Return the event loss after applying layer terms """

        return sum((lyr.ceded_loss_in_year(event_losses) for lyr in self.layers))
