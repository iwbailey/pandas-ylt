"""Class to define a generic policy layer"""
from typing import List
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
            'reinst_rate': 0.0,
            'premium': 0.0,
        }

        # Override defaults with inputs
        for k in other_layer_params:
            if k in kwargs and kwargs[k] is not None:
                other_layer_params[k] = kwargs[k]

        limit = min(limit, other_layer_params['agg_limit'])

        self._limit = limit
        self._xs = xs
        self._share = share
        self._agg_limit = other_layer_params['agg_limit']
        self._agg_xs = other_layer_params['agg_xs']
        self._reinst_rate = other_layer_params['reinst_rate']
        self._premium = other_layer_params['premium']

        self._validate(self)

    @staticmethod
    def _validate(obj):
        """Validate parameters"""
        if obj.limit <= 0.0:
            raise ValueError("The limit must be greater than zero")

    @property
    def limit(self):
        """Get the layer limit"""
        return self._limit

    @property
    def notional_limit(self):
        """The share of the limit"""
        return self._limit * self._share

    @property
    def agg_limit(self):
        """The aggregate limit for the layer"""
        return self._agg_limit

    @property
    def premium(self):
        """Premium for the layer"""
        return self._premium

    @property
    def reinst_rate(self):
        """Get the reinstatement rate-on-line"""
        return self._reinst_rate

    @property
    def max_reinstated_limit(self) -> float:
        """The maximum amount of full limit that can be reinstated in the term"""

        if self._agg_limit == np.inf:
            return np.inf

        return max(self._agg_limit - self._limit, 0.0)

    def reinst_cost(self, agg_loss):
        """Calculate the reinstatement cost for a given annual loss"""

        reinstated_limit = min(max(agg_loss - self._agg_xs, 0.0),
                               self.max_reinstated_limit)

        return reinstated_limit * self._reinst_rate * self._share

    def loss(self, event_loss, prior_agg_loss=0.0):
        """Return the event loss after applying layer terms """

        loss = np.clip(event_loss - self._xs, a_min=0.0, a_max=self._limit)

        updated_agg_xs = max(self._agg_xs - prior_agg_loss, 0.0)
        remaining_limit = max(self._agg_limit -
                              max(prior_agg_loss - self._agg_xs, 0.0), 0.0)

        loss = np.clip(loss - updated_agg_xs, a_min=0.0, a_max=remaining_limit)

        return loss * self._share

    def yelt_loss(self, yelt_in, net_reinst=False):
        """Get the YELT for losses to the layer"""

        # Apply occurrence conditions
        occ_loss = yelt_in.yel.apply_layer(limit=self.limit, xs=self._xs)

        # Calculate cumulative loss in year and apply agg conditions
        cumul_loss = occ_loss.yel.to_aggloss_in_year()
        cumul_loss = np.clip(cumul_loss - self._agg_xs, a_min=0.0, a_max=self.agg_limit)
        cumul_loss = cumul_loss.loc[cumul_loss != 0.0]

        # Convert back into the occurrence loss
        lyr_loss = cumul_loss.groupby('Year').diff()
        lyr_loss = lyr_loss.fillna(cumul_loss)
        lyr_loss.attrs['n_yrs'] = yelt_in.yel.n_yrs

        if net_reinst:
            reinst_closs = np.minimum(cumul_loss, self.max_reinstated_limit)
            reinst_lyr_loss = reinst_closs.groupby('Year').diff().fillna(reinst_closs)
            reinst_costs = reinst_lyr_loss * self.reinst_rate

            # Reinstatements offset the loss of the layer, so subtract
            lyr_loss = lyr_loss - reinst_costs

        return lyr_loss


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

    def loss(self, event_loss, prior_agg_loss=0.0):
        """Return the event loss after applying layer terms """

        return sum((lyr.loss(event_loss, prior_agg_loss) for lyr in self.layers))
