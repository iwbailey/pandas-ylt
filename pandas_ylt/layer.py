"""Class to define a generic policy layer"""
#pylint: disable=too-many-arguments
from typing import List
import numpy as np


class Layer:
    """A policy layer"""

    def __init__(
        self,
        limit: float = None,
        xs: float = 0.0,
        share: float = 1.0,
        agg_limit: float = None,
        agg_xs: float = 0.0,
        reinst_rate: float = 0.0,
    ):
        """Define the layer properties"""

        if limit is None:
            limit = np.inf

        if agg_limit is None:
            agg_limit = np.inf

        limit = min(limit, agg_limit)

        self._limit = limit
        self._xs = xs
        self._share = share
        self._agg_limit = agg_limit
        self._agg_xs = agg_xs
        self._reinst_rate = reinst_rate

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
    def reinst_rate(self):
        """Get the reinstatement rate on line"""
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
    
    def yelt_loss(self, yelt_in):
        """Get the YELT for losses to the layer"""

        # Apply occurrence conditions
        occ_loss = yelt_in.apply_layer(limit=self.limit, xs=self._xs)

        # Calclate cumulative loss in year and apply agg conditions
        agg_loss = occ_loss.to_aggloss_in_year()
        agg_loss = np.clip(agg_loss - self._agg_xs, a_min=0.0, a_max=self._agg_limit)
        agg_loss = agg_loss.loc[agg_loss != 0.0]

        # Convert back into the occurrence loss
        lyr_loss = agg_loss.groupby('Year').diff()
        lyr_loss = lyr_loss.fillna(agg_loss)

        return lyr_loss


class MultiLayer:
    """Class for a series of layers that acts as a single layer"""

    def __init__(self, layers: List[float] | None = None):
        self._layers = layers

    @classmethod
    def from_variable_reinst_lyr_params(
        cls,
        limit: float,
        reinst_rates: List[float],
        xs: float = 0.0,
        share: float = 1.0,
        agg_xs: float = 0.0,
    ):
        """Initialise a multilayer to represent a single layer with variable
        reinstatement costs"""

        n_reinst = len(reinst_rates)

        layers = []
        for i in range(n_reinst):
            this_agg_xs = agg_xs + i * limit
            layers.append(
                Layer(limit, xs, share,
                    agg_limit=limit*2,
                    agg_xs=this_agg_xs,
                    reinst_rate=reinst_rates[i],
                )
            )

        layers.append(
            Layer(limit, xs, share, agg_limit=limit,
                  agg_xs=agg_xs + n_reinst * limit,
                  reinst_rate=0.0)
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
