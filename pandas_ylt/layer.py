"""Class to define a generic policy layer"""

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
        reinst_rate=0.0,
    ):
        """Define the layer properties"""

        if limit is None and agg_limit is not None:
            limit = agg_limit

        if limit is not None and agg_limit is not None:
            limit = min(limit, agg_limit)

        self._limit = limit
        self._xs = xs
        self._share = share
        self._agg_limit = agg_limit
        self._agg_xs = agg_xs
        self._reinst_rates = reinst_rate

        self._validate(self)

    @staticmethod
    def _validate(obj):
        # Validation
        assert (
            obj.limit is None or obj.limit > 0.0
        ), "The limit must be greater than zero"

        if not obj.is_const_reinst_rate and len(obj.reinst_rates) != int(
            np.ceil(obj.n_avail_reinst)
        ):
            msg = (
                f"{obj.reinst_rates} costs specified for {obj.n_avail_reinst}"
                + " reinstatements"
            )
            raise ValueError(msg)

    @property
    def limit(self):
        """Get the layer limit"""
        return self._limit

    @property
    def reinst_rates(self):
        """Get the reinstatement rate on line"""
        return self._reinst_rates

    @property
    def is_const_reinst_rate(self) -> bool:
        """Returns True if all reinstatements cost the same"""
        return isinstance(self._reinst_rates, (int, float, complex))

    @property
    def max_reinstated_limit(self) -> float:
        """The maximum amount of limit that can be reinstated in the term"""

        if self._limit is None:
            raise ValueError("Cannot define reinstatements without a limit")

        if self._agg_limit is None:
            return np.inf

        return max(self._agg_limit - self._limit, 0.0)

    @property
    def n_avail_reinst(self) -> float:
        """The number of reinstatements derived from agg and layer limit"""

        return self.max_reinstated_limit / self._limit

    def reinst_cost(self, agg_loss):
        """Calculate the reinstatement cost for a given annual loss"""

        reinstated_limit = min(max(agg_loss - self._agg_xs, 0.0), 
                               self.max_reinstated_limit)

        if self.is_const_reinst_rate:
            return reinstated_limit * self._reinst_rates

        total_reinst = 0.0
        for i, c in enumerate(self._reinst_rates):
            lower = i * self._limit
            amount_reinstated = min(max(reinstated_limit - lower, 0.0), self._limit)
            total_reinst += amount_reinstated * c

        return total_reinst


class MultiLayer:
    """Class for a series of layers"""

    def __init__(self, layers: list[float] | None = None):
        self._layers = layers

    @classmethod
    def from_variable_reinst_lyr_params(
        cls,
        limit: float = None,
        xs: float = 0.0,
        share: float = 1.0,
        agg_xs: float = 0.0,
        reinst_rates: list[float] | None = None,
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

        total_reinst = 0.0
        for lyr in self.layers:
            total_reinst += lyr.reinst_cost(agg_loss)

        return total_reinst
