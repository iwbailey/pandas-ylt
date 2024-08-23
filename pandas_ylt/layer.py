"""Class to define a generic policy layer"""
import numpy as np


class Layer:
    """A policy layer"""
    def __init__(self, limit=None, xs=0.0, share=1.0, agg_limit=None, agg_xs=0.0,
                 reinst_rates=0.0):
        """Define the layer properties"""

        self._limit = limit
        self._xs = xs
        self._share = share
        self._agg_limit = agg_limit
        self._agg_xs = agg_xs

        # Validation
        assert self._limit is None or self._limit > 0.0, \
            "The limit must be greater than zero"

        assert self._agg_limit is None or self._limit is None or self._agg_limit >= self._limit, \
            "Agg limit cannot be smaller than layer limit"

        # Set up the reinstatements as a list
        self._reinst_rates = reinst_rates

        if self._agg_limit is not None and self._limit is not None:
            if isinstance(reinst_rates, (float, int)):
                self._reinst_rates = [reinst_rates] * int(np.ceil(self.n_avail_reinst))
            else:
                if len(reinst_rates) != int(np.ceil(self.n_avail_reinst)):
                    raise(ValueError("{} costs specified for {} reinstatements".format(
                        len(reinst_rates), self.n_avail_reinst)))




    @property
    def n_avail_reinst(self) -> float:
        """The number of reinstatements derived from agg and layer limit"""            
        
        if self._limit is None:
            raise ValueError("Cannot define reinstatements without a limit")

        if self._agg_limit is None:
            return None
        
        n_reinst = max(self._agg_limit - self._limit, 0.0) / self._limit

        return n_reinst