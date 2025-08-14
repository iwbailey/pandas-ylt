"""Base class for all loss tables"""
import pandas as pd
import numpy as np


DEFAULT_COLNAME_LOSS = "Loss"


@pd.api.extensions.register_series_accessor("ls")
class LossSeries:
    """A loss series with number of years stored in attributes"""
    def __init__(self, pandas_obj, n_yrs=None):
        """Initialise the class"""
        if n_yrs is not None:
            pandas_obj.attrs['n_yrs'] = n_yrs
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Check it is a valid loss table series"""

        # Check the series is numeric
        if not pd.api.types.is_numeric_dtype(obj):
            raise TypeError(f"Series should be numeric. It is {obj.dtype}")

        # Check unique
        if not obj.index.is_unique:
            raise AttributeError("Index not unique")

    @property
    def col_loss(self):
        """Return the name of the loss column based on series name or default"""

        return self._obj.name

    @property
    def is_valid(self):
        """Dummy function to pass validation"""

        # Check n_yrs stored in attributes
        if 'n_yrs' not in self._obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the series attrs")

        return True

    @property
    def n_yrs(self):
        """Return the number of years for the ylt"""

        # Check n_yrs stored in attributes
        if 'n_yrs' not in self._obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the series attrs")

        return self._obj.attrs['n_yrs']

    @property
    def aal(self):
        """Return the average annual loss"""
        return self._obj.sum() / self.n_yrs

    def set_n_yrs(self, n_yrs):
        """Add n_yrs as an attr of the series"""
        loss_series = self._obj.copy()
        loss_series.attrs['n_yrs'] = n_yrs

        return loss_series

    def apply_layer(self, limit=None, xs=0.0, share=1.0, is_franchise=False,
                    is_step=False):
        """Calculate the loss to a layer for each entry

        No franchise: If loss > xs, then loss is min(limit, loss - xs) * share.
        With franchise: If loss > xs, then loss is min(limit + xs, loss) * share

        :param limit: maximum loss to the layer, before share aplied

        :param xs: minimum loss a.k.a excess/deductible for the layer

        :param share: proportion of loss after applying limit and excess

        :param is_franchise: if True, the xs acts as a loss threshold rather than
        retention.

        :param is_step: if True, all losses above the excess have a loss=share

        :returns: a loss series for the loss to the layer. Zero losses are included.

        """

        # Apply layer attachment and limit
        layer_losses = np.clip(self._obj - xs, a_min=0.0, a_max=limit)

        # Apply the franchise xs for non-zero losses
        if is_franchise:
            layer_losses.loc[layer_losses > 0] += xs

        if is_step:
            layer_losses.loc[layer_losses > 0] = 1.0

        # Apply the share and exit
        return layer_losses * share