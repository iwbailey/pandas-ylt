"""Base class for all loss tables"""
import warnings
import pandas as pd


class LossSeries:
    """A loss series with number of years stored in attributes"""
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Check it is a valid loss table series"""

        # Check the series is numeric
        if not pd.api.types.is_numeric_dtype(obj):
            raise TypeError(f"Series should be numeric. It is {obj.dtype}")

        # Check indices can be unique and sortable
        if any((pd.api.types.is_float_dtype(c) for c in obj.index.levels)):
            warnings.warn("Float indices found which might cause errors: ")
            for this_level in obj.index_levels:
                print(this_level.name, ":", this_level.dtype)

        # Check unique
        if not obj.index.is_unique:
            raise AttributeError("Index not unique")

        # Check n_yrs stored in attributes
        if 'n_yrs' not in obj.attrs.keys():
            raise AttributeError("Must have 'n_yrs' in the series attrs")

    @property
    def is_valid(self):
        """Dummy function to pass validation"""
        return True

    @property
    def n_yrs(self):
        """Return the number of years for the ylt"""
        return self._obj.attrs['n_yrs']

    @property
    def aal(self):
        """Return the average annual loss"""
        return self._obj.sum() / self.n_yrs
