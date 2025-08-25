"""Class to define a generic policy layer"""
import numpy as np


def apply_layer(losses, limit=None, xs=0.0, share=1.0, **kwargs):
    """Calculate the loss to a basic layer for each entry

    No franchise: If loss > xs, then loss is min(limit, loss - xs) * share.
    With franchise: If loss > xs, then loss is min(limit, loss) * share

    :param losses: the losses before the layer terms have been applied

    :param limit: maximum loss to the layer, before share aplied

    :param xs: minimum loss a.k.a excess/deductible for the layer

    :param share: proportion of loss after applying limit and excess

    :param kwargs: additional arguments.
        is_franchise (bool): if True, the xs acts as a loss threshold rather than
        retention.
        is_step: if True, all losses above the excess have a loss=share

    :returns: a loss series for the loss to the layer. Zero losses are included.

    """

    if isinstance(losses, list):
        losses = np.array(losses)

    if 'is_franchise' in kwargs and kwargs['is_franchise']:
        # Apply the franchise xs for non-zero losses
        layer_losses = np.clip(np.where(losses >= xs, losses, 0.0), a_min=0.0, a_max=limit)

    elif 'is_step' in kwargs and kwargs['is_step']:
        # Use fixed loss for all losses above xs if a step layer
        layer_losses = (losses >= xs) * 1.0

    else:
        # Apply layer attachment and limit
        layer_losses = np.clip(losses - xs, a_min=0.0, a_max=limit)

    # Apply the share and exit
    return layer_losses * share


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

    def _apply_occurrence_terms(self, event_losses):
        """Return the loss net of occurrence limit and excess"""
        return apply_layer(event_losses, limit=self._occ_limit, xs=self._xs)

    def ceded_loss_in_year(self, event_losses):
        """Return the total ceded loss for a set of event losses in a single year. """
        return apply_layer(
                np.sum(self._apply_occurrence_terms(event_losses)),
                limit=self._agg_limit, xs=self._agg_xs, share=self._share)

    def reinstated_limit_in_year(self, event_losses):
        """Return the reinstated limit for a set of event losses in a single year. """
        return apply_layer(
                np.sum(self._apply_occurrence_terms(event_losses)),
                limit=self.max_reinstated_limit, xs=self._agg_xs, share=self._share)

    def ceded_event_losses_in_year(self, event_losses):
        """Return the ceded loss per event for a set of event losses. """

        cumulative_losses = apply_layer(
                np.cumsum(self._apply_occurrence_terms(event_losses)),
                limit=self._agg_limit, xs=self._agg_xs, share=self._share)

        return np.diff(cumulative_losses, prepend=0.0)

    def reinstated_event_losses_in_year(self, event_losses):
        """Return the reinstated limit per event"""
        cumulative_losses = apply_layer(
                np.cumsum(self._apply_occurrence_terms(event_losses)),
                limit=self.max_reinstated_limit, xs=self._agg_xs, share=self._share)

        return np.diff(cumulative_losses, prepend=0.0)

    def ceded_ylt(self, yelt_in, only_reinstated=False):
        """Get the YLT for losses to the layer from an input year-event loss table"""

        if only_reinstated:
            agg_limit = self.max_reinstated_limit
        else:
            agg_limit = self._agg_limit

        year_loss = (yelt_in
                     .apply(apply_layer, limit=self._occ_limit, xs=self._xs)
                     .yel.to_ylt()
                     .apply(apply_layer, limit=agg_limit, xs=self._agg_xs)
                     )

        return year_loss * self._share

    def ceded_yelt(self, yelt_in, only_reinstated=False):
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
                      .apply(apply_layer, limit=self._occ_limit, xs=self._xs)
                      # Calculate cumulative loss in year and apply agg conditions
                      .groupby(yelt_in.yel.col_year).cumsum()
                      .apply(apply_layer, limit=agg_limit, xs=self._agg_xs)
                      )

        # Convert back into the occurrence loss
        lyr_loss = cumul_loss.groupby(yelt_in.yel.col_year).diff().fillna(cumul_loss)
        lyr_loss.attrs['n_yrs'] = yelt_in.yel.n_yrs

        return lyr_loss * self._share

    def paid_reinstatements(self, reinstated_limit):
        """Fraction of premium paid for reinstatements calculated from the amount of
        limit reinstated

        :param reinstated_limit: The amount of limit reinstated after applying the share
        """

        return (reinstated_limit / self.notional_limit ) * self._reinst_at


def variable_reinst_layer(limit: float = None,
                          xs: float = 0.0,
                          share: float = 1.0,
                          reinst_at: list = None,
                          ):
    """Single CATXL layer with different cost of each reinstatement represented by
    multiple layers """

    agg_xs = 0.0
    agg_limit = limit * 2
    layers = []
    for this_reinst_at in reinst_at:
        this_layer = Layer(limit=limit, xs=xs, share=share, agg_xs=agg_xs,
                           agg_limit=agg_limit, reinst_at=this_reinst_at)
        agg_xs += limit
        layers.append(this_layer)

    return layers


#
# class MultiLayer:
#     """Class for a series of layers that acts as a single layer"""
#
#     def __init__(self, layers: List[Layer]  = None):
#         self._layers = layers
#
#     @classmethod
#     def from_variable_reinst_lyr_params(
#             cls,
#             limit,
#             reinst_rates: List[float],
#             **kwargs
#     ):
#         """Initialise a multilayer to represent a single layer with variable
#         reinstatement costs"""
#
#         n_reinst = len(reinst_rates)
#
#         if 'agg_xs' not in kwargs:
#             agg_xs = 0
#         else:
#             agg_xs = kwargs['agg_xs']
#
#         other_layer_params = {k: v for k, v in kwargs.items()
#                               if k not in ('limit', 'agg_xs', 'agg_limit', 'reinst_rate')}
#
#         layers = []
#         for i in range(n_reinst):
#             this_agg_xs = agg_xs + i * limit
#             layers.append(
#                 Layer(limit,
#                     agg_limit=limit*2,
#                     agg_xs=this_agg_xs,
#                     reinst_rate=reinst_rates[i],
#                       **other_layer_params
#                 )
#             )
#
#         layers.append(
#             Layer(limit, agg_limit=limit,
#                   agg_xs=agg_xs + n_reinst * limit,
#                   reinst_rate=0.0, **other_layer_params)
#         )
#
#         return cls(layers)
#
#     @property
#     def layers(self):
#         """Return the list of layers"""
#         return self._layers
#
#
#     def reinst_cost(self, agg_loss):
#         """Calculate the reinstatement cost for a given annual loss"""
#
#         return sum((lyr.reinst_cost(agg_loss) for lyr in self.layers))
#
#     def loss(self, event_losses):
#         """Return the event loss after applying layer terms """
#
#         return sum((lyr.ceded_loss_in_year(event_losses) for lyr in self.layers))
