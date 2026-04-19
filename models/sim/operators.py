from typing import Tuple

import torch



def sort_ascending_beginning(x, dim=-1, default_element=0):
    y, y_index = x.sort(dim=dim, descending=False, stable=True)
    y_res = torch.zeros_like(y) + default_element
    y_res_index = torch.zeros_like(y_index)
    y_res[y.flip(dim) != default_element] = y[y != default_element]
    y_res_index[y.flip(dim) != default_element] = y_index[y != default_element]

    return y_res, y_res_index


# General Matching Operator
def matching(bids: torch.Tensor, asks: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Returns a tensor of matched bids and asks and a the corresponding tensor of original indices.
    Typically, the input tensors are expected to be of shape :math:`G \times R \time T_{max} \time N` for

    A successful match is performed when a buyer bid is greater or equal than a seller ask at a given time.
    If equal bids or asks exist, tie breaks are performed by stable sorting.

    Both input tensors need to have the same shape.
    It is assumed that tensors are padded with appropriate elements for sorting, e.g. bids contain :math:`\infty` or a
    very small number less or equal to zero. Asks contain :math:`\infty` for pad elemetns or a very large number, larger
    than the greatest bid.


    Parameters
    ----------
    bids: torch.Tensor
        A tensor of buyer bids (or valuations). The last dimension is expected to be of size :math:`N` `max_buyers`
        which indicates the maximum number of buyer prices.
    asks: torch.Tensor
        A tensor of buyer bids (or valuations). The last dimension is expected to be of size :math:`N` `max_sellers`
        which indicates the maximum number of seller prices.

    Returns
    -------
    matching_mask: torch.Tensor
        A boolean tensor on sorted bids and asks (same dims), that indicates which buyer-seller pairs traded.
    diffs: torch.Tensor
        The differences of sorted bids and asks, used to generate the mask.
    sorted_bids: torch.Tensor
        The bids sorted in descending order across the buyer (last) dimension.
    sorted_asks: torch.Tensor
        The asks sorted in ascending order across the seller (last) dimension.
    sorted_buyer_idx: torch.Tensor
        A long tensor of same dims as (`sorted_bids`) containing the sorted original index.
        This can be used to retrieve the original ordering (prior to sorting) from any other result.
    sorted_seller_idx: torch.Tensor
        A long tensor of same dims as (`sorted_asks`) containing the sorted original index.
        This can be used to retrieve the original ordering (prior to sorting) from any other result.

    """
    sorted_bids, sorted_buyer_idx = torch.sort(bids, dim=-1, descending=True, stable=True)

    sorted_asks, sorted_seller_idx = sort_ascending_beginning(asks, dim=-1, default_element=0)

    diffs = sorted_bids - sorted_asks
    matching_mask = torch.as_tensor(diffs >= 0) & (sorted_asks>0) & (sorted_bids>0)

    return matching_mask, diffs, sorted_bids, sorted_asks, sorted_buyer_idx, sorted_seller_idx


def reorder_by_index(values:torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Reorders a `values` tensor by an `index` long tensor.

    Parameters
    ----------
    values
    index
    dim

    Returns
    -------
    reordered_tensor: torch.Tensor
        The re-ordered `values` tensor.
    """
    return values.gather(dim, index.argsort(dim))


def calculate_ce_metrics(buyer_prices, seller_prices):
    matching_mask, diffs, sorted_bids, sorted_asks, sorted_buyer_idx, sorted_seller_idx = \
        matching(buyer_prices, seller_prices)
    max_b_quantity = (buyer_prices > 0).sum(-1).float()
    max_b_quantity[max_b_quantity == 0] = torch.inf
    max_s_quantity = (seller_prices > 0).sum(-1).float()
    max_s_quantity[max_s_quantity == 0] = torch.inf
    quantity_tensor = matching_mask.sum(-1) #torch.clip(matching_mask.sum(-1), max=max_quantity).long() # matching_mask.sum(-1)

    got_tensor = (matching_mask * diffs).nansum(-1)

    seller_at_idx = quantity_tensor.unsqueeze(-1)
    seller_at_idx = torch.clamp(seller_at_idx, max=sorted_asks.shape[-1] - 1)
    seller_val_at_eq = sorted_asks.gather(dim=-1, index=seller_at_idx)
    seller_val_before_eq = sorted_asks.gather(dim=-1, index=torch.relu(quantity_tensor - 1).unsqueeze(-1))

    buyer_at_idx = quantity_tensor.unsqueeze(-1)
    buyer_at_idx =  torch.clamp(buyer_at_idx, max=sorted_bids.shape[-1] - 1)
    buyer_val_at_eq = sorted_bids.gather(dim=-1, index=buyer_at_idx)
    buyer_val_before_eq = sorted_bids.gather(dim=-1, index=torch.relu(quantity_tensor - 1).unsqueeze(-1))

    ce_left = buyer_val_at_eq.fmax(seller_val_before_eq)
    ce_right = seller_val_at_eq.fmin(buyer_val_before_eq) + buyer_val_before_eq * (
            seller_val_at_eq == 0) + seller_val_at_eq * (buyer_val_before_eq == 0)
    ce = (ce_left + ce_right) / 2

    res = dict(eq_quantity=quantity_tensor, got=got_tensor, ce_low=ce_left, ce_high=ce_right, ce=ce,
               matching_mask=matching_mask, sorted_buyer_idx=sorted_buyer_idx, sorted_seller_idx=sorted_seller_idx,
               sorted_buyer_vals=sorted_bids, sorted_seller_asks=sorted_asks)
    return res
