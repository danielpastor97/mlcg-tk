import torch

from typing import Dict

from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import _Prior
from mlcg.geometry._symmetrize import _symmetrise_map, _flip_map
from mlcg.utils import tensor2tuple

def _get_all_unique_keys(
    unique_types: torch.Tensor, order: int
) -> torch.Tensor:
    """Helper function for returning all unique, symmetrised atom type keys

    Parameters
    ----------
    unique_types:
        Tensor of unique atom types of shape (order, n_unique_atom_types)
    order:
        The order of the interaction type

    Returns
    -------
    torch.Tensor:
       Tensor of unique atom types, symmetrised
    """
    # get all combinations of size order between the elements of unique_types
    keys = torch.cartesian_prod(*[unique_types for ii in range(order)]).t()
    # symmetrize the keys and keep only unique entries
    sym_keys = _symmetrise_map[order](keys)
    unique_sym_keys = torch.unique(sym_keys, dim=1)
    return unique_sym_keys

def _get_bin_centers(
    nbins: int, b_min: float, b_max: float
) -> torch.Tensor:
    """Returns bin centers for histograms.

    Parameters
    ----------
    feature:
        1-D input values of a feature.
    nbins:
        Number of bins in the histogram
    b_min
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    b_max
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature

    Returns
    -------
    torch.Tensor:
        torch tensor containing the locaations of the bin centers
    """

    if b_min >= b_max:
        raise ValueError("b_min must be less than b_max.")

    bin_centers = torch.zeros((nbins,), dtype=torch.float64)

    delta = (b_max - b_min) / nbins
    bin_centers = (
        b_min
        + 0.5 * delta
        + torch.arange(0, nbins, dtype=torch.float64) * delta
    )
    return bin_centers

def compute_hist(
    values: torch.Tensor,
    atom_types: torch.Tensor,
    mapping:torch.Tensor,
    nbins: int,
    bmin: float,
    bmax: float,
) -> Dict:
    r"""Function for computing atom type-specific statistics for
    every combination of atom types present in a collated AtomicData
    structure.


    """
    if target_fit_kwargs == None:
        target_fit_kwargs = {}
    unique_types = torch.unique(atom_types)
    order = mapping.shape[0]
    unique_keys = _get_all_unique_keys(unique_types, order)

    interaction_types = torch.vstack(
        [atom_types[mapping[ii]] for ii in range(order)]
    )

    interaction_types = _symmetrise_map[order](interaction_types)

    histograms = {}
    for unique_key in unique_keys.t():
        # find which values correspond to unique_key type of interaction
        mask = torch.all(
            torch.vstack(
                [
                    interaction_types[ii, :] == unique_key[ii]
                    for ii in range(order)
                ]
            ),
            dim=0,
        )
        val = values[mask]
        if len(val) == 0:
            continue

        hist = torch.histc(val, bins=nbins, min=bmin, max=bmax)

        kf = tensor2tuple(_flip_map[order](unique_key))
        histograms[kf] = hist

    return histograms

def compute_hist_old(
    data: AtomicData,
    target: str,
    nbins: int,
    bmin: float,
    bmax: float,
    TargetPrior: _Prior,
) -> Dict:
    r"""Function for computing atom type-specific statistics for
    every combination of atom types present in a collated AtomicData
    structure.


    """
    if target_fit_kwargs == None:
        target_fit_kwargs = {}
    unique_types = torch.unique(data.atom_types)
    order = data.neighbor_list[target]["index_mapping"].shape[0]
    unique_keys = _get_all_unique_keys(unique_types, order)

    mapping = data.neighbor_list[target]["index_mapping"]
    values = TargetPrior.compute_features(data.pos, mapping)

    interaction_types = torch.vstack(
        [data.atom_types[mapping[ii]] for ii in range(order)]
    )

    interaction_types = _symmetrise_map[order](interaction_types)

    histograms = {}
    for unique_key in unique_keys.t():
        # find which values correspond to unique_key type of interaction
        mask = torch.all(
            torch.vstack(
                [
                    interaction_types[ii, :] == unique_key[ii]
                    for ii in range(order)
                ]
            ),
            dim=0,
        )
        val = values[mask]
        if len(val) == 0:
            continue

        hist = torch.histc(val, bins=nbins, min=bmin, max=bmax)

        kf = tensor2tuple(_flip_map[order](unique_key))
        histograms[kf] = hist

    return histograms
