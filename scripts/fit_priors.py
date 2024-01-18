
import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset
from input_generator.embedding_maps import embedding_fivebead, CGEmbeddingMapFiveBead, CGEmbeddingMap

from tqdm import tqdm
import torch
from time import ctime
import numpy as np

from typing import Dict,List,Union, Callable
from jsonargparse import CLI
from scipy.integrate import trapezoid

from torch_geometric.data.collate import collate

from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import Harmonic,_Prior
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
    data: AtomicData,
    target: str,
    nbins: int,
    bmin: float,
    bmax: float,
    TargetPrior: _Prior = Harmonic,
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

def get_strides(n_structure:int, batch_size:int):
    n_elem, remain = np.divmod(n_structure, batch_size)
    assert remain > -1, f"remain: {remain}"
    if remain == 0:
        batches = np.zeros(n_elem+1)
        batches[1:] = batch_size
    else:
        batches = np.zeros(n_elem+2)
        batches[1:-1] = batch_size
        batches[-1] = remain
    strides = np.cumsum(batches, dtype=int)
    strides = np.vstack([strides[:-1], strides[1:]]).T
    return strides

class CGDataBatch:
    def __init__(self, cg_coords, cg_forces, cg_embeds, cg_prior_nls,batch_size:int, stride:int, concat_forces:bool=False) -> None:
        self.batch_size = batch_size
        self.stride = stride
        self.concat_forces = concat_forces
        self.cg_coords = torch.from_numpy(cg_coords[::stride])
        self.cg_forces = torch.from_numpy(cg_forces[::stride])
        self.cg_embeds = torch.from_numpy(cg_embeds)
        self.cg_prior_nls = cg_prior_nls

        self.n_structure = self.cg_coords.shape[0]
        if batch_size > self.n_structure:
            self.batch_size = self.n_structure

        self.strides = get_strides(self.n_structure, self.batch_size)
        self.n_elem = self.strides.shape[0]

    def __len__(self):
        return self.n_elem
    def __getitem__(self, idx):
        st,nd = self.strides[idx]
        data_list = []
        for ii in range(st,nd):
            dd = dict(
                pos=self.cg_coords[ii],
                atom_types=self.cg_embeds,
                masses=None,
                neighborlist=self.cg_prior_nls,
            )
            if self.concat_forces:
                dd['forces'] = self.cg_forces[ii]

            data = AtomicData.from_points(**dd)
            data_list.append(data)
        datas, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )
        return datas

def compute_statistics(
    dataset_name:str,
    names: List[str],
    tag:str,
    pdb_template_fn:str,
    save_dir:str,
    prior_tag:str,
    stride:int,
    batch_size: int,

):
    dataset = RawDataset(dataset_name, names, tag, pdb_template_fn)
    for samples in tqdm(dataset, f"Compute histograms of CG data for {dataset_name} dataset..."):
        cg_coords, cg_forces, cg_embeds, cg_pdb, cg_prior_nls = samples.load_cg_output(save_dir, prior_tag)
        batch_list = CGDataBatch(cg_coords, cg_forces, cg_embeds, cg_prior_nls, batch_size, stride)
        for batch in tqdm(batch_list, f"molecule name: {samples.name}", leave=False):
            



def fit_priors(
    dataset_name:str,
    names: List[str],
    sample_loader_func: Callable,
    raw_data_dir:str,
    tag:str,
    pdb_template_fn:str,
    save_dir:str,
    cg_atoms: List[str],
    embedding_map: CGEmbeddingMap,
    embedding_func: Callable,
    skip_residues:List[str],
    use_terminal_embeddings: bool,
    cg_mapping_strategy:str,
    prior_tag:str,
    prior_dict:dict,
):

    dataset = RawDataset(dataset_name, names, tag, pdb_template_fn)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
        )

        if use_terminal_embeddings:
            #TODO: fix usage add_terminal_embeddings wrt inputs
            samples.add_terminal_embeddings(
                N_term=sub_data_dict["N_term"],
                C_term=sub_data_dict["C_term"]
            )

        prior_nls = samples.get_prior_terms(
            prior_dict,
            save_nls=True,
            save_dir=save_dir,
            prior_tag=prior_tag
        )

        aa_coords, aa_forces =  sample_loader_func(raw_data_dir, samples.name)

        cg_coords, cg_forces = samples.process_coords_forces(aa_coords, aa_forces, mapping=cg_mapping_strategy)

        samples.save_cg_output(save_dir, save_coord_force=True)




if __name__ == "__main__":
    print("Start fit_priors.py: {}".format(ctime()))

    CLI([fit_priors])

    print("Finish fit_priors.py: {}".format(ctime()))
