
import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset
from input_generator.raw_data_loader import DatasetLoader
from input_generator.embedding_maps import embedding_fivebead, CGEmbeddingMapFiveBead, CGEmbeddingMap
from input_generator.prior_gen import Bonds,PriorBuilder

from tqdm import tqdm
import torch
from time import ctime
import numpy as np

from typing import Dict,List,Union, Callable, Optional
from jsonargparse import CLI
from scipy.integrate import trapezoid

from torch_geometric.data.collate import collate

from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import Harmonic,_Prior
from mlcg.nn.gradients import GradientsOut, SumOut
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
    bmin: Optional[float] = None,
    bmax: Optional[float] = None,
    TargetPrior: _Prior = Harmonic,
) -> Dict:
    r"""Function for computing atom type-specific statistics for
    every combination of atom types present in a collated AtomicData
    structure.


    """
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
        bin_centers = _get_bin_centers(nbins, b_min=bmin, b_max=bmax)

        kf = tensor2tuple(_flip_map[order](unique_key))
        histograms[kf] = [hist, bin_centers]

    return histograms


def fit_potentials(
        histograms: Dict,
        prior_fit_fn:Callable,
        target_fit_kwargs: Optional[Dict] = None
):
    if target_fit_kwargs == None:
        target_fit_kwargs = {}

    kB = 0.0019872041
    beta = 1 / (300 * kB) # this will be tunable just hard coded for testing

    statistics = {}
    for kf in histograms.keys():
        hist, bin_centers = histograms[kf][0], histograms[kf][1]

        mask = hist > 0
        bin_centers_nz = bin_centers[mask]
        ncounts_nz = hist[mask]
        dG_nz = -torch.log(ncounts_nz) / beta

        # need to make fit_from_values possible, these functions are new
        params = prior_fit_fn(bin_centers_nz, dG_nz, **target_fit_kwargs)

        statistics[kf] = params

        statistics[kf]["p"] = hist / trapezoid(
            hist.cpu().numpy(), x=bin_centers.cpu().numpy()
        )
        statistics[kf]["p_bin"] = bin_centers
        statistics[kf]["V"] = dG_nz
        statistics[kf]["V_bin"] = bin_centers_nz

    return statistics

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
    
class NLhist(dict):
    def __setitem__(self, key, value) -> None:
        if key not in self:
            return super().__setitem__(key, value)
    def add_hist(self, target: str, histograms: Dict):
        for key, value in histograms.items():
            if key not in self[target]:
                self[target][key] = value
            else:
                hist, bin_centers = value[0], value[1]
                assert np.array_equal(self[target][key][1], bin_centers)
                self[target][key][0] += hist


def fit_priors(
    dataset_name:str,
    names: List[str],
    sample_loader: DatasetLoader,
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
    prior_builders: List[PriorBuilder],
    prior_tag:str,
    fit_parameters: Dict,
):
    dataset = RawDataset(dataset_name, names, tag)
    prior_counts = NLhist()

    print("Accumulating data...")
    for samples in tqdm(dataset, f"Compute histograms of CG data for {dataset_name} dataset...", disable=False):
        cg_coords, cg_forces, cg_embeds, cg_pdb, cg_prior_nls = samples.load_cg_output(
            save_dir=save_dir,
            prior_tag=prior_tag
        )

        batch_list = CGDataBatch(
            cg_coords,
            cg_forces,
            cg_embeds,
            cg_prior_nls,
            fit_parameters["batch_size"],
            fit_parameters["stride"]
        )

        for batch in tqdm(batch_list, f"molecule name: {samples.name}", leave=False, disable=False):
            for builder in prior_builders:
                targets = [name for name in builder.nl_builder.nl_names if name in cg_prior_nls]
                for target in targets:
                    prior_counts[target] = {}
                    histograms = compute_hist(
                        data=batch,   
                        target=target,
                        nbins=builder.n_bins,
                        bmin=builder.bmin,
                        bmax=builder.bmax,
                        TargetPrior=builder.target_prior,
                    )
                    prior_counts.add_hist(target, histograms)
    
    print("Fitting priors...")
    statistics = {}
    prior_models = {}
    for builder in prior_builders:
        nl_names = builder.nl_builder.nl_names
        targets = [name for name in nl_names if name in cg_prior_nls]
        for target in targets:
            statistics[target] = fit_potentials(
                histograms = prior_counts[target],
                prior_fit_fn = builder.prior_fit_fn,
            )
            prior_models[target] = GradientsOut(
                builder.prior_model(statistics[target], name=target),
                targets="forces"
            )
    
    modules = torch.nn.ModuleDict(prior_models)
    full_prior_model = SumOut(modules, targets=["energy", "forces"])
    torch.save(
        full_prior_model,
        "test_prior_model.pt",
    )



if __name__ == "__main__":
    print("Start fit_priors.py: {}".format(ctime()))

    CLI([fit_priors])

    print("Finish fit_priors.py: {}".format(ctime()))
