
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
import pickle as pck
from typing import Dict,List,Union, Callable, Optional
from jsonargparse import CLI
from scipy.integrate import trapezoid
from collections import defaultdict

from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import Harmonic,_Prior
from mlcg.nn.gradients import GradientsOut, SumOut
from mlcg.geometry._symmetrize import _symmetrise_map, _flip_map
from mlcg.utils import tensor2tuple




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


def compute_statistics(
    dataset_name:str,
    names: List[str],
    tag:str,
    save_dir:str,
    stride:int,
    batch_size: int,
    prior_tag:str,
    # prior_builders: List[PriorBuilder],
    device:str="cpu"
):
    fnout = osp.join(save_dir,f'prior_builders_{prior_tag}.pck')

    dataset = RawDataset(dataset_name, names, tag)
    # histograms = defaultdict(get_zeros)
    for samples in tqdm(dataset, f"Compute histograms of CG data for {dataset_name} dataset..."):
        batch_list = samples.load_cg_output_into_batches(save_dir, prior_tag, batch_size, stride)

        fn = osp.join(save_dir, f"{samples.name}_prior_builders_nl_{prior_tag}.pck")
        with open(fn, 'rb') as f:
            prior_builders = pck.load(f)
        for batch in tqdm(batch_list, f"molecule name: {samples.name}", leave=False):
            # TODO: check NL names and how to deal with it in PriorFit
            for prior_builder in prior_builders:
                prior_builder.accumulate_histogram(batch.to(device))

    with open(fnout, 'wb') as f:
        pck.dump(prior_builders, f)


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

        prior_nls = samples.get_prior_nls(
            prior_dict,
            save_nls=True,
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

    CLI([compute_statistics])

    print("Finish fit_priors.py: {}".format(ctime()))
