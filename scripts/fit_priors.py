
import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset
from input_generator.embedding_maps import embedding_fivebead, CGEmbeddingMapFiveBead, CGEmbeddingMap
from input_generator.prior_gen import PriorBuilder

from tqdm import tqdm
import torch
from time import ctime
import numpy as np
import pickle as pck
from typing import Dict,List,Union, Callable
from jsonargparse import CLI
from scipy.integrate import trapezoid
from collections import defaultdict

from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import _Prior
from mlcg.geometry._symmetrize import _symmetrise_map, _flip_map
from mlcg.utils import tensor2tuple


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

        prior_nls = samples.get_prior_nls(
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

    CLI([compute_statistics])

    print("Finish fit_priors.py: {}".format(ctime()))
