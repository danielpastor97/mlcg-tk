import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset, SimInput
from input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from input_generator.raw_data_loader import DatasetLoader, SimInput_loader
from input_generator.prior_gen import Bonds, PriorBuilder
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI
import pickle as pck

import numpy as np

from mlcg.data import AtomicData
import torch
from copy import deepcopy

def process_sim_input(
    dataset_name: str,
    raw_data_dir: str,
    save_dir: str,
    tag: str,
    pdb_fns: List[str],
    cg_atoms: List[str],
    embedding_map: CGEmbeddingMap,
    embedding_func: Callable,
    skip_residues: List[str],
    use_terminal_embeddings: bool,
    cg_mapping_strategy: str,
    copies: int,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    mass_scale: Optional[float] = 418.4,
    save_specific_prior: bool = False,
):
    """_summary_

    Parameters
    ----------
    dataset_name : str
        _description_
    raw_data_dir : str
        _description_
    pdb_fns : str
        _description_
    save_dir : str
        _description_
    tag : str
        _description_
    cg_atoms : List[str]
        _description_
    embedding_map : CGEmbeddingMap
        _description_
    embedding_func : Callable
        _description_
    skip_residues : List[str]
        _description_
    use_terminal_embeddings : bool
        _description_
    cg_mapping_strategy : str
        _description_
    """
    cg_coord_list = []
    cg_type_list = []
    cg_mass_list = []
    cg_nls_list = []

    dataset = SimInput(dataset_name, tag, pdb_fns)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        sample_loader = SimInput_loader()
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(name=samples.name, raw_data_dir=raw_data_dir)

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
        )

        if use_terminal_embeddings:
            # TODO: fix usage add_terminal_embeddings wrt inputs
            samples.add_terminal_embeddings(
                N_term=sub_data_dict["N_term"], C_term=sub_data_dict["C_term"]
            )

        cg_traj = samples.input_traj.atom_slice(samples.cg_atom_indices)
        cg_coords = cg_traj.xyz * 10
        cg_types = samples.cg_dataframe["type"].to_list()
        cg_masses = np.array([int(atom.element.mass) for atom in cg_traj.topology.atoms]) / mass_scale

        prior_nls = samples.get_prior_nls(
            prior_builders=prior_builders, save_nls=False, save_dir=save_dir, prior_tag=prior_tag
        )

        for i in range(copies):
            cg_coord_list.append(cg_coords)
            cg_type_list.append(cg_types)
            cg_mass_list.append(cg_masses)
            cg_nls_list.append(prior_nls)

    data_list = []
    for coords, types, masses, nls in zip(
        cg_coord_list, cg_type_list, cg_mass_list, cg_nls_list
    ):
        data = AtomicData.from_points(
            pos=torch.tensor(coords[0]),
            atom_types=torch.tensor(types),
            masses=torch.tensor(masses),
        )
        data.neighbor_list = deepcopy(nls)
        data_list.append(data)

    torch.save(data_list, f"{save_dir}{dataset_name}_configurations.pt")

        

if __name__ == "__main__":
    print("Start gen_sim_input.py: {}".format(ctime()))

    CLI([process_sim_input])

    print("Finish gen_sim_input.py: {}".format(ctime()))
