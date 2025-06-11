import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset, SimInput
from input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from input_generator.raw_data_loader import DatasetLoader, SimInput_loader, DPPC_loader, POPC_loader
from input_generator.prior_gen import Bonds, PriorBuilder
from input_generator.utils import get_output_tag, LIPID_MASSES
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI
import pickle as pck

import numpy as np

from mlcg.data import AtomicData
import torch
from copy import deepcopy

def get_dimensions(xyz_dims):
    return [[xyz_dims[0],   0.0000,   0.0000], [  0.0000, xyz_dims[1],   0.0000], [  0.0000,   0.0000, xyz_dims[2]]]

def process_sim_input(
    dataset_name: str,
    raw_data_dir: str,
    save_dir: str,
    tag: str,
    pdb_fns: List[str],
    cg_atoms: List[str],
    embedding_map: CGEmbeddingMap,
    embedding_func: Callable,
    martini_map: bool,
    martini_ref: str,
    skip_residues: List[str],
    class_loader: DatasetLoader,
    copies: int,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    mass_scale: Optional[float] = 418.4,
):
    """
    Generates input AtomicData objects for coarse-grained simulations

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    raw_data_dir : str
        Path to location of input structures
    save_dir : str
        Path to directory in which output will be saved
    tag : str
        Label given to all output files produced from dataset
    pdb_fns : str
        List of pdb filenames from which input will be generated
    cg_atoms : List[str]
        List of atom names to preserve in coarse-grained resolution
    embedding_map : CGEmbeddingMap
        Mapping object
    embedding_func : Callable
        Function which will be used to apply CG mapping
    skip_residues : List[str]
        List of residues to skip, can be None
    copies : int
        Copies that will be produced of each structure listing in pdb_fns
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_builders : List[PriorBuilder]
        List of PriorBuilder objects and their corresponding parameters
    mass_scale : str
        Optional scaling factor applied to atomic masses
    """
    cg_coord_list = []
    cg_type_list = []
    cg_mass_list = []
    cg_nls_list = []

    dataset = SimInput(dataset_name, tag, pdb_fns)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        sample_loader = class_loader
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            name=samples.name, raw_data_dir=raw_data_dir
        )
        if martini_map:
            sample_loader_ref = POPC_loader()

            atomistic_ref_traj, atomistic_ref_top = sample_loader_ref.get_traj_top(
                samples.name, martini_ref
            ) 
        else:
            atomistic_ref_traj = None
            atomistic_ref_top = None

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
            atomistic_ref_traj=atomistic_ref_traj,
            atomistic_ref_top=atomistic_ref_top,
        )

        cg_trajs = samples.input_traj.atom_slice(samples.cg_atom_indices)
        cg_masses = (
                    np.array([int(LIPID_MASSES[atom.name]) for atom in cg_trajs.topology.atoms])
                    / mass_scale
                )
        
        cg_dims = cg_trajs.unitcell_lengths * 10

        prior_nls = samples.get_prior_nls(
            prior_builders=prior_builders,
            save_nls=False,
            save_dir=save_dir,
            prior_tag=prior_tag,
        )
        cg_types = samples.cg_dataframe["type"].to_list()
        for i in range(cg_trajs.n_frames):
            cg_traj = cg_trajs[i]
            cg_coords = cg_traj.xyz * 10
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
            pos=torch.tensor(coords[0], dtype=torch.float64),
            atom_types=torch.tensor(types),
            masses=torch.tensor(masses),
            cell=torch.from_numpy(np.array(get_dimensions(cg_dims[0]), dtype=np.float64)),
            pbc=torch.from_numpy(np.ones((1, 3), dtype=bool)),
        )
        data.neighbor_list = deepcopy(nls)
        data_list.append(data)

    torch.save(
        data_list,
        f"{save_dir}{get_output_tag([dataset_name, tag], placement='before')}configurations.pt",
    )


if __name__ == "__main__":
    print("Start gen_sim_input.py: {}".format(ctime()))

    CLI([process_sim_input])

    print("Finish gen_sim_input.py: {}".format(ctime()))
