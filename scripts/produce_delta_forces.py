import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

import mdtraj as md
import numpy as np
import torch

from mlcg.data.atomic_data import AtomicData
from mlcg.datasets.utils import remove_baseline_forces

from input_generator.raw_dataset import *
from input_generator.utils import get_output_tag

from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI


def produce_delta_forces(
    dataset_name: str,
    names: List[str],
    tag: str,
    save_dir: str,
    prior_tag: str,
    prior_fn: str,
    device: str,
    batch_size: int,
    force_tag: Optional[str] = None,
    mol_num_batches: Optional[int] = 1,
):
    """
    Removes prior energy terms from input forces to produce delta force input
    for training

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    tag : str
        Label given to all output files produced from dataset
    save_dir : str
        Path to directory from which input will be loaded and to which output will be saved
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_fn : str
        Path to filename in which prior model is saved
    device: str
        Device on which to run delta force calculations
    batch_size : int
        Number of frames to take per batch
    force_tag: str
        Optional tag to identify input for a particular run of delta force calculation
    mol_num_batches : int
        If greater than 1, will load each molecule data from the specified number of batches
        that were be treated as different samples
    """

    prior_model = torch.load(open(prior_fn, "rb")).models.to(device)
    dataset = RawDataset(dataset_name, names, tag, n_batches=mol_num_batches)
    for samples in tqdm(
        dataset, f"Processing delta forces for {dataset_name} dataset..."
    ):
        if not samples.has_saved_cg_output(save_dir, prior_tag):
            continue
        coords, forces, embeds, cell, pdb, prior_nls = samples.load_cg_output(
            save_dir=save_dir, prior_tag=prior_tag
        )

        # prior_nls = { # this is relevant for martini neighbours
        #     'bonds': {
        #         'tag': 'bonds',
        #         'order': 2,
        #         'index_mapping': torch.tensor([[ 0,  1,  1,  2,  2,  3,  4,  5,  6,  8,  9, 10],
        #                                  [ 1,  2,  3,  3,  4,  8,  5,  6,  7,  9, 10, 11]]),
        #         'cell_shifts': None,
        #         'rcut': None,
        #         'self_interaction': None,
        #         'mapping_batch': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
        #     'angles': {
        #         'tag': 'angles',
        #         'order': 3,
        #         'index_mapping': torch.tensor([[ 0,  1,  3,  2,  2,  4,  5,  3,  8,  9],
        #                                  [ 1,  2,  2,  3,  4,  5,  6,  8,  9, 10],
        #                                  [ 2,  4,  4,  8,  5,  6,  7,  9, 10, 11]]),
        #         'cell_shifts': None,
        #         'rcut': None,
        #         'self_interaction': None,
        #         'mapping_batch': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
        #     'non_bonded': {
        #         'tag': 'non_bonded',
        #         'order': 2,
        #         'index_mapping': torch.tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
        #                                     2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,
        #                                     4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  9],
        #                                  [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  4,  5,  6,  7,  8,  9, 10, 11,
        #                                     5,  6,  7,  8,  9, 10, 11,  4,  5,  6,  7,  9, 10, 11,  6,  7,  8,  9,
        #                                     10, 11,  7,  8,  9, 10, 11,  8,  9, 10, 11,  8,  9, 10, 11, 10, 11, 11]]),
        #         'cell_shifts': None,
        #         'rcut': None,
        #         'self_interaction': None,
        #         'mapping_batch': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                                  0, 0, 0, 0, 0, 0])}}

        num_frames = coords.shape[0]
        delta_forces = []
        iterator = range(0, num_frames, batch_size)
        if len(names) == 1:
            iterator = tqdm(iterator)
        for i in iterator:
            sub_data_list = []
            for j in range(batch_size):
                if i + j < len(coords):
                    data = AtomicData.from_points(
                        pos=torch.tensor(coords[i + j]),
                        forces=torch.tensor(forces[i + j]),
                        cell=torch.from_numpy(np.array(get_dimensions(cell[i+j]))),
                        pbc=torch.from_numpy(np.ones((1, 3), dtype=bool)),
                        atom_types=torch.tensor(embeds),
                        masses=None,
                        neighborlist=prior_nls,
                    )
                    sub_data_list.append(data.to(device))
            sub_data_list = tuple(sub_data_list)
            _ = remove_baseline_forces(
                sub_data_list,
                prior_model,
            )
            for j in range(batch_size):
                if j < len(sub_data_list):
                    delta_force = sub_data_list[j].forces.detach().cpu()
                    delta_forces.append(delta_force.numpy())

        fnout = os.path.join(
            save_dir,
            f"{get_output_tag([tag, samples.name, prior_tag, force_tag], placement='before')}delta_forces.npy",
        )
        np.save(
            fnout,
            np.concatenate(delta_forces, axis=0).reshape(*coords.shape),
        )


if __name__ == "__main__":
    print("Start produce_delta_forces.py: {}".format(ctime()))

    CLI([produce_delta_forces])

    print("Finish produce_delta_forces.py: {}".format(ctime()))
