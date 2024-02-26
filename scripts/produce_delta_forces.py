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

from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable
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
):
    """_summary_

    Parameters
    ----------
    dataset_name : str
        _description_
    names : List[str]
        _description_
    tag : str
        _description_
    save_dir : str
        _description_
    prior_tag : str
        _description_
    prior_fn : str
        _description_
    batch_size : int
        _description_
    """

    prior_model = torch.load(open(prior_fn, "rb")).models.to(device)
    dataset = RawDataset(dataset_name, names, tag)
    for samples in tqdm(
        dataset, f"Processing delta forces for {dataset_name} dataset..."
    ):
        coords, forces, embeds, pdb, prior_nls = samples.load_cg_output(
            save_dir=save_dir, prior_tag=prior_tag
        )

        num_frames = coords.shape[0]
        delta_forces = []
        for i in range(0, num_frames, batch_size):
            sub_data_list = []
            for j in range(batch_size):
                if i + j < len(coords):
                    data = AtomicData.from_points(
                        pos=torch.tensor(coords[i + j]),
                        forces=torch.tensor(forces[i + j]),
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

        np.save(
            os.path.join(save_dir, f"{tag}{samples.name}_{prior_tag}_delta_forces.npy"),
            np.concatenate(delta_forces, axis=0).reshape(*coords.shape),
        )


if __name__ == "__main__":
    print("Start produce_delta_forces.py: {}".format(ctime()))

    CLI([produce_delta_forces])

    print("Finish produce_delta_forces.py: {}".format(ctime()))
