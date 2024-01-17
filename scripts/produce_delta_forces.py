import mdtraj as md
from glob import glob
import pandas as pd
import numpy as np
import torch

from mlcg.data.atomic_data import AtomicData
from mlcg.datasets.utils import remove_baseline_forces

from input_generator.raw_dataset import *
from prior_terms import *

from tqdm import tqdm
import yaml
import argparse
import os


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Script for generating input CG data and prior neighbourlists for transferable datasets.",
    )
    parser.add_argument(
        "--data_dict",
        type=str,
        nargs="+",
        help="Configuration file containing sub-dataset parameters and prior terms.",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        help="Sample names for each sub-dataset in data_dict.",
    )
    parser.add_argument(
        "--prior_model",
        type=str,
        help="Pytorch file in which prior model is saved.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        required=False,
        type=str,
        help="Optional arguement to set device on which delta forces will be calculated."
    )
    return parser

if __name__ == "__main__":
    parser = parse_cli()
    args = parser.parse_args()

    device = args.device

    data_dict_list = [yaml.safe_load(open(dict, "rb")) for dict in args.data_dict]
    data_dict = {k: v for d in data_dict_list for k, v in d.items()}

    names_list = [yaml.safe_load(open(dict, "rb")) for dict in args.names]
    names = {k: v for d in names_list for k, v in d.items()}

    prior_model = torch.load(open(args.prior_model, "rb")).models.to(device)

    for dataset in data_dict.keys():
        sub_data_dict = data_dict[dataset]
        dataset_names = names[dataset]

        for name in tqdm(dataset_names, f"Producing delta forces for {dataset} dataset..."):
            data_list = []
            coords = np.load(
                os.path.join(
                    sub_data_dict["save_dir"],
                    f"{sub_data_dict['base_tag']}{name}_cg_coords.npy"
                    )
            )
            forces = np.load(
                os.path.join(
                    sub_data_dict["save_dir"],
                    f"{sub_data_dict['base_tag']}{name}_cg_forces.npy"
                    )
            )
            embeds = np.load(
                os.path.join(
                    sub_data_dict["save_dir"],
                    f"{sub_data_dict['base_tag']}{name}_cg_embeds.npy"
                    )
            )
            nls = pickle.load(open(
                os.path.join(
                    sub_data_dict["save_dir"],
                    f"{sub_data_dict['base_tag']}{name}_prior_nls_{sub_data_dict['prior_tag']}.pkl",
                    ),
                    "rb"
                ))
            num_frames = coords.shape[0]
            for i in range(num_frames):
                data = AtomicData.from_points(
                    pos=torch.tensor(coords[i]),
                    forces=torch.tensor(forces[i]),
                    atom_types=torch.tensor(embeds),
                    masses=None,
                    neighborlist=nls,
                )
                data_list.append(data)

            if "batch_size" in sub_data_dict:
                batch_size = sub_data_dict["batch_size"]
            else:
                batch_size = 100

            num_frames = coords.shape[0]
            delta_forces = []
            for i in range(0, num_frames, batch_size):
                sub_data_list = []
                for j in range(batch_size):
                    data = AtomicData.from_points(
                            pos=torch.tensor(coords[i+j]),
                            forces=torch.tensor(forces[i+j]),
                            atom_types=torch.tensor(embeds),
                            masses=None,
                            neighborlist=nls,
                    )
                    sub_data_list.append(data.to(device))
                sub_data_list = tuple(sub_data_list)
                _ = remove_baseline_forces(
                    sub_data_list,
                    prior_model,
                )
                for j in range(batch_size):
                    delta_force = sub_data_list[j].forces.detach().cpu()
                    delta_forces.append(delta_force.numpy())

            np.save(
                os.path.join(
                    sub_data_dict["save_dir"],
                    f"{sub_data_dict['base_tag']}_{name}_cg_forces.npy"
                ),
                np.concatenate(delta_forces, axis=0).reshape(*coords.shape),
            )
