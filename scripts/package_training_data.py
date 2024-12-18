import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import RawDataset
from tqdm import tqdm
from time import ctime
import numpy as np
import pickle as pck
from typing import List, Union, Optional
from sklearn.model_selection import train_test_split
from jsonargparse import CLI
from copy import deepcopy
import h5py
import yaml


def package_training_data(
    dataset_name: str,
    names: List[str],
    dataset_tag: str,
    force_tag: str,
    training_data_dir: str,
    save_dir: str,
    batch_size: int = 256,
    stride: int = 1,
    train_size: Optional[Union[float,int,None]] = 0.8,
    train_mols: Optional[List] = None,
    val_mols: Optional[List] = None,
    random_state: Optional[str] = None,
):
    """
    Computes structural features and accumulates statistics on dataset samples

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    dataset_tag : str
        Label given to all output files produced from dataset
    names : List[str]
        List of sample names
    force_tag : str
        Label given to produced delta forces and saved packaged data
    training_data_dir : str
        Path to directory from which input will be loaded
    save_dir : str
        Path to directory to which output will be saved
    batch_size : int
        Number of samples of dataset to include in each training batch
    stride : int
        Integer by which to stride frames
    train_size : Union[float, int]
        Either the proportion (if float) or number of samples (if int) of molecules in training data
        If None, lists should be supplied for training and validation samples
    train_mols : Optional[List]
        Molecules to be used for training set
    val_mols : Optional[List]
        Molecules to be used for validation set
    random_state : Optional[str]
        Controls shuffling applied to the data before applying the split
    """

    dataset = RawDataset(dataset_name, names, dataset_tag)

    # Create H5 of training data
    if force_tag != "":
        fnout_h5 = osp.join(save_dir, f"{dataset_name}_{force_tag}.h5")
        fnout_part = osp.join(save_dir, f"partition_{dataset_name}_{force_tag}.yaml")
    else:
        fnout_h5 = osp.join(save_dir, f"{dataset_name}.h5")
        fnout_part = osp.join(save_dir, f"partition_{dataset_name}.yaml")

    with h5py.File(fnout_h5, "w") as f:
        metaset = f.create_group(dataset_name)
        for samples in tqdm(
            dataset, f"Compute histograms of CG data for {dataset_name} dataset..."
        ):
            cg_coords, cg_delta_forces, cg_embeds = samples.load_training_inputs(
                training_data_dir=training_data_dir,
                force_tag=force_tag,
            )
        
            name = f"{samples.tag}{samples.name}"
            hdf_group = metaset.create_group(name)

            hdf_group.create_dataset(
                "cg_coords", data=cg_coords.astype(np.float32)
            )
            hdf_group.create_dataset(
                "cg_delta_forces", data=cg_delta_forces.astype(np.float32)
            )
            hdf_group.attrs["cg_embeds"] = cg_embeds
            hdf_group.attrs["N_frames"] = cg_coords.shape[0]
    
    # Create partition file
    if train_mols == None and val_mols == None:
        if train_size == None:
            raise ValueError("Either a train size or predefined lists for training and validation samples must be specified.")
        
        train_mols, val_mols = train_test_split(
            names,
            train_size=train_size,
            shuffle=True,
            random_state=random_state,
        )
    elif train_mols != None:
        val_mols = deepcopy(names).remove(train_mols)
    elif val_mols != None:
        train_mols = deepcopy(names).remove(val_mols)

    partition_opts = {"train": {}, "val": {}}
    
    # make training data partition
    partition_opts["train"]["metasets"] = {}
    partition_opts["train"]["metasets"][dataset_name] = {
        "molecules": train_mols,
        "stride": stride,
    }
    partition_opts["train"]["batch_sizes"] = {dataset_name: batch_size}

    # make validation data partition
    partition_opts["val"]["metasets"] = {}
    partition_opts["val"]["metasets"][dataset_name] = {
        "molecules": train_mols,
        "stride": stride,
    }
    partition_opts["val"]["batch_sizes"] = {dataset_name: batch_size}
    
    with open(fnout_part, "w") as ofile:
        yaml.dump(partition_opts, ofile)


def combine_datasets(
    dataset_names: List[str],
    save_dir: str,
    force_tag: Optional[str],
):
    """
    Computes structural features and accumulates statistics on dataset samples

    Parameters
    ----------
    dataset_names : List[str]
        List of dataset name to combine
    save_dir : str
        Path to directory from which datasets will be loaded and to which output will be saved
    force_tag : str
        Label given to produced delta forces and saved packaged data
    """

    datasets_label = "_".join(dataset_names)
    if force_tag != "":
        fnout_h5 = osp.join(save_dir, f"combined_{datasets_label}_{force_tag}.h5")
        fnout_part = osp.join(save_dir, f"partition_{datasets_label}_{force_tag}.yaml")
        data_fn = osp.join(save_dir, f"partition_%s_{force_tag}.yaml")
    else:
        fnout_h5 = osp.join(save_dir, f"combined_{datasets_label}.h5")
        fnout_part = osp.join(save_dir, f"partition_{datasets_label}.yaml")
        data_fn = osp.join(save_dir, f"partition_%s.yaml")

    with h5py.File(fnout_h5, "w") as f:
        for dataset in dataset_names:
            if force_tag != "":
                f[dataset] = h5py.ExternalLink(f"{dataset}_{force_tag}.h5", f"/{dataset}")
            else:
                f[dataset] = h5py.ExternalLink(f"{dataset}.h5", f"/{dataset}")

    partition_opts = {"train": {}, "val": {}}
    partition_opts["train"]["metasets"] = {}
    partition_opts["train"]["batch_sizes"] = {}
    partition_opts["val"]["metasets"] = {}
    partition_opts["val"]["batch_sizes"] = {}

    for dataset in dataset_names:
        with open(data_fn.format(dataset), "r") as ifile:
            data_partition = yaml.safe_load(ifile)
    
        # make training data partition
        partition_opts["train"]["metasets"][dataset] = {
            data_partition["train"]["metasets"][dataset]
        }
        partition_opts["train"]["batch_sizes"] = {
            dataset: data_partition["train"]["batch_sizes"]
        }

        # make validation data partition
        partition_opts["val"]["metasets"][dataset] = {
            data_partition["val"]["metasets"][dataset]
        }
        partition_opts["val"]["batch_sizes"] = {
            dataset: data_partition["val"]["batch_sizes"]
        }
    
    with open(fnout_part, "w") as ofile:
        yaml.dump(partition_opts, ofile)


if __name__ == "__main__":
    print("Start fit_priors.py: {}".format(ctime()))

    CLI([package_training_data, combine_datasets])

    print("Finish fit_priors.py: {}".format(ctime()))
