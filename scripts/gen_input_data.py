import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset
from input_generator.embedding_maps import (
    embedding_fivebead,
    CGEmbeddingMapFiveBead,
    CGEmbeddingMap,
)
from input_generator.raw_data_loader import DatasetLoader
from input_generator.prior_gen import Bonds, PriorBuilder
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable
from jsonargparse import CLI
import pickle as pck


def process_raw_dataset(
    dataset_name: str,
    names: List[str],
    sample_loader: DatasetLoader,
    raw_data_dir: str,
    tag: str,
    pdb_template_fn: str,
    save_dir: str,
    cg_atoms: List[str],
    embedding_map: CGEmbeddingMap,
    embedding_func: Callable,
    skip_residues: List[str],
    use_terminal_embeddings: bool,
    cg_mapping_strategy: str,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
):
    """_summary_

    Parameters
    ----------
    dataset_name : str
        _description_
    names : List[str]
        _description_
    sample_loader : Callable
        _description_
    raw_data_dir : str
        _description_
    tag : str
        _description_
    pdb_template_fn : str
        _description_
    save_dir : str
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
    prior_tag : str
        _description_
    prior_builders : List[PriorBuilder]
    """
    dataset = RawDataset(dataset_name, names, tag)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        samples.aa_traj, samples.top_dataframe = sample_loader.get_traj_top(
            samples.name, pdb_template_fn
        )

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

        prior_nls = samples.get_prior_nls(
            prior_builders, save_nls=True, save_dir=save_dir, prior_tag=prior_tag
        )

        aa_coords, aa_forces = sample_loader.load_coords_forces(
            raw_data_dir, samples.name
        )

        cg_coords, cg_forces = samples.process_coords_forces(
            aa_coords, aa_forces, mapping=cg_mapping_strategy
        )

        samples.save_cg_output(save_dir, save_coord_force=True)

        fn = osp.join(save_dir, f"{samples.name}_prior_builders_nl_{prior_tag}.pck")
        with open(fn, "wb") as f:
            pck.dump(prior_builders, f)


if __name__ == "__main__":
    print("Start gen_input_data.py: {}".format(ctime()))

    CLI([process_raw_dataset])

    print("Finish gen_input_data.py: {}".format(ctime()))
