import mdtraj as md
from glob import glob
import pandas as pd
import numpy as np

from sample import *
from prior_terms import *

from tqdm import tqdm
import yaml
import argparse


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Script for generating input CG data and prior neighbourlists for transferable datasets.",
    )
    parser.add_argument(
        "--data_dict", 
        type=str,
        help="Configuration file containing sub-dataset parameters and prior terms.", 
    )
    parser.add_argument(
        "--names", 
        type=str,
        help="Sample names for each sub-dataset in data_dict.", 
    )
    parser.add_argument(
        "--prior_dict", 
        type=str,
        help="Dictionary of prior terms and their corresponding parameters.", 
    )
    return parser


if __name__ == "__main__":
    parser = parse_cli()
    args = parser.parse_args()

    data_dict = yaml.safe_load(open(args.data_dict, "rb"))
    names = yaml.safe_load(open(args.names, "rb"))
    prior_dict = yaml.safe_load(open(args.prior_dict, "rb"))

    for dataset in data_dict.keys():
        sub_data_dict = data_dict[dataset]
        try:
            sample_class = eval(sub_data_dict["sample_class"])
        except NameError:
            print(f"The training sample class {dataset} has not been implemented and will be skipped.")
            continue

        dataset_names = names[dataset]

        for name in tqdm(dataset_names[:5]):
            # create sample object
            pdb_file = sub_data_dict["pdb_file"].format(name)
            
            sample = sample_class(
                name=name, 
                tag=sub_data_dict["base_tag"],
                pdb_fn=pdb_file,
                )

            # apply cg mapping
            sample.apply_cg_mapping(
                cg_atoms=sub_data_dict["cg_atoms"], 
                embedding_function=sub_data_dict["embedding_function"],
                embedding_dict=sub_data_dict["embedding_dict"],
                skip_residues=sub_data_dict["skip_residues"]
            )

            if sub_data_dict["use_terminal_embeddings"]: 
                sample.add_terminal_embeddings(
                    N_term=sub_data_dict["N_term"],
                    C_term=sub_data_dict["C_term"]
                )

            aa_coords, aa_forces =  sample.load_coords_forces(sub_data_dict["base_dir"])
            cg_coords, cg_forces = sample.process_coords_forces(aa_coords, aa_forces, mapping=sub_data_dict["mapping"])

            sample.save_cg_output(sub_data_dict["save_dir"], save_coord_force=True)

            prior_nls = sample.get_prior_terms(
                prior_dict, 
                save_nls=True, 
                save_dir=sub_data_dict["save_dir"], 
                prior_tag=sub_data_dict["prior_tag"]
            )