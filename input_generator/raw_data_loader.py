import numpy as np
import os
from natsort import natsorted
from glob import glob
import h5py
from typing import Tuple


def CATH_loader(base_dir: str,name:str,):
    #return sorted(name, key=alphanum_key)
    outputs_fns = natsorted(glob(
        os.path.join(base_dir, f"output/{name}/*_part_*")
        ))
    aa_coord_list = []
    aa_force_list = []
    # load the files, checking against the mol dictionary
    for fn in outputs_fns:
        output = np.load(fn)
        coord = output["coords"]
        coord = 10.0 * coord  # convert nm to angstroms
        force = output["Fs"]
        force = force / 41.84  # convert to from kJ/mol/nm to kcal/mol/ang
        assert coord.shape == force.shape
        aa_coord_list.append(coord)
        aa_force_list.append(force)
    aa_coords = np.concatenate(aa_coord_list)
    aa_forces = np.concatenate(aa_force_list)
    return aa_coords, aa_forces

def CATH_ext_loader(base_dir: str,name:str,):
    traj_dirs = glob(
        os.path.join(base_dir, f"group_*/{name}_*/")
        )
    all_coords = []
    all_forces = []
    for traj_dir in traj_dirs:
        traj_coords = []
        traj_forces = []
        fns = glob(
            os.path.join(traj_dir, "prod_out_full_output/*.npz")
        )
        fns.sort(key = lambda file : int(file.split("_")[-2]))
        last_parent_id = None
        for fn in fns:
            np_dict = np.load(fn,allow_pickle=True)
            current_id = np_dict['id']
            parent_id = np_dict['parent_id']
            if parent_id  is not None:
                assert parent_id == last_parent_id
            traj_coords.append(np_dict['coords'])
            traj_forces.append(np_dict['Fs'])
            last_parent_id = current_id
        traj_full_coords = np.concatenate(traj_coords)
        traj_full_forces = np.concatenate(traj_forces)
        if traj_full_coords.shape[0] != 25000:
            continue
        else:
            all_coords.append(traj_full_coords)
            all_forces.append(traj_full_forces)
    full_coords = np.concatenate(all_coords)
    full_forces = np.concatenate(all_forces)
    return full_coords, full_forces

def DIMER_loader(base_dir: str,name:str,):
    with h5py.File(
        os.path.join(base_dir, "allatom.h5"),
        "r"
        ) as data:
        coord = data["MINI"][name]["aa_coords"][:]
        force = data["MINI"][name]["aa_forces"][:]

    # convert to kcal/mol/angstrom and angstrom
    # from kJ/mol/nm and nm
    coord = coord*10
    force = force/41.84

    return coord, force

def DIMER_ext_loader(base_dir: str,name:str,):
    coord = np.load(
        glob(os.path.join(base_dir, f"dip_dimers_*/data/{name}_coord.npy"))[0],
        allow_pickle=True
        )
    force = np.load(
        glob(os.path.join(base_dir, f"dip_dimers_*/data/{name}_force.npy"))[0],
        allow_pickle=True
        )

    # convert to kcal/mol/angstrom and angstrom
    # from kJ/mol/nm and nm
    coord = coord*10
    force = force/41.84

    return coord, force

def Trpcage_loader(base_dir: str, name:str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        coords_fns = natsorted(glob(
            os.path.join(base_dir, f"coords_nowater/trp_coor_folding-trpcage_*.npy")
            ))

        forces_fns = [
            fn.replace("coords_nowater/trp_coor_folding","forces_nowater/trp_force_folding") for fn in coords_fns
        ]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn,ffn in zip(coords_fns, forces_fns):
            force = np.load(ffn)
            coord = np.load(cfn)
            coord = 10.0 * coord  # convert nm to angstroms
            force = force / 41.84  # convert to from kJ/mol/nm to kcal/mol/ang
            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces