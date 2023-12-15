import torch
from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import *
from mlcg.nn.gradients import *
from mlcg.datasets.utils import remove_baseline_forces, chunker
import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import mdtraj as md

from aggforce import linearmap as lm
from aggforce import agg as ag
from aggforce import constfinder as cf

from embedding_maps import *


def map_cg_topology(
        atom_df: pd.DataFrame, 
        cg_atoms: List[str], 
        embedding_function: str,
        skip_residues: Optional[Union[List,str]] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    atom_df: 
        Pandas DataFrame row from mdTraj topology.
    cg_atoms:
        List of atoms needed in CG mapping.
    embedding_function:
        Function that slices coodinates, if not provided will fail.
    special_typing:
        Optional dictionary of alternative atom properties to use in assigning types instead of atom names.
    skip_residues:
        Optional list of residues to skip when assigning CG atoms (can be used to skip caps for example);
        As of now, this skips all instances of a given residue.
    
    
    Returns
    -------
    New DataFrame columns indicating atom involvement in CG mapping and type assignment.

    Example
    -------
    First obtain a Pandas DataFrame object using the built-in MDTraj function:
    >>> top_df = aa_traj.topology.to_dataframe()[0]
    
    For a five-bead resolution mapping without including caps:
    >>> cg_atoms = ["N", "CA", "CB", "C", "O"]
    >>> embedding_function = embedding_fivebead
    >>> skip_residues = ["ACE", "NME"]

    Apply row-wise function:
    >>> top_df = top_df.apply(map_cg_topology, axis=1, cg_atoms, embedding_dict, skip_residues)
    """
    if isinstance(embedding_function, str):
        try:
            embedding_function = eval(embedding_function)
        except NameError: 
            print("The specified embedding function has not been defined.")
            exit
    name, res = atom_df["name"], atom_df["resName"]
    if skip_residues != None and res in skip_residues:
        atom_df["mapped"] = False
        atom_df["type"] = "NA"
    
    else:
        if name in cg_atoms:
            atom_df["mapped"] = True
            atom_type = embedding_function(atom_df)
            atom_df["type"] = atom_type
        else:
            atom_df["mapped"] = False
            atom_df["type"] = "NA"
    
    return atom_df


def slice_coord_forces(
        coords,
        forces,
        cg_map,
        mapping: str="slice_aggregate",
        force_stride: int=100
) -> Tuple:
    '''
    Parameters
    ----------
    coords: [n_frames, n_atoms, 3]
        Numpy array of atomistic coordinates
    forces: [n_frames, n_atoms, 3]
        Numpy array of atomistic forces
    cg_map: [n_cg_atoms, n_atomistic_atoms]
        Linear map characterizing the atomistic to CG configurational map with shape.
    mapping:
        Mapping scheme to be used, must be either 'slice_aggregate' or 'slice_optimize'.
    force_stride:
        Striding to use for force projection results
    
    Returns
    -------
    Coarse-grained coordinates and forces 
    '''
    config_map = lm.LinearMap(cg_map)
    config_map_matrix = config_map.standard_matrix
    # taking only first 100 frames gives same results in ~1/15th of time
    constraints = cf.guess_pairwise_constraints(
        coords[:100], threshold=5e-3
        )
    if mapping == "slice_aggregate":
        method = lm.constraint_aware_uni_map
        force_agg_results = ag.project_forces(
            xyz=None,
            forces=forces[::force_stride],
            config_mapping=config_map,
            constrained_inds=constraints,
            method=method,
        )
    elif mapping == "slice_optimize":
        method = lm.qp_linear_map
        l2=1e3
        force_agg_results = ag.project_forces(
            xyz=None,
            forces=forces[::force_stride],
            config_mapping=config_map,
            constrained_inds=constraints,
            method=method,
            l2_regularization=l2
        )
    else:
        raise RuntimeError(f"Force mapping {mapping} is neither 'slice_aggregate' nor 'slice_optimize'.")
    
    force_map_matrix = force_agg_results["map"].standard_matrix
    cg_coords = config_map_matrix @ coords
    cg_forces = force_map_matrix @ forces

    return cg_coords, cg_forces


def get_terminal_atoms(
        prior_dict: Dict, 
        cg_dataframe: pd.DataFrame, 
        N_term: Union[None,str]=None, 
        C_term: Union[None,str]=None, 
) -> Dict:
    """
    Parameters
    ----------
    prior_dict:
        Dictionary of prior terms and their corresponding parameters.
    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).
    N_term: (Optional)
        Atom used in definition of N-terminus embedding.
    C_term: (Optional)
        Atom used in definition of C-terminus embedding.
    """
    for prior in prior_dict: 
        if "separate_termini" in prior_dict[prior] and prior_dict[prior]["separate_termini"] == True:
            first_res, last_res = cg_dataframe["resSeq"].min(), cg_dataframe["resSeq"].max()
            prior_dict[prior]["n_term_atoms"] = cg_dataframe.loc[(cg_dataframe["resSeq"] == first_res)].index.to_list()
            prior_dict[prior]["c_term_atoms"] = cg_dataframe.loc[(cg_dataframe["resSeq"] == last_res)].index.to_list()
            if N_term != None:
                prior_dict[prior]["n_atoms"] = cg_dataframe.loc[
                    (cg_dataframe["resSeq"] == first_res) & (cg_dataframe["name"] == N_term)
                    ].index.to_list()
            else:
                prior_dict[prior]["n_atoms"] = cg_dataframe.loc[
                    (cg_dataframe["resSeq"] == first_res) & (cg_dataframe["name"] == "N")
                    ].index.to_list() 
            if N_term != None:
                prior_dict[prior]["c_atoms"] = cg_dataframe.loc[
                    (cg_dataframe["resSeq"] == last_res) & (cg_dataframe["name"] == C_term)
                    ].index.to_list()
            else:
                prior_dict[prior]["c_atoms"] = cg_dataframe.loc[
                    (cg_dataframe["resSeq"] == first_res) & (cg_dataframe["name"] == "C")
                    ].index.to_list() 
    
    return prior_dict
                

def get_edges_and_orders(
        prior_dict: Dict, 
        topology: md.Topology,
) -> List:
    """
    Parameters
    ----------
    prior_dict:
        Dictionary of prior terms and their corresponding parameters.
    topology:
        MDTraj topology object from which atom groups defining each prior term will be created.
    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).
    
    Returns
    -------
    List of edges, orders, and tag for each prior term specified in prior_dict.
    """
    all_edges_and_orders = []
    # process bond priors
    bond_dicts = [prior for prior in prior_dict if prior_dict[prior]["type"] == "bonds"]
    all_bond_edges = []
    if len(bond_dicts) != 0:
        for bdict in bond_dicts:
            edges_and_orders = prior_dict[bdict]["prior_function"](topology, **prior_dict[bdict])
            if isinstance(edges_and_orders, list):
                all_edges_and_orders.extend(edges_and_orders)
                all_bond_edges.extend([p[2] for p in edges_and_orders])
            else:
                all_edges_and_orders.append(edges_and_orders)
                all_bond_edges.append(edges_and_orders[2])

    # process angle priors
    angle_dicts = [prior for prior in prior_dict if prior_dict[prior]["type"] == "angles"]
    all_angle_edges = []
    if len(angle_dicts) != 0:
        for adict in angle_dicts:
            edges_and_orders = prior_dict[adict]["prior_function"](topology, **prior_dict[adict])
            if isinstance(edges_and_orders, list):
                all_edges_and_orders.extend(edges_and_orders)
                all_angle_edges.extend([p[2] for p in edges_and_orders])
            else:
                all_edges_and_orders.append(edges_and_orders)
                all_angle_edges.append(edges_and_orders[2])
    
    # get nonbonded priors using bonded and angle edges
    if len(all_bond_edges) != 0:
        all_bond_edges = np.concatenate(all_bond_edges, axis=1)
    if len(all_angle_edges) != 0:
        all_angle_edges = np.concatenate(all_angle_edges, axis=1)

    nonbonded_dicts = [prior for prior in prior_dict if prior_dict[prior]["type"] == "non_bonded"]
    for nbdict in nonbonded_dicts:
        edges_and_orders = prior_dict[nbdict]["prior_function"](topology, all_bond_edges, all_angle_edges, **prior_dict[nbdict])
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)
    
    # process dihedral priors
    dihedral_dicts = [prior for prior in prior_dict if prior_dict[prior]["type"] == "dihedrals"]
    for dihdict in dihedral_dicts:
        edges_and_orders = prior_dict[dihdict]["prior_function"](topology, **prior_dict[dihdict])
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)
    
    return all_edges_and_orders


def split_bulk_termini(
        N_term, 
        C_term, 
        all_edges
) -> Tuple:
    '''
    Parameters
    ----------
    N_term: 
        List of atom indices to be split as part of the N-terminal.
    C_term: 
        List of atom indices to be split as part of the C-terminal.
    all_edges:
        All atom groups forming part of prior term.
    
    Returns
    -------
    Separated edges for bulk and terminal groups
    '''
    n_term_idx = np.where(np.isin(all_edges.T, N_term))
    n_term_edges = all_edges[:,np.unique(n_term_idx[0])]

    c_term_idx = np.where(np.isin(all_edges.T, C_term))
    c_term_edges = all_edges[:,np.unique(c_term_idx[0])]

    term_edges = np.concatenate([n_term_edges, c_term_edges], axis=1)
    bulk_edges = np.array([e for e in all_edges.T if not np.all(term_edges == e[:, None], axis=0).any()]).T

    return n_term_edges, c_term_edges, bulk_edges

def get_dihedral_groups(
        top: md.Topology, 
        atoms_needed: List[str], 
        offset: List[int], 
        tag: Optional[str]
) -> Dict:
    """
    Parameters
    ----------
    top:
        MDTraj topology object.
    atoms_needed: [4]
        Names of atoms forming dihedrals, should correspond to existing atom name in topology.
    offset: [4]
        Residue offset of each atom in atoms_needed from starting point.
    tag:
        Dihedral prior tag.
    
    Returns
    -------
    Dictionary of atom groups for each residue corresponding to dihedrals.

    Example
    -------
    To obtain all phi dihedral atom groups for a backbone-preserving resolution:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["C", "N", "CA", "C"], offset=[-1.,0.,0.,0.], tag="_phi"
    >>> )

    For a one-bead-per-residue mapping with only CA atoms preserved:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["CA", "CA", "CA", "CA"], offset=[-3.,-2.,-1.,0.]
    >>> )
    """
    res_per_chain = [[res for res in chain.residues] for chain in top.chains]
    atom_groups = {}
    for chain_idx,chain in enumerate(res_per_chain):
        for res in chain:
            if any(res.index+ofs < 0 or res.index+ofs >= len(chain) for ofs in offset):
                continue
            if any(atom not in [a.name for a in res.atoms] for atom in atoms_needed):
                continue
            label = f"{res.name}{tag}"
            if label not in atom_groups:
                atom_groups[label] = []
            dihedral = []
            for i,atom in enumerate(atoms_needed):
                atom_idx = top.select(f"(chainid {chain_idx}) and (resid {res.index+offset[i]}) and (name {atom})")
                dihedral.append(atom_idx)
            atom_groups[label].append(np.concatenate(dihedral))
    
    return atom_groups
