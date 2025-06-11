import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import mdtraj as md
import warnings
from functools import wraps

from aggforce import (
    LinearMap,
    guess_pairwise_constraints,
    project_forces,
    constraint_aware_uni_map,
    qp_linear_map,
)


from .prior_gen import PriorBuilder

LIPID_MAPPINGS = {
        'DPPC': {
                'NC3' : ['N', 'C13', 'H13A', 'H13B', 'H13C', 'C14', 'H14A', 'H14B', 'H14C', 'C15', 'H15A', 'H15B', 'H15C', 'C12', 'H12A', 'H12B', 'C11', 'H11A', 'H11B'],      
                'PO4'  : ['P', 'O13', 'O14', 'O11', 'O12'],
                'GL1'  : ['C2', 'HS', 'O21', 'C21', 'O22', 'C1', 'HA', 'HB' ],
                'GL2'  : ['C3', 'HX', 'HY', 'O31', 'O32', 'C31'],
                'C1A'  : ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 'C25', 'H5R', 'H5S'],
                'C2A'  : ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 'C29', 'H9R', 'H9S'],
                'C3A'  : ['C210', 'H10R', 'H10S', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'H12S', 'C213', 'H13R', 'H13S'],
                'C4A'  : ['C214', 'H14R', 'H14S', 'C215', 'H15R', 'H15S', 'C216', 'H16R', 'H16S', 'H16T'],
                'C1B'  : ['C32', 'H2Y', 'H2X', 'C33', 'H3Y', 'H3X', 'C34', 'H4Y', 'H4X', 'C35', 'H5Y', 'H5X'],
                'C2B'  : ['C36', 'H6Y', 'H6X', 'C37', 'H7Y', 'H7X', 'C38', 'H8Y', 'H8X', 'C39', 'H9Y', 'H9X'],
                'C3B'  : ['C310', 'H10Y', 'H10X', 'C311', 'H11Y', 'H11X', 'C312', 'H12Y', 'H12X', 'C313', 'H13Y', 'H13X'],
                'C4B'  : ['C314', 'H14Y', 'H14X', 'C315', 'H15Y', 'H15X', 'C316', 'H16Y', 'H16X', 'H16Z'],
        },
        'POPC': {
                'NC3'  : ['N', 'C13', 'H13A', 'H13B', 'H13C', 'C14', 'H14A', 'H14B', 'H14C', 'C15', 'H15A', 'H15B', 'H15C', 'C12', 'H12A', 'H12B', 'C11', 'H11A', 'H11B'],  
                'PO4'  : ['P', 'O13', 'O14', 'O11', 'O12'],
                'GL1'  : ['C2', 'HS', 'O21', 'C21', 'O22', 'C1', 'HA', 'HB' ],
                'GL2'  : ['C3', 'HX', 'HY', 'O31', 'O32', 'C31'],
                'C1A'  : ['C22', 'H2R', 'H2S', 'C23', 'H3R', 'H3S', 'C24', 'H4R', 'H4S', 
                'C25', 'H5R', 'H5S'],
                'D2A'  : ['C26', 'H6R', 'H6S', 'C27', 'H7R', 'H7S', 'C28', 'H8R', 'H8S', 
                'C29', 'H91', 'H101'],
                'C3A'  : ['C210', 'C211', 'H11R', 'H11S', 'C212', 'H12R', 'H12S', 'C213',
                'H13R', 'H13S', 'C214', 'H14R', 'H14S'],
                'C4A'  : ['C215', 'H15R', 'H15S', 'C216', 'H16R', 'H16S', 'C217', 'H17R',
                'H17S', 'C218', 'H18R', 'H18S', 'H18T'],
                'C1B'  : ['C32', 'H2Y', 'H2X', 'C33', 'H3Y', 'H3X', 'C34', 'H4Y', 'H4X', 
                'C35', 'H5Y', 'H5X'],
                'C2B'  : ['C36', 'H6Y', 'H6X', 'C37', 'H7Y', 'H7X', 'C38', 'H8Y', 'H8X', 
                'C39', 'H9Y', 'H9X'],
                'C3B'  : ['C310', 'H10Y', 'H10X', 'C311', 'H11Y', 'H11X', 'C312', 'H12Y',
                'H12X', 'C313', 'H13Y', 'H13X'],
                'C4B'  : ['C314', 'H14Y', 'H14X', 'C315', 'H15Y', 'H15X', 'C316', 'H16Y',
                'H16X', 'H16Z']
        }
}

LIPID_BONDS = {
    'DPPC': [
        (0, 1), # N-P
        (1, 2), # P-C2
        (1, 3), # P-C3
        (2, 3), # C2-C3
        (2, 4), # C2-C24
        (4, 5), # C24-C28
        (5, 6), # C28-C212
        (6, 7), # C212-C216
        (3, 8), # C3-C34
        (8, 9), # C34-C38
        (9, 10), # C38-C312
        (10, 11), # C312-C316
    ],
    'POPC': [
        (0, 1), # N-P
        (1, 2), # P-C2
        (1, 3), # P-C3
        (2, 3), # C2-C3
        (2, 4), # C2-C24
        (4, 5), # C24-C28
        (5, 6), # C28-C212
        (6, 7), # C212-C216
        (3, 8), # C3-C34
        (8, 9), # C34-C38
        (9, 10), # C38-C312
        (10, 11), # C312-C316
    ],
}

LIPID_MASSES = {
        'NC3':  72,
        'PO4':  72,
        'GL1':  54,
        'GL2':  54,
        'C1A':  72,
        'D2A':  54,
        'C3A':  72,
        'C4A':  72,
        'C1B':  72,
        'C2B':  72,
        'C3B':  72,
        'C4B':  72,
}

def with_attrs(**func_attrs):
    """Set attributes in the decorated function, at definition time.
    Only accepts keyword arguments.
    """

    def attr_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        for attr, value in func_attrs.items():
            setattr(wrapper, attr, value)

        return wrapper

    return attr_decorator


def get_output_tag(tag_label: Union[List, str], placement: str = "before"):
    """
    Helper function for combining output tag labels neatly.
    Fixes issues of connecting/preceding '_' being included in some labels but not others.

    Parameters
    ----------
    tag_label : List, str
        Either a list of labels to include (ex: for datasets, delta force computation) or individual label item.
    placement : str
        Placement of tag in output name. One of: 'before', 'after'.
    """

    if isinstance(tag_label, str):
        if tag_label in [None, "", " "]:
            return ""
        else:
            return f"_{tag_label.strip('_')}"
    elif isinstance(tag_label, List):
        for l in tag_label:
            if l in [None, "", " "]:
                tag_label.remove(l)
        joined_label = "_".join([l.strip("_") for l in tag_label])
        if placement == "before":
            return f"{joined_label}_"
        elif placement == "after":
            return f"_{joined_label}"
        else:
            raise ValueError("Please specify placement from: 'before', 'after'.")


def map_cg_topology(
    atom_df: pd.DataFrame,
    cg_atoms: List[str],
    embedding_function: str,
    skip_residues: Optional[Union[List, str]] = None,
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


def batch_matmul(map_matrix, X, batch_size):
    """
    Perform matrix multiplication in chunks.

    Parameters:
      map_matrix: np.ndarray of shape (N_CG_ats, N_FG_ats)
      X: np.ndarray of shape (M_frames, N_FG_ats, 3)
      batch_size: int, the number of rows (from the M dimension) to process at a time.

    Returns:
      result: np.ndarray of shape (M_frames, N_CG_ats, 3)
    """
    results = []
    M = X.shape[0]
    for i in range(0, M, batch_size):
        # Slice a batch along the M dimension
        X_batch = X[i : i + batch_size]  # shape: (batch, N, 3)
        # Perform matrix multiplication:
        # map_matrix (CG, FG) multiplied by each X_batch (FG, 3) gives (GC, 3) for each sample.
        # The broadcasting ensures the result is (batch, CG, 3)
        result_batch = map_matrix @ X_batch
        results.append(result_batch)
    # Concatenate all chunks along the first axis (M dimension)
    return np.concatenate(results, axis=0)


def chunker(array, n_batches):
    """
    Chunks an input array into a specified number of batches.

    This function divides the input array into approximately equal-sized chunks.
    The last chunk may contain more elements if the array length is not perfectly
    divisible by the number of batches.

    Parameters:
    -----------
    array : np.ndarray or List
        The input array to be chunked.
    n_batches : int
        The number of batches to divide the array into. Must be a positive
        integer and less than or equal to the length of the array.

    Returns:
    --------
    batched_array: List
        A list of lists/arrays, where each inner list/array is a chunk of the original array.

    Examples:
    >>> chunker([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    >>> chunker([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4, 5]]

    >>> chunker([1, 2, 3, 4, 5], 5)
    [[1], [2], [3], [4], [5]]

    >>> chunker([1, 2, 3, 4, 5], 1)
    [[1, 2, 3, 4, 5]]
    """
    if n_batches == 1:
        return [array]
    assert n_batches <= len(
        array
    ), "n_batches needs to be smaller than the array to chunk"
    batched_array = []
    n_elts_per_batch = len(array) // n_batches
    for i in range(n_batches - 1):
        batched_array.append(array[i * n_elts_per_batch : (i + 1) * n_elts_per_batch])
    # last batch might be larger, it contains the rest of the elements in the array
    batched_array.append(array[(i + 1) * n_elts_per_batch :])
    return batched_array


def slice_coord_forces(
    coords,
    forces,
    cg_map,
    mapping: str = "slice_aggregate",
    force_stride: int = 100,
    batch_size: Optional[int] = None,
) -> Tuple:
    """
    Parameters
    ----------
    coords: [n_frames, n_atoms, 3]
        Numpy array of atomistic coordinates
    forces: [n_frames, n_atoms, 3]
        Numpy array of atomistic forces
    cg_map: [n_cg_atoms, n_atomistic_atoms]
        Linear map characterizing the atomistic to CG configurational map with shape.
    mapping:
        Mapping scheme to be used,
        Can be either a string, then must be either 'slice_aggregate' or 'slice_optimize',
        Or can be directly a numpy array to use for projection
    force_stride:
        Striding to use for force projection results
    batch_size:
        Optional length of batch in which divide the AA mapping of coords and forces
        to CG ones

    Returns
    -------
    Coarse-grained coordinates and forces
    """
    config_map = LinearMap(cg_map)
    config_map_matrix = config_map.standard_matrix
    # taking only first 100 frames every 2 frames gives same results in ~1/15th of time
    constraints = guess_pairwise_constraints(coords[:100:2], threshold=5e-3)
    if isinstance(mapping, str):
        if mapping == "slice_aggregate":
            method = constraint_aware_uni_map
            force_agg_results = project_forces(
                coords=coords[::force_stride],
                forces=forces[::force_stride],
                coord_map=config_map,
                constrained_inds=constraints,
                method=method,
            )
        elif mapping == "slice_optimize":
            method = qp_linear_map
            l2 = 1e3
            force_agg_results = project_forces(
                coords=coords[::force_stride],
                forces=forces[::force_stride],
                coord_map=config_map,
                constrained_inds=constraints,
                method=method,
                l2_regularization=l2,
            )
        else:
            raise RuntimeError(
                f"Force mapping {mapping} is neither 'slice_aggregate' nor 'slice_optimize'."
            )
        force_map_matrix = force_agg_results["tmap"].force_map.standard_matrix
    elif isinstance(mapping, np.ndarray):
        force_map_matrix = mapping
    else:
        raise RuntimeError(
            f"Force mapping {mapping} is neither a string nor a numpy array."
        )

    if batch_size != None:
        cg_coords = batch_matmul(config_map_matrix, coords, batch_size=batch_size)
        cg_forces = batch_matmul(force_map_matrix, forces, batch_size=batch_size)
    else:
        cg_coords = config_map_matrix @ coords
        cg_forces = force_map_matrix @ forces

    return cg_coords, cg_forces, force_map_matrix


def filter_cis_frames(
    coords: np.ndarray, forces: np.ndarray, topology: md.Topology, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    filters out frames containing cis-omega angles

    Parameters
    ----------
    coords: [n_frames, n_atoms, 3]
        Non-filtered atomistic coordinates
    forces: [n_frames, n_atoms, 3]
        Non-filtered atomistic forces
    topology:
        mdtraj topology to load the coordinates with
    verbose:
        If True, will print a warning containing the number of discarded frames for this sample

    Returns
    -------
    Tuple of np.ndarray's for filtered coarse grained coordinates and forces
    """
    min_omega_atoms = set(["N", "CA", "C"])
    unique_atom_types = set([atom.name for atom in topology.atoms])
    if not min_omega_atoms.issubset(unique_atom_types):
        raise ValueError(
            "Provided pdb file must contain at least N, CA and C atoms for cis-omega filtering"
        )

    cis_omega_mask = np.zeros(coords.shape[0], dtype=bool)
    md_traj = md.Trajectory(coords, topology)

    omega_idx, omega_values = md.compute_omega(md_traj)

    cis_omega_threshold = 1.0  # rad
    mask = np.all(np.abs(omega_values) > 1, axis=1)
    if not np.all(mask):
        warnings.warn(f"Discarding {len(mask) - np.sum(mask)} cis frames")
    if np.sum(mask) == 0:
        warnings.warn(f"This amounts to removing all frames for this molecule")

    return coords[mask], forces[mask]


def get_terminal_atoms(
    prior_builder: PriorBuilder,
    cg_dataframe: pd.DataFrame,
    N_term: Union[None, str] = None,
    C_term: Union[None, str] = None,
) -> Dict:
    """
    Parameters
    ----------
    prior_builder:

    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).
    N_term: (Optional)
        Atom used in definition of N-terminus embedding.
    C_term: (Optional)
        Atom used in definition of C-terminus embedding.
    """
    chains = cg_dataframe.chainID.unique()
    # all atoms belonging to monopeptide chains will be removed from termini list
    monopeptide_atoms = []
    for chain in chains:
        residues = cg_dataframe.loc[cg_dataframe.chainID == chain].resSeq.unique()
        if len(residues) == 1:
            monopeptide_atoms.extend(
                cg_dataframe.loc[cg_dataframe.chainID == chain].index.to_list()
            )

    n_term_atoms = []
    c_term_atoms = []

    for chain in chains:
        chain_filter = cg_dataframe["chainID"] == chain
        first_res_chain, last_res_chain = (
            cg_dataframe[chain_filter]["resSeq"].min(),
            cg_dataframe[chain_filter]["resSeq"].max(),
        )
        n_term_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == first_res_chain) & chain_filter
            ].index.to_list()
        )
        c_term_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == last_res_chain) & chain_filter
            ].index.to_list()
        )

    prior_builder.n_term_atoms = [a for a in n_term_atoms if a not in monopeptide_atoms]
    prior_builder.c_term_atoms = [a for a in c_term_atoms if a not in monopeptide_atoms]

    N_term_name = "N" if N_term is None else N_term
    C_term_name = "C" if C_term is None else C_term

    n_term_name_atoms = []
    c_term_name_atoms = []
    for chain in chains:
        chain_filter = cg_dataframe["chainID"] == chain
        first_res_chain, last_res_chain = (
            cg_dataframe[chain_filter]["resSeq"].min(),
            cg_dataframe[chain_filter]["resSeq"].max(),
        )
        n_term_name_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == first_res_chain)
                & (cg_dataframe["name"] == N_term_name)
                & chain_filter
            ].index.to_list()
        )

        c_term_name_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == last_res_chain)
                & (cg_dataframe["name"] == C_term_name)
                & chain_filter
            ].index.to_list()
        )

    prior_builder.n_atoms = n_term_name_atoms
    prior_builder.c_atoms = c_term_name_atoms

    return prior_builder


def get_edges_and_orders(
    prior_builders: List[PriorBuilder],
    topology: md.Topology,
) -> List:
    """
    Parameters
    ----------
    prior_builders:
        List of PriorBuilder's to use for defining neighbour lists
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
    bond_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "bonds"
    ]
    all_bond_edges = []
    for prior_builder in bond_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
            all_bond_edges.extend([p[2] for p in edges_and_orders])
        else:
            all_edges_and_orders.append(edges_and_orders)
            all_bond_edges.append(edges_and_orders[2])

    # process angle priors
    angle_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "angles"
    ]
    all_angle_edges = []
    for prior_builder in angle_builders:
        edges_and_orders = prior_builder.build_nl(topology)
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

    nonbonded_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "non_bonded"
    ]
    for prior_builder in nonbonded_builders:
        edges_and_orders = prior_builder.build_nl(
            topology, bond_edges=all_bond_edges, angle_edges=all_angle_edges
        )
        # edges_and_orders = prior_dict[nbdict]["prior_function"](topology, all_bond_edges, all_angle_edges, **prior_dict[nbdict])
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)
    # process dihedral priors
    dihedral_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "dihedrals"
    ]
    for prior_builder in dihedral_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)

    return all_edges_and_orders


def split_bulk_termini(N_term, C_term, all_edges) -> Tuple:
    """
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
    """
    n_term_idx = np.where(np.isin(all_edges.T, N_term))
    n_term_edges = all_edges[:, np.unique(n_term_idx[0])]

    c_term_idx = np.where(np.isin(all_edges.T, C_term))
    c_term_edges = all_edges[:, np.unique(c_term_idx[0])]

    term_edges = np.concatenate([n_term_edges, c_term_edges], axis=1)
    bulk_edges = np.array(
        [e for e in all_edges.T if not np.all(term_edges == e[:, None], axis=0).any()]
    ).T

    return n_term_edges, c_term_edges, bulk_edges


def get_dihedral_groups(
    top: md.Topology, atoms_needed: List[str], offset: List[int], tag: Optional[str]
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
    for chain_idx, chain in enumerate(res_per_chain):
        for res in chain:
            res_idx = chain.index(res)
            if any(res_idx + ofs < 0 or res_idx + ofs >= len(chain) for ofs in offset):
                continue
            if any(atom not in [a.name for a in res.atoms] for atom in atoms_needed):
                continue
            label = f"{res.name}{tag}"
            if label not in atom_groups:
                atom_groups[label] = []
            dihedral = []
            for i, atom in enumerate(atoms_needed):
                atom_idx = top.select(
                    f"(chainid {chain_idx}) and (resid {res.index+offset[i]}) and (name {atom})"
                )
                dihedral.append(atom_idx)
            atom_groups[label].append(np.concatenate(dihedral))

    return atom_groups

def normalize_to_one(numbers):
    total = sum(numbers)
    if total == 0:
        raise ValueError("Sum of the input list is zero, cannot normalize.")
    
    # Normalize the numbers
    normalized = [x / total for x in numbers]
    
    # Adjust the last element to account for floating-point inaccuracies
    correction = 1 - sum(normalized)
    normalized[-1] += correction
    
    return normalized

def add_bonds_to_cg_topology(cg_top: md.Topology):
    """
    Parameters
    ----------
    cg_top:
        MDTraj topology object of CG topology.

    Returns
    -------
    MDTraj topology object with bonds added.
    """
    # add bonds to CG topology
    for residue in cg_top.residues:
        res_atoms = [a for a in residue.atoms]
        for bond in LIPID_BONDS[residue.name]:
            cg_top.add_bond(res_atoms[bond[0]], res_atoms[bond[1]])

    return cg_top   