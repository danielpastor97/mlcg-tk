import torch

import mdtraj as md
from typing import Any, List, Union, Tuple, Optional
import numpy as np

from mlcg.geometry._symmetrize import _symmetrise_distance_interaction
from networkx.algorithms.shortest_paths.unweighted import (
    bidirectional_shortest_path,
)
import networkx as nx
from mlcg.geometry.topology import (
    Topology,
    get_connectivity_matrix,
    get_n_paths,
)

from .utils import get_dihedral_groups, split_bulk_termini, with_attrs
from .embedding_maps import all_residues

class StandardBonds:
    nl_names = ["n_term_bonds", "bulk_bonds", "c_term_bonds", "bonds"]
    def __call__(
        self,
        topology: md.Topology,
        separate_termini: bool=True,
        **kwargs
) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        mlcg_top = Topology.from_mdtraj(topology)
        conn_mat = get_connectivity_matrix(mlcg_top).numpy()
        bond_edges = get_n_paths(conn_mat, n=2).numpy()
        if separate_termini:
            n_term_atoms, c_term_atoms = kwargs["n_term_atoms"], kwargs["c_term_atoms"]
            n_term_bonds, c_term_bonds, bulk_bonds = split_bulk_termini(
                n_term_atoms, c_term_atoms, bond_edges
            )

            if len(bulk_bonds) == 0:
                bonds = [("n_term_bonds", 2, n_term_bonds),
                        ("bulk_bonds", 2, torch.tensor([]).reshape(2, 0)),
                        ("c_term_bonds", 2, c_term_bonds)]

            elif len(n_term_bonds) == 0 or len(c_term_bonds) == 0:
                bonds = [("n_term_bonds", 2, torch.tensor([]).reshape(2, 0)),
                        ("bulk_bonds", 2, bulk_bonds),
                        ("c_term_bonds", 2, torch.tensor([]).reshape(2, 0))]
            else:
                bonds = [("n_term_bonds", 2, n_term_bonds),
                        ("bulk_bonds", 2, bulk_bonds),
                        ("c_term_bonds", 2, c_term_bonds)]

        else:
            bonds = ("bonds", 2, bond_edges)

        return bonds
    
    def get_fit_kwargs(self, nl_name):
        return {}


class StandardAngles:
    nl_names = ["n_term_angles", "bulk_angles", "c_term_angles", "angles"]
    def __call__(
        self,
        topology: md.Topology,
        separate_termini: bool=True,
        **kwargs
) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        mlcg_top = Topology.from_mdtraj(topology)
        conn_mat = get_connectivity_matrix(mlcg_top).numpy()
        angle_edges = get_n_paths(conn_mat, n=3).numpy()

        if separate_termini:
            n_term_atoms, c_term_atoms = kwargs["n_term_atoms"], kwargs["c_term_atoms"]
            n_term_angles, c_term_angles, bulk_angles = split_bulk_termini(
                n_term_atoms, c_term_atoms, angle_edges
            )
            if len(bulk_angles) == 0:
                angles = [("n_term_angles", 3, n_term_angles),
                        ("bulk_angles", 3, torch.tensor([]).reshape(3, 0)),
                        ("c_term_angles", 3, c_term_angles)]

            elif len(n_term_angles) == 0 or len(c_term_angles) == 0:
                angles = [("n_term_angles", 3, torch.tensor([]).reshape(3, 0)),
                        ("bulk_angles", 3, bulk_angles),
                        ("c_term_angles", 3, torch.tensor([]).reshape(3, 0))]
            else:
                angles = [("n_term_angles", 3, n_term_angles),
                        ("bulk_angles", 3, bulk_angles),
                        ("c_term_angles", 3, c_term_angles)]
        else:
            angles = ("angles", 3, angle_edges)

        return angles
    
    def get_fit_kwargs(self, nl_name):
        return {}


class Non_Bonded:
    nl_names = ["n_term_nonbonded", "bulk_nonbonded", "c_term_nonbonded", "non_bonded"]
    allow_fit_from_values = True
    def __call__(
        self,
        topology: md.Topology,
        bond_edges: Union[np.array, List, None]=None,
        angle_edges: Union[np.array, List, None]=None,
        min_pair: int=6,
        res_exclusion: int=1,
        separate_termini: bool=False,
        **kwargs
) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        mlcg_top = Topology.from_mdtraj(topology)
        fully_connected_edges = _symmetrise_distance_interaction(
            mlcg_top.fully_connected2torch()
        ).numpy()
        conn_mat = get_connectivity_matrix(mlcg_top).numpy()
        graph = nx.Graph(conn_mat)

        pairs_parsed = np.array([
            p for p in fully_connected_edges.T
                if (abs(topology.atom(p[0]).residue.index - topology.atom(p[1]).residue.index) >= res_exclusion)
                and (graph.has_edge(p[0], p[1]) == False or len(bidirectional_shortest_path(graph, p[0], p[1])) >= min_pair)
                and not np.all(bond_edges == p[:, None], axis=0).any()
                and not np.all(angle_edges[[0,2],:] == p[:, None], axis=0).any()
        ])

        non_bonded_edges = torch.tensor(pairs_parsed.T)
        non_bonded_edges = torch.unique(
            _symmetrise_distance_interaction(non_bonded_edges), dim=1
        ).numpy()

        if separate_termini:
            if "use_terminal_res" in kwargs and kwargs["use_terminal_res"] == True:
                n_atoms = kwargs["n_term_atoms"]
                c_atoms = kwargs["c_term_atoms"]
            else:
                n_atoms = kwargs["n_atoms"]
                c_atoms = kwargs["c_atoms"]
            n_term_nonbonded, c_term_nonbonded, bulk_nonbonded = split_bulk_termini(
                n_atoms, c_atoms, non_bonded_edges
            )
            return [("n_term_nonbonded", 2, n_term_nonbonded),
                    ("bulk_nonbonded", 2, bulk_nonbonded),
                    ("c_term_nonbonded", 2, c_term_nonbonded)]
        else:
            return ("non_bonded", 2, non_bonded_edges)
        
    def get_fit_kwargs(self, nl_name):
        return {}


class Phi:
    nl_names = [f"{res}_phi" for res in all_residues]
    def __call__(
        self,
        topology: md.Topology,
        **kwargs
) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        dihedral_dict = get_dihedral_groups(
            topology, atoms_needed=["C", "N", "CA", "C"], offset=[-1.,0.,0.,0.], tag="_phi"
        )
        dihedrals = []
        for res in all_residues:
            dihedral_tag = f"{res}_phi"
            if dihedral_tag in dihedral_dict:
                atom_groups = np.array(dihedral_dict[dihedral_tag])
                dihedrals.append((dihedral_tag, 4, torch.tensor(atom_groups).T))
            else:
                dihedrals.append((dihedral_tag, 4, torch.tensor([]).reshape(4, 0)))
        return dihedrals
    def get_fit_kwargs(self, nl_name):
        if nl_name == "PRO_phi":
            return {"n_degs": 1, "constrain_deg": 1}
        else:
            return {"n_degs": 3, "constrain_deg": 3}


class Psi:
    nl_names = [f"{res}_psi" for res in all_residues]
    def __call__(
        self,
        topology: md.Topology,
        **kwargs
) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        dihedral_dict = get_dihedral_groups(
            topology, atoms_needed=["N", "CA", "C", "N"], offset=[0.,0.,0.,1.], tag="_psi"
        )
        dihedrals = []
        for res in all_residues:
            dihedral_tag = f"{res}_psi"
            if dihedral_tag in dihedral_dict:
                atom_groups = np.array(dihedral_dict[dihedral_tag])
                dihedrals.append((dihedral_tag, 4, torch.tensor(atom_groups).T))
            else:
                dihedrals.append((dihedral_tag, 4, torch.tensor([]).reshape(4, 0)))
        return dihedrals
    def get_fit_kwargs(self, nl_name):
        return {"n_degs": 3, "constrain_deg": 3}


class Omega:
    nl_names = ["pro_omega", "non_pro_omega"]
    replace_gly_ca_stats = True
    def __call__(
        self,
        topology: md.Topology,
        **kwargs
) -> List[Tuple[str, int, torch.Tensor]]:
        dihedral_dict = get_dihedral_groups(
            topology, atoms_needed=["CA", "C", "N", "CA"], offset=[-1,-1,0,0], tag="_omega"
        )
        pro_omega = []
        non_pro_omega = []
        for dihedral_tag in dihedral_dict.keys():
            atom_groups = np.array(dihedral_dict[dihedral_tag])
            if dihedral_tag == "PRO_omega":
                pro_omega.extend(atom_groups)
            else:
                non_pro_omega.extend(atom_groups)
        dihedrals = []
        for dihedral in ["pro_omega", "non_pro_omega"]:
            if len(eval(dihedral)) == 0:
                dihedrals.append((dihedral, 4, torch.tensor([]).reshape(4, 0)))
            else:
                dihedrals.append((dihedral, 4, torch.tensor(np.array(eval(dihedral))).T))
        return dihedrals
    def get_fit_kwargs(self, nl_name):
        if nl_name == "pro_omega":
            return {"n_degs": 2, "constrain_deg": 2}
        else:
            return {"n_degs": 1, "constrain_deg": 1}
    

class Gamma1:
    nl_names = ["gamma_1"]
    def __call__(
        self,
        topology: md.Topology,
        **kwargs
) -> Tuple[str, int, torch.Tensor]:
        dihedral_dict = get_dihedral_groups(
            topology, atoms_needed=["N", "CB", "C", "CA"], offset=[0,0,0,0], tag="_gamma_1"
        )
        atom_groups = []
        for res in dihedral_dict:
            atom_groups.extend(dihedral_dict[res])
        if len(atom_groups) == 0:
            dihedrals = ("gamma_1", 4, torch.tensor([]).reshape(4, 0))
        else:
            dihedrals = ("gamma_1", 4, torch.tensor(np.array(atom_groups)).T)
        return dihedrals
    def get_fit_kwargs(self, nl_name):
        return {"n_degs": 1, "constrain_deg": 1}


class Gamma2:
    nl_names = ["gamma_2"]
    def __call__(
        self,
        topology: md.Topology,
        **kwargs
) -> Tuple[str, int, torch.Tensor]:
        dihedral_dict = get_dihedral_groups(
            topology, atoms_needed=["CA", "O", "N", "C"], offset=[0,0,1,0], tag="_gamma_2"
        )
        atom_groups = []
        for res in dihedral_dict:
            atom_groups.extend(dihedral_dict[res])
        if len(atom_groups) == 0:
            dihedrals = ("gamma_2", 4, torch.tensor([]).reshape(4, 0))
        else:
            dihedrals = ("gamma_2", 4, torch.tensor(np.array(atom_groups)).T)
        return dihedrals
    def get_fit_kwargs(self, nl_name):
        return {"n_degs": 1, "constrain_deg": 1}
