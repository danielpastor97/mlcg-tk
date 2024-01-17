import mdtraj as md
import pickle

from typing import List, Dict, Tuple, Optional, Union
from copy import deepcopy
import numpy as np
import torch
import warnings
import os

from mlcg.neighbor_list.neighbor_list import make_neighbor_list

from .utils import map_cg_topology, slice_coord_forces, get_terminal_atoms, get_edges_and_orders


class SampleCollection:
    """
    Input generation object for loading, manupulating, and saving training data samples.

    Parameters
    ----------
    name:
        String associated with atomistic trajectory output.
    tag:
        String to identify dataset in output files.
    pdb_fn:
        File location of atomistic structure to be used for topology.
    """
    def __init__(
            self,
            name: str,
            tag: str,
            pdb_fn: str
    ) -> None:
        self.name = name
        self.tag = tag
        pdb = md.load(pdb_fn.format(name))
        self.aa_traj = pdb.atom_slice([a.index for a in pdb.topology.atoms if a.residue.is_protein])
        self.top_dataframe = self.aa_traj.topology.to_dataframe()[0]

    def apply_cg_mapping(
            self,
            cg_atoms: List[str],
            embedding_function: str,
            embedding_dict: str,
            skip_residues: Optional[List[str]]=None
    ):
        """
        Applies mapping function to atomistic topology to obtain CG representation.

        Parameters
        ----------
        cg_atoms:
            List of atom names to preserve in CG representation.
        embedding_function:
            Name of function (should be defined in embedding_maps) to apply CG mapping.
        embedding_dict:
            Name of dictionary (should eb defined in embedding_maps) to define embeddings of CG beads.
        skip_residues: (Optional)
            List of residue names to skip (can be used to skip terminal caps, for example).
            Currently, can only be used to skip all residues with given name.
        """
        if isinstance(embedding_dict, str):
            self.embedding_dict = eval(embedding_dict)

        self.top_dataframe = self.top_dataframe.apply(
            map_cg_topology,
            axis=1,
            cg_atoms=cg_atoms,
            embedding_function=embedding_function,
            skip_residues=skip_residues
        )
        cg_df= deepcopy(self.top_dataframe.loc[self.top_dataframe["mapped"] == True])

        cg_atom_idx = cg_df.index.values.tolist()
        self.cg_atom_indices = cg_atom_idx

        cg_df.index = [i for i in range(len(cg_df.index))]
        cg_df.serial = [i+1 for i in range(len(cg_df.index))]
        self.cg_dataframe = cg_df

        cg_map = np.zeros((len(cg_atom_idx), self.aa_traj.n_atoms))
        cg_map[[i for i in range(len(cg_atom_idx))], cg_atom_idx] = 1
        if not all([sum(row) == 1 for row in cg_map]):
            warnings.warn("WARNING: Slice mapping matrix is not unique.")
        if not all([row.tolist().count(1) == 1 for row in cg_map]):
            warnings.warn("WARNING: Slice mapping matrix is not linear.")

        self.cg_map = cg_map

        # save N_term and C_term as None, to be overwritten if terminal embeddings used
        self.N_term = None
        self.C_term = None

    def add_terminal_embeddings(
            self,
            N_term: Union[str,None]="N",
            C_term: Union[str,None]="C"
    ):
        """
        Adds separate embedding to terminals (do not need to be defined in original embedding_dict).

        Parameters
        ----------
        N_term:
            Atom of N-terminus to which N_term embedding will be assigned.
        C_term:
            Atom of C-terminus to which C_term embedding will be assigned.

        Either of N_term and/or C_term can be None; in this case only one (or no) terminal embedding(s) will be assigned.
        """
        df_cg = self.cg_dataframe
        # proteins with multiple chains will have multiple N- and C-termini
        self.N_term = N_term
        self.C_term = C_term
        if N_term != None:
            if "N_term" not in self.embedding_dict:
                self.embedding_dict["N_term"] = max(self.embedding_dict.values()) + 1
            N_term_atom = df_cg.loc[
                (df_cg["resSeq"] == df_cg["resSeq"].min()) & (df_cg["name"] == N_term)
                ].index
            for idx in N_term_atom:
                self.cg_dataframe.at[idx, "type"] = self.embedding_dict["N_term"]

        if C_term != None:
            if "C_term" not in self.embedding_dict:
                self.embedding_dict["C_term"] = max(self.embedding_dict.values()) + 1
            C_term_atom = df_cg.loc[
                (df_cg["resSeq"] == df_cg["resSeq"].max()) & (df_cg["name"] == C_term)
                ].index
            for idx in C_term_atom:
                self.cg_dataframe.at[idx, "type"] = self.embedding_dict["C_term"]

    def process_coords_forces(
            self,
            coords: np.ndarray,
            forces: np.ndarray,
            mapping: str="slice_aggregate",
            force_stride: int=100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps coordinates and forces to CG resolution

        Parameters
        ----------
        coords: [n_frames, n_atoms, 3]
            Atomistic coordinates
        forces: [n_frames, n_atoms, 3]
            Atomistic forces
        mapping:
            Mapping scheme to be used, must be either 'slice_aggregate' or 'slice_optimize'.
        force_stride:
            Striding to use for force projection results
        """
        if coords.shape != forces.shape:
            warnings.warn("Cannot process coordinates and forces: mismatch between array shapes.")
            return
        else:
            cg_coords, cg_forces = slice_coord_forces(
                coords,
                forces,
                self.cg_map,
                mapping,
                force_stride
                )

            self.cg_coords = cg_coords
            self.cg_forces = cg_forces

            return cg_coords, cg_forces

    def save_cg_output(
            self,
            save_dir: str,
            save_coord_force: bool=True,
            cg_coords: Union[np.ndarray,None]=None,
            cg_forces: Union[np.ndarray,None]=None
    ):
        """
        Saves processed CG data.

        Parameters
        ----------
        save_dir:
            Path of directory to which output will be saved.
        save_coord_force:
            Whether coordinates and forces should also be saved.
        cg_coords:
            CG coordinates; if None, will check whether these are saved as attribute.
        cg_forces:
            CG forces; if None, will check whether these are saved as an object attribute.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not hasattr(self, "cg_atom_indices"):
            print("CG mapping must be applied before outputs can be saved.")
            return

        save_templ = os.path.join(save_dir, f"{self.tag}{self.name}")
        cg_xyz = self.aa_traj.atom_slice(self.cg_atom_indices).xyz
        cg_traj = md.Trajectory(cg_xyz, md.Topology.from_dataframe(self.cg_dataframe))
        cg_traj.save_pdb(f"{save_templ}_cg_structure.pdb")

        embeds = np.array(self.cg_dataframe["type"].to_list())
        np.save(f"{save_templ}_cg_embeds.npy", embeds)

        if save_coord_force:
            if cg_coords == None:
                if not hasattr(self, "cg_coords"):
                    print("No coordinates found; only CG structure, embeddings and loaded forces will be saved.")
                else:
                    np.save(f"{save_templ}_cg_coords.npy", self.cg_coords)
            else:
                np.save(f"{save_templ}_cg_coords.npy", cg_coords)

            if cg_forces == None:
                if not hasattr(self, "cg_forces"):
                    print("No forces found;  only CG structure, embeddings, and loaded coordinates will be saved.")
                else:
                    np.save(f"{save_templ}_cg_forces.npy", self.cg_forces)
            else:
                np.save(f"{save_templ}_cg_forces.npy", cg_forces)

    def get_prior_terms(
            self,
            prior_dict: Dict,
            save_nls: bool=True,
            **kwargs
    ) -> Dict:
        """
        Creates neighbourlists for all prior terms specified in the prior_dict.

        Parameters
        ----------
        prior_dict:
            Dictionary of prior terms and their corresponding parameters.
            Must minimally contain the following information for each key:

            str(prior_name) : {
                "type" : string specifying type as one of 'bonds', 'angles', 'dihedrals', 'non_bonded'
                "prior_function" : name of a function implemented in priors.py which will be used to collect
                            atom groups associated with the prior term.
                ...
                }
        save_nls:
            If true, will save an output of the molecule's neighbourlist.
        kwargs:
            save_dir:
                If save_nls = True, the neighbourlist will be saved to this directory.
            prior_tag:
                String identifying the specific combination of prior terms.

        Returns
        -------
        Dictionary of prior terms with specific index mapping for the given molecule.

        Example
        -------

        prior_dict: {
            bonds:
                type: bonds
                prior_function: standard_bonds
                separate_termini: true
            angles:
                type: angles
                prior_function: standard_angles
                separate_termini: true
            non_bonded:
                type: non_bonded
                prior_function: non_bonded
                min_pair: 6
                res_exclusion: 1
                separate_termini: false
            phi:
                type: dihedrals
                prior_function: phi
            psi:
                type: dihedrals
                prior_function: psi
        }
        """
        # if function has not been written for prior term, will be skipped
        omit_prior = []
        for prior in prior_dict.keys():
            if isinstance(prior_dict[prior]["prior_function"], str):
                try:
                    prior_dict[prior]["prior_function"] = eval(prior_dict[prior]["prior_function"])
                except NameError:
                    print(f"The prior term {prior} has not been defined and will be omitted.")
                    omit_prior.append(prior)

        for prior in omit_prior:
            del prior_dict[prior]

        if any("separate_termini" in prior_dict[prior] for prior in prior_dict.keys()):
            prior_dict = get_terminal_atoms(
                prior_dict,
                cg_dataframe=self.cg_dataframe,
                N_term=self.N_term,
                C_term=self.C_term
                )

        # get atom groups for edges and orders for all prior terms
        cg_top = self.aa_traj.atom_slice(self.cg_atom_indices).topology

        all_edges_and_orders = get_edges_and_orders(
            prior_dict,
            topology=cg_top,
            )

        tags = [x[0] for x in all_edges_and_orders]
        orders = [x[1] for x in all_edges_and_orders]
        edges = [
            torch.tensor(x[2]).type(torch.LongTensor) if isinstance(x[2], np.ndarray)
            else x[2].type(torch.LongTensor) for x in all_edges_and_orders
        ]

        prior_nls = {
            tag: make_neighbor_list(tag, order, edge)
            for tag, order, edge in zip(tags, orders, edges)
        }

        if save_nls:
            ofile = os.path.join(kwargs["save_dir"], f"{self.tag}{self.name}_prior_nls_{kwargs['prior_tag']}.pkl")
            with open(
                ofile,"wb") as pfile:
                pickle.dump(prior_nls, pfile)

        return prior_nls


class RawDataset:
    def __init__(self, dataset_name:str, names: List[str], tag: str, pdb_template_fn:str) -> None:
        self.dataset_name = dataset_name
        self.names = names
        self.tag = tag
        self.pdb_template_fn = pdb_template_fn
        self.dataset = []

        for name in names:
            data_samples = SampleCollection(
                name=name,
                tag=tag,
                pdb_fn=pdb_template_fn.format(name),
            )
            self.dataset.append(data_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


