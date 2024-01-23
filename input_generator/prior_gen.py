import torch
import numpy as np

from typing import Callable, Optional
from collections import defaultdict
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral, Repulsion



class PriorBuilder:
    def __init__(self, n_bins:int,nl_builder_fn:Callable, prior_fit_fn:Callable) -> None:
        self.n_bins = n_bins
        self.histograms = defaultdict(lambda : np.zeros(n_bins, dtype=np.float64))
        self.prior_fit_fn = prior_fit_fn
        self.nl_builder_fn = nl_builder_fn

    def build_nl(self, topology, **kwargs):
        return self.nl_builder_fn(topology)

class Bonds(PriorBuilder):
    def __init__(self, name:str, nl_builder_fn:Callable,
                    separate_termini:bool, n_bins:int,
                    prior_fit_fn:Callable) -> None:
        super().__init__(n_bins,nl_builder_fn,prior_fit_fn)
        self.name = name
        self.type = "bonds"
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        return self.nl_builder_fn(
            topology, separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,c_atoms=self.c_atoms,
        )

class Angles(PriorBuilder):
    def __init__(self, name:str, nl_builder_fn:Callable,
                    separate_termini:bool, n_bins:int,
                    prior_fit_fn:Callable) -> None:
        super().__init__(n_bins,nl_builder_fn,prior_fit_fn)
        self.name = name
        self.type = "angles"
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        return self.nl_builder_fn(
            topology, separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,c_atoms=self.c_atoms,
        )

class NonBonded(PriorBuilder):
    def __init__(self, name:str, nl_builder_fn:Callable,
                    min_pair:int,res_exclusion:int,separate_termini:bool , n_bins:int,
                    prior_fit_fn:Callable) -> None:
        super().__init__(n_bins,nl_builder_fn,prior_fit_fn)
        self.name = name
        self.type = "non_bonded"
        self.min_pair = min_pair
        self.res_exclusion = res_exclusion
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        bond_edges = kwargs['bond_edges']
        angle_edges =  kwargs['angle_edges']
        return self.nl_builder_fn(
            topology, bond_edges=bond_edges,angle_edges=angle_edges,
            separate_termini=self.separate_termini,
            min_pair=self.min_pair, res_exclusion=self.res_exclusion,
            n_term_atoms=self.n_term_atoms,c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,c_atoms=self.c_atoms,
        )

class Dihedrals(PriorBuilder):
    def __init__(self, name:str, nl_builder_fn:Callable,
                    n_bins:int,
                    prior_fit_fn:Callable) -> None:
        super().__init__(n_bins,nl_builder_fn,prior_fit_fn)
        self.name = name
        self.type = "dihedrals"

    def build_nl(self, topology, **kwargs):
        return self.nl_builder_fn(topology)