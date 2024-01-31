import torch
import numpy as np

from typing import Callable, Optional
from collections import defaultdict
from copy import deepcopy

from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral, Repulsion, _Prior, GeneralBonds, GeneralAngles
from mlcg.data import AtomicData
from .prior_fit.histogram import compute_hist, _get_bin_centers


class PriorBuilder:
    def __init__(self,
            n_bins:int,
            bmin: int,
            bmax: int,
            nl_builder:Callable,
            prior_fit_fn:Callable,
            prior_cls: _Prior,
    ) -> None:

        self.n_bins = n_bins
        self.bmin = bmin
        self.bmax = bmax
        self.bin_centers = _get_bin_centers(n_bins, bmin, bmax)
        self.histograms = defaultdict(lambda :defaultdict(lambda : np.zeros(n_bins, dtype=np.float64)))

        self.prior_fit_fn = prior_fit_fn
        self.nl_builder = nl_builder
        self.prior_cls = prior_cls
        # self.neighbor_lists = {}

    def build_nl(self, topology, **kwargs):
        return self.nl_builder(topology)

    def set_neighbor_list(self, tag, nl):
        self.neighbor_lists[tag] = deepcopy(nl)

    def accumulate_histogram(self, data: AtomicData):
        atom_types = data.atom_types
        for nl_tag in self.neighbor_lists.keys():
            mapping = data.neighbor_list[nl_tag]["index_mapping"]
            values = self.prior_cls.compute_features(data.pos, mapping)
            hists = compute_hist(values,atom_types,mapping,self.n_bins,self.bmin,self.bmax)
            for k, hist in hists.items():
                self.histograms[nl_tag][k] += hist.cpu().numpy()

    # def plot_histograms(self, fn:Optional[str]):
    #     for k, hist in self.histograms.items():




class Bonds(PriorBuilder):
    def __init__(
            self,
            name:str,
            nl_builder:Callable,
            separate_termini:bool,
            n_bins:int,
            bmin:int,
            bmax:int,
            prior_fit_fn:Callable
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            bmin=bmin,
            bmax=bmax,
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=HarmonicBonds,
        )
        self.name = name
        self.type = "bonds"
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        return self.nl_builder(
            topology, separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,c_atoms=self.c_atoms,
        )

class Angles(PriorBuilder):
    def __init__(self,
                 name:str,
                 nl_builder:Callable,
                separate_termini:bool,
                n_bins:int,
                bmin:int,
                bmax:int,
                prior_fit_fn:Callable
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            bmin=bmin,
            bmax=bmax,
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=HarmonicAngles,
        )
        self.name = name
        self.type = "angles"
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        return self.nl_builder(
            topology, separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,c_atoms=self.c_atoms,
        )

class NonBonded(PriorBuilder):
    def __init__(self,
                 name:str,
                 nl_builder:Callable,
                min_pair:int,
                res_exclusion:int,
                separate_termini:bool,
                n_bins:int,bmin:float,bmax:float,
                prior_fit_fn:Callable
    ) -> None:
        super().__init__(n_bins,nl_builder,prior_fit_fn,prior_cls=Repulsion)
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
        return self.nl_builder(
            topology, bond_edges=bond_edges,angle_edges=angle_edges,
            separate_termini=self.separate_termini,
            min_pair=self.min_pair, res_exclusion=self.res_exclusion,
            n_term_atoms=self.n_term_atoms,c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,c_atoms=self.c_atoms,
        )

class Dihedrals(PriorBuilder):
    def __init__(self, name:str, nl_builder:Callable,
                    n_bins:int,bmin:float,bmax:float,
                    prior_fit_fn:Callable,) -> None:
        super().__init__(n_bins,nl_builder,prior_fit_fn,prior_cls=Dihedral)
        self.name = name
        self.type = "dihedrals"

