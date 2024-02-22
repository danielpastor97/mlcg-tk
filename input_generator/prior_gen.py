from typing import Callable, Optional
from functools import partial

from mlcg.nn.prior import (
    HarmonicBonds,
    HarmonicAngles,
    Dihedral,
    Repulsion,
    _Prior,
    GeneralBonds,
    GeneralAngles,
)
from mlcg.nn.gradients import GradientsOut

from mlcg.data import AtomicData
from .prior_fit.histogram import HistogramsNL


class PriorBuilder:
    def __init__(
        self,
        histograms: HistogramsNL,
        nl_builder: Callable,
        prior_fit_fn: Callable,
        prior_cls: _Prior,
    ) -> None:
        self.histograms = histograms
        self.prior_fit_fn = prior_fit_fn
        self.nl_builder = nl_builder
        self.prior_cls = prior_cls

    def build_nl(self, topology, **kwargs):
        return self.nl_builder(topology=topology)

    def accumulate_statistics(self, nl_name: str, data: AtomicData):
        atom_types = data.atom_types
        mapping = data.neighbor_list[nl_name]["index_mapping"]
        values = self.prior_cls.compute_features(data.pos, mapping)
        self.histograms.accumulate_statistics(nl_name, values, atom_types, mapping)


class Bonds(PriorBuilder):
    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        separate_termini: bool,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
    ) -> None:
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=GeneralBonds,
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
            topology=topology,
            separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,
            c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,
            c_atoms=self.c_atoms,
        )

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        return GradientsOut(self.prior_cls(statistics, name=name), targets="forces")


class Angles(PriorBuilder):
    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        separate_termini: bool,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
    ) -> None:
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=GeneralAngles,
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
            topology=topology,
            separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,
            c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,
            c_atoms=self.c_atoms,
        )

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        return GradientsOut(self.prior_cls(statistics, name=name), targets="forces")


class NonBonded(PriorBuilder):
    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        min_pair: int,
        res_exclusion: int,
        separate_termini: bool,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
        percentile: float = 1,
        cutoff: Optional[float] = None,
    ) -> None:
        prior_fit_fn = partial(prior_fit_fn, percentile=percentile, cutoff=cutoff)
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=Repulsion,
        )
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
        bond_edges = kwargs["bond_edges"]
        angle_edges = kwargs["angle_edges"]
        return self.nl_builder(
            topology=topology,
            bond_edges=bond_edges,
            angle_edges=angle_edges,
            separate_termini=self.separate_termini,
            min_pair=self.min_pair,
            res_exclusion=self.res_exclusion,
            n_term_atoms=self.n_term_atoms,
            c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,
            c_atoms=self.c_atoms,
        )

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        prior = self.prior_cls(statistics)
        prior.name = name
        return GradientsOut(prior, targets="forces")


class Dihedrals(PriorBuilder):
    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
    ) -> None:
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=Dihedral,
        )
        self.name = name
        self.type = "dihedrals"

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        prior = self.prior_cls(statistics, n_degs=kwargs["n_degs"])
        prior.name = name
        return GradientsOut(prior, targets="forces")
