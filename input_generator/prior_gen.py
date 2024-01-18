import torch
import numpy as np

from typing import Callable
from collections import defaultdict
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral, Repulsion



class PriorBuilder:
    def __init__(self, n_bins:int, prior_fit_fn:Callable) -> None:
        self.n_bins = n_bins
        self.histograms = defaultdict(lambda : np.zeros(n_bins, dtype=np.float64))
        self.prior_fit_fn = prior_fit_fn



class Bonds(PriorBuilder):
    def __init__(self, name:str, nl_builder_fn:Callable, separate_termini:bool, n_bins:int, prior_fit_fn:Callable) -> None:
        super().__init__(n_bins)
        self.name = name
        self.type = "bonds"
        self.nl_builder_fn = nl_builder_fn
        self.separate_termini = separate_termini


    def

class Angles(PriorBuilder):
    def __init__(self, n_bins) -> None:
        super().__init__(n_bins)