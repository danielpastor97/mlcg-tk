import torch
from typing import Dict, Callable, Optional
import numpy as np
from scipy.integrate import trapezoid
from input_generator.prior_gen import PriorBuilder
from input_generator.embedding_maps import CGEmbeddingMap
from copy import deepcopy


def fit_potentials(
    nl_name: str,
    prior_builder: PriorBuilder,
    embedding_map: Optional[CGEmbeddingMap],
    temperature: float = 300.0,
):
    histograms = prior_builder.histograms[nl_name]
    bin_centers = prior_builder.histograms.bin_centers
    prior_fit_fn = prior_builder.prior_fit_fn

    target_fit_kwargs = prior_builder.nl_builder.get_fit_kwargs(nl_name)

    kB = 0.0019872041
    beta = 1 / (temperature * kB)

    statistics = {}
    for kf in list(histograms.keys()):
        hist = torch.tensor(histograms[kf])

        mask = hist > 0
        bin_centers_nz = bin_centers[mask]
        ncounts_nz = hist[mask]
        dG_nz = -torch.log(ncounts_nz) / beta

        params = prior_fit_fn(
            bin_centers_nz=bin_centers_nz,
            dG_nz=dG_nz,
            ncounts_nz=ncounts_nz,
            **target_fit_kwargs
        )

        statistics[kf] = params

        statistics[kf]["p"] = hist / trapezoid(
            hist.cpu().numpy(), x=bin_centers.cpu().numpy()
        )
        statistics[kf]["p_bin"] = bin_centers
        statistics[kf]["V"] = dG_nz
        statistics[kf]["V_bin"] = bin_centers_nz

    if getattr(prior_builder.nl_builder, "replace_gly_ca_stats", False):
        statistics = replace_gly_stats(
            statistics, gly_bead=embedding_map["GLY"], ca_bead=embedding_map["CA"]
        )

    prior_model = prior_builder.get_prior_model(
        statistics, nl_name, targets="forces", **target_fit_kwargs
    )

    return prior_model


def replace_gly_stats(statistics, gly_bead, ca_bead):
    gly_atom_groups = [group for group in list(statistics.keys()) if gly_bead in group]
    for group in gly_atom_groups:
        gly_idx = group.index(gly_bead)
        ca_group = list(deepcopy(group))
        ca_group[gly_idx] = ca_bead
        try:
            statistics[group] = statistics[tuple(ca_group)]
        except KeyError:
            ca_group = (ca_group[0], ca_group[2], ca_group[1], ca_group[3])
    return statistics
