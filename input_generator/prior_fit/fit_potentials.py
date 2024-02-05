import torch
from typing import Dict, Callable, Optional
import numpy as np
from scipy.integrate import trapezoid
from .histogram import HistogramsNL
from mlcg.nn.gradients import GradientsOut


def fit_potentials(
    nl_name: str,
    prior_builder: HistogramsNL,
    temperature: float = 300.,
):
    histograms = prior_builder.histograms[nl_name]
    bin_centers=prior_builder.histograms.bin_centers
    prior_fit_fn=prior_builder.prior_fit_fn
    
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

        # need to make fit_from_values possible, these functions are new
        params = prior_fit_fn(bin_centers_nz, dG_nz, **target_fit_kwargs)

        statistics[kf] = params

        statistics[kf]["p"] = hist / trapezoid(
            hist.cpu().numpy(), x=bin_centers.cpu().numpy()
        )
        statistics[kf]["p_bin"] = bin_centers
        statistics[kf]["V"] = dG_nz
        statistics[kf]["V_bin"] = bin_centers_nz
    
    prior_model =  prior_builder.get_prior_model(statistics, targets="forces", **target_fit_kwargs)

    return prior_model
