import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

import pickle as pkl

from input_generator.prior_gen import PriorBuilder

from jsonargparse import CLI
from typing import List

def merge_statistics(
    save_dir: str,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    stats_fns: List[str]
):
    all_stats = []
    for fn in stats_fns:
        with open(fn, "rb") as ifile:
            stats = pkl.load(ifile)
        all_stats.append(stats)

    builder_dict = {}
    for prior_builder in prior_builders:
        builder_dict[prior_builder.name] = prior_builder

    for statistics in all_stats:
        for builder in statistics:
            combined_builder = builder_dict[builder.name]
            for nl_name in list(builder.histograms.data.keys()):
                if nl_name not in builder.nl_builder.nl_names:
                    continue
                hists = builder.histograms[nl_name]
                for k, hist in hists.items():
                    combined_builder.histograms.data[nl_name][k] += hist

    with open(f"{save_dir}{prior_tag}_prior_builders.pck", "wb") as ofile:
        pkl.dump(prior_builders, ofile)


if __name__ == "__main__":
    CLI([merge_statistics], as_positional=False)
