#!/usr/bin/env python3
"""The reference dataset v11: every captured robot position by the role it plays.

D_mu trains the correction, D_R fits the covariance models, D_dev compares and checks them,
and the final audit is sealed until the models are frozen. Read through pipeline/dataset.py,
the only loader.

    python3 figures/make_data_roles.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import paper as P
from pipeline import dataset

ROLES = (("D_mu", "correction training ($D_\\mu$)", "#0072B2", "o"),
         ("D_R", "covariance fitting ($D_R$)", "#B34D8C", "o"),
         ("D_dev", "validation ($D_\\mathrm{val}$)", "#E69F00", "o"),
         ("final_audit", "test ($D_\\mathrm{test}$)", P.INK, "o"))


def main():
    positions = {}
    for r in dataset.load_rows():
        positions.setdefault(r["position_key"], (float(r["robot_x"]), float(r["robot_y"]), r["stratum"]))
    counts = Counter(role for _, _, role in positions.values())
    fig, ax = P.plt.subplots(figsize=(P.COLUMN, 2.75))
    P.draw_map(ax)
    for role, label, colour, marker in ROLES:
        xy = np.array([(x, y) for x, y, r in positions.values() if r == role])
        ax.scatter(xy[:, 0], xy[:, 1], s=1.6, marker=marker, c=colour,
                   edgecolors="none", linewidths=0, alpha=0.9, zorder=6,
                   label=f"{label}: {counts[role]}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.3, markerscale=3.0,
              handletextpad=0.2, columnspacing=1.0)
    P.save(fig, "data_collection_roles")
    print(dict(counts))


if __name__ == "__main__":
    main()
