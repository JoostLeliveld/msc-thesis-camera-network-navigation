#!/usr/bin/env python3
"""Every campaign run at a glance.

One square per run: rows are tasks (labelled by the camera the task drops), column groups
are models, and within a group the three seeds with all cameras (intact, left) and with the
task camera dropped out (right). Filled in the model colour: success. Failures are empty
cells with a black glyph: x the footprint left the driveable region, o the run failed
without leaving it. A small black dot above a dropout cell: the planned route differs from
the matched intact run. The matched differences with intervals are quoted in the text.

    python3 figures/make_campaign_outcomes.py
"""
from __future__ import annotations

import csv

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import paper as P

SEEDS = ("91500", "91501", "91502")
STATE_LABEL = {"intact": "intact", "removal": "dropout"}


def main():
    with (P.ANALYSIS / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    route = {(r["task"], r["model"], r["state"], r["seed"]): r["route"] for r in rows}
    tasks = [t["name"] for t in P.tasks()]
    fig = P.plt.figure(figsize=(0.82 * P.TEXT, 2.25))
    ax = fig.add_axes((0.0, 0.10, 1.0, 0.88))
    s, gap_seed, gap_state, gap_model, gap_row = 1.0, 0.18, 0.6, 1.3, 0.5
    x0, x = {}, 0.0
    for model in P.MODELS:
        for state in ("intact", "removal"):
            x0[(model, state)] = x
            x += 3 * s + 2 * gap_seed + (gap_state if state == "intact" else gap_model)
    width = x - gap_model
    for i, task in enumerate(tasks):
        y = (len(tasks) - 1 - i) * (s + gap_row)
        for r in rows:
            if r["task"] != task:
                continue
            xi = x0[(r["model"], r["state"])] + SEEDS.index(r["seed"]) * (s + gap_seed)
            colour = P.MODEL_COLOUR[r["model"]]
            if r["success"] == "1":
                ax.add_patch(Rectangle((xi, y), s, s, fc=colour, ec=colour, lw=0.8))
            else:
                ax.add_patch(Rectangle((xi, y), s, s, fc="white", ec=colour, lw=0.9))
                glyph = "x" if r["collision"] == "1" else "o"
                ax.plot(xi + s / 2, y + s / 2, glyph, color=P.INK, mfc="none", ms=5.5, mew=1.2)
            if r["state"] == "removal" and r["route"] != route[(task, r["model"], "intact", r["seed"])]:
                ax.plot(xi + s / 2, y + s + 0.17, ".", color=P.INK, ms=4)
        ax.text(-0.4, y + s / 2, P.TASK_LABEL[task], ha="right", va="center", fontsize=7.5)
    top = len(tasks) * (s + gap_row) - gap_row
    for model in P.MODELS:
        xa, xb = x0[(model, "intact")], x0[(model, "removal")] + 3 * s + 2 * gap_seed
        ax.text((xa + xb) / 2, top + 1.25, P.MODEL_LABEL[model], ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=P.MODEL_COLOUR[model])
        for state in ("intact", "removal"):
            n = sum(r["success"] == "1" for r in rows if r["model"] == model and r["state"] == state)
            ax.text(x0[(model, state)] + 1.5 * s + gap_seed, top + 0.35, f"{STATE_LABEL[state]} {n}/15",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-6.2, width + 0.2); ax.set_ylim(-0.3, top + 2.1)
    ax.set_aspect("equal"); ax.axis("off")
    handles = [Line2D([], [], marker="s", ls="none", ms=6, mfc=P.MODEL_COLOUR["spatial"], mec="none"),
               Line2D([], [], marker="x", ls="none", ms=5.5, color=P.INK, mew=1.2),
               Line2D([], [], marker="o", ls="none", ms=5.5, mfc="none", mec=P.INK, mew=1.2),
               Line2D([], [], marker=".", ls="none", ms=6, color=P.INK)]
    ax.legend(handles, ["success", "failed, left the driveable region", "failed, stayed inside",
                        "route changed"],
              loc="upper center", bbox_to_anchor=(0.55, 0.02), ncol=4, fontsize=7, handlelength=0.9,
              columnspacing=1.0, handletextpad=0.35, frameon=False)
    P.save(fig, "campaign_outcomes")


if __name__ == "__main__":
    main()
