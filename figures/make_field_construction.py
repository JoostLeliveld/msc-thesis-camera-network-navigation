#!/usr/bin/env python3
"""How the spatial covariance R2 is built from the collected residuals.

Top row: for one camera, the D_R positions it observed, two query points and
their k nearest positions shaded by the Gaussian weight w_{i,g}(p); beside each
query, the neighbours' corrected ray-frame residuals and the resulting 2-sigma
ellipse of R2 at that query.  Bottom row: the worst-axis standard deviation of
R2 for every camera over the whole map, with cells at the prior (no local
support) shown as their own stratum.

R2 is evaluated with the pipeline's own function (planning_precision.py), and
the script first checks that it reproduces the stored planning field.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "figures"), str(ROOT / "pipeline"),
                str(ROOT / "src/reliability"), str(ROOT / "src/unav_common"), str(ROOT)]
from style import CAM_COLOUR, draw_warehouse, layout  # noqa: E402
from planning_precision import ray_basis, spatial_ray_covariance  # noqa: E402

PAPER_FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
FITS = ROOT / "logs/thesis/fits"
INK = "#222831"
NO_SUPPORT_CM = 999.0  # the (10 m)^2 prior: absence of evidence, not accuracy
BANDS_CM = (0, 1, 2, 5, 10, 25, 50, NO_SUPPORT_CM)


def load():
    with np.load(FITS / "covariance/models.npz") as archive:
        model = {key: archive[key] for key in archive.files}
    with np.load(FITS / "corrected_residuals/corrected_residuals.npz") as archive:
        residuals = {key: archive[key] for key in archive.files}
    with np.load(FITS / "planning_precision/m2_planning_precision.npz") as archive:
        field = {key: archive[key] for key in archive.files}
    mount = {f"camera_{c.name}": np.array([c.x, c.y]) for c in layout().cameras}
    return model, residuals, field, mount


def world_covariance(model, mount, camera, point):
    ray, support = spatial_ray_covariance(model, camera, point)
    basis = ray_basis(mount[camera], point)
    return basis @ ray @ basis.T, ray, support


def check_against_stored_field(model, field, mount, samples=400):
    """The figure is only valid if it reproduces what the planner used."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(samples):
        c = rng.integers(len(field["camera_ids"]))
        yi, xi = rng.integers(len(field["ys"])), rng.integers(len(field["xs"]))
        camera = str(field["camera_ids"][c])
        point = np.array([field["xs"][xi], field["ys"][yi]])
        world, _, _ = world_covariance(model, mount, camera, point)
        stored = np.linalg.inv(field["matched_precision_m2_inv"][c, yi, xi])
        worst = max(worst, float(np.max(np.abs(world - stored)) / np.max(np.abs(stored))))
    if worst > 1e-6:
        raise RuntimeError(f"R2 does not reproduce the stored field (rel. error {worst:.2e})")
    print(f"check: R2 reproduces the stored planning field (max rel. error {worst:.1e})")


def neighbours(model, camera, point):
    local = model["spatial_camera"].astype(str) == camera
    points = model["spatial_reference_xy_m"][local]
    distance2 = np.sum((points - point) ** 2, axis=1)
    count = min(int(model["k_neighbors"][0]), len(points))
    selected = np.argpartition(distance2, count - 1)[:count]
    weight = np.exp(-0.5 * distance2[selected] / float(model["length_scale_m"][0]) ** 2)
    return points, points[selected], weight


def ellipse(ax, covariance, colour, **kwargs):
    values, vectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    ax.add_patch(Ellipse((0, 0), 4 * np.sqrt(values[1]), 4 * np.sqrt(values[0]),
                         angle=angle, fc="none", ec=colour, **kwargs))


def draw_query_map(ax, model, camera, queries):
    draw_warehouse(ax, layout(), show_cameras=True, camera_labels=True, rack_alpha=0.82)
    points, _, _ = neighbours(model, camera, queries[0][1])
    ax.scatter(points[:, 0], points[:, 1], s=2.5, color="#9b9a94", lw=0, zorder=10,
               label=f"positions observed by camera {camera[-1]}")
    for label, point in queries:
        _, chosen, weight = neighbours(model, camera, point)
        ax.scatter(chosen[:, 0], chosen[:, 1], s=10, c=weight, cmap="Greys", vmin=0,
                   vmax=1, edgecolors=INK, linewidths=0.35, zorder=12)
        ax.plot(*point, marker="X", ms=6, color=CAM_COLOUR[camera[-1]], mec="white",
                mew=0.6, zorder=14)
        ax.annotate(label, point, xytext=(5, 4), textcoords="offset points",
                    fontsize=7, weight="bold", color=INK, zorder=15)
    ax.set(xlim=(-11.8, 11.8), ylim=(-9.75, 9.75), aspect="equal",
           xticks=[-10, -5, 0, 5, 10], yticks=[-5, 0, 5], xlabel="x [m]", ylabel="y [m]")
    ax.set_title(f"(a) Camera {camera[-1]}: neighbours of two queries", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=5.6, markerscale=2.5)


def draw_query_residuals(ax, model, residuals, mount, camera, label, point, title,
                         limit):
    """Residuals of the k neighbouring positions, in the query's ray frame."""
    _, chosen, weight = neighbours(model, camera, point)
    use = (residuals["role"] == "D_R") & (residuals["camera"] == camera)
    truth, world_residual = residuals["truth_xy_m"][use], residuals["residual_world_m"][use]
    basis = ray_basis(mount[camera], point)
    for position, w in zip(chosen, weight):
        rows = np.linalg.norm(truth - position, axis=1) < 0.05
        ray = 100 * world_residual[rows] @ basis
        ax.scatter(ray[:, 0], ray[:, 1], s=5, color=plt.cm.Greys(0.25 + 0.75 * w),
                   lw=0, zorder=4)
    _, ray_cov, support = world_covariance(model, mount, camera, point)
    ellipse(ax, 1e4 * ray_cov, CAM_COLOUR[camera[-1]], lw=1.3, zorder=6)
    sd = 100 * np.sqrt(np.linalg.eigvalsh(ray_cov)[::-1])
    ax.axhline(0, color="#d0d0d0", lw=0.5, zorder=1)
    ax.axvline(0, color="#d0d0d0", lw=0.5, zorder=1)
    ax.set(aspect="equal", xlabel="along-ray error [cm]", ylabel="across-ray error [cm]",
           xlim=(-limit, limit), ylim=(-limit, limit))
    # Statistics in the title, so no text covers the residuals.
    ax.set_title(f"{title}\n$W_i(p)={support:.2f}$, $\\sigma$ = {sd[0]:.1f}, {sd[1]:.1f} cm",
                 loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def worst_axis_sd_cm(field):
    covariance = np.linalg.inv(field["matched_precision_m2_inv"])
    return 100 * np.sqrt(np.linalg.eigvalsh(covariance)[..., -1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="camera_A")
    parser.add_argument("--dense", type=float, nargs=2, default=(6.0, -6.4))
    parser.add_argument("--limit", type=float, default=12.0,
                        help="shared half-width of panels (b, c) in cm")
    parser.add_argument("--sparse", type=float, nargs=2, default=(-10.2, 4.4))
    arguments = parser.parse_args()

    model, residuals, field, mount = load()
    check_against_stored_field(model, field, mount)
    queries = [("1", np.array(arguments.dense)), ("2", np.array(arguments.sparse))]
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.2,
                         "axes.titleweight": "bold", "axes.labelsize": 6.3,
                         "xtick.labelsize": 5.6, "ytick.labelsize": 5.6})

    fig = plt.figure(figsize=(7.16, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.35, 1.0))
    top = grid[0].subgridspec(1, 3, width_ratios=(1.45, 1.0, 1.0))
    draw_query_map(fig.add_subplot(top[0]), model, arguments.camera, queries)
    for column, ((label, point), title) in enumerate(
            zip(queries, ("(b) Query 1, dense support", "(c) Query 2, sparse support")), start=1):
        draw_query_residuals(fig.add_subplot(top[column]), model, residuals, mount,
                             arguments.camera, label, point, title, arguments.limit)

    sd = worst_axis_sd_cm(field)
    colours = ["#1f5c9e", "#3f7fc0", "#6fa3d6", "#a9c8e8", "#e6c79a", "#d98b4f",
               "#9b9a94"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(BANDS_CM, cmap.N)
    bottom = grid[1].subgridspec(1, len(field["camera_ids"]))
    mesh = None
    for index, camera in enumerate(field["camera_ids"].astype(str)):
        ax = fig.add_subplot(bottom[index])
        values = np.minimum(sd[index], NO_SUPPORT_CM - 1e-3)
        mesh = ax.pcolormesh(field["xs"], field["ys"], values, cmap=cmap, norm=norm,
                             shading="nearest", rasterized=True, zorder=0)
        draw_warehouse(ax, layout(), show_cameras=True, camera_labels=False, rack_alpha=0.55)
        ax.set(xlim=(-11.8, 11.8), ylim=(-9.75, 9.75), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"camera {camera[-1]}", fontsize=6.8,
                     color=CAM_COLOUR[camera[-1]])
    fig.text(0.0, 0.405, "(d) Worst-axis standard deviation of $R_2$ per camera",
             fontsize=7.2, weight="bold")
    bar = fig.colorbar(mesh, ax=fig.axes[-5:], location="bottom", shrink=0.6,
                       aspect=40, pad=0.02, ticks=BANDS_CM[:-1])
    bar.set_label("worst-axis $\\sigma$ [cm]; grey: no local support (prior)")
    bar.ax.tick_params(labelsize=5.6)

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = PAPER_FIGURES / f"field_construction.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
