#!/usr/bin/env python3
"""Drop-in LaTeX for the results section, every number read from `logs/thesis/`.

Writes logs/thesis/figures/dropins.tex: the covariance table and the fusion table with the
geometric baseline row, the navigation table for the five-task campaign, and a list of the
in-text numbers with the artifact each comes from. Nothing here edits the manuscript; the
author pastes what they accept.

Populations: correction numbers come from D_dev (`fits/ddev_evaluation/manifest.json`);
covariance and fusion from the sealed final audit; navigation from `analysis/`.

    python3 figures/make_dropins.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter

import paper as P

T = P.THESIS


def bold_best(values, lower=True):
    best = min(values) if lower else max(values)
    return [v == best for v in values]


def covariance_table(report):
    rows = (("Global", "R0_global_full"), ("Per-camera", "R1_per_camera_full"),
            ("Spatial", "R2_spatial_full"), ("Geometric", "Rproj"))
    stats = [report["covariance"][k] for _, k in rows]
    nll = [s["equal_position_mean_nll"] for s in stats]
    area = [s["equal_position_mean_ellipse_area_95_cm2"] for s in stats]
    nis_gap = [abs(s["equal_position_mean_nis"] - 2.0) for s in stats]
    c95_gap = [abs(s["equal_position_coverage"]["95"] - 0.95) for s in stats]
    bn, ba, bnis, bc = bold_best(nll), bold_best(area), bold_best(nis_gap), bold_best(c95_gap)
    b = lambda text, flag: f"\\textbf{{{text}}}" if flag else text
    lines = []
    for i, ((label, _), s) in enumerate(zip(rows, stats)):
        c = s["equal_position_coverage"]
        if label == "Geometric":
            lines.append("        \\midrule")
        nll_s = b(f"{nll[i]:.3f}", bn[i])
        nis_s = b(f"{s['equal_position_mean_nis']:.3f}", bnis[i])
        c95_s = b(f"{100 * c['95']:.2f}", bc[i])
        area_s = b(f"{area[i]:.1f}", ba[i])
        lines.append(f"        {label:<11} & ${nll_s}$ & {nis_s} & {100 * c['50']:.2f} & {c95_s}"
                     f" & {100 * c['99']:.2f} & {area_s} \\\\")
    n, p = stats[0]["observations"], stats[0]["positions"]
    return n, p, "\n".join(lines)


def fusion_table(fusion):
    order = (("best_spatial_single", "$R_2$-selected single"), ("equal", "Equal weights"),
             ("rproj", "Geometric"), ("global", "Global"), ("per_camera", "Per-camera"), ("spatial", "Spatial"))
    m = fusion["metrics"]
    rmse = [m[k]["rmse_m"] for k, _ in order]
    best = min(rmse)
    lines = []
    for k, label in order:
        r = m[k]
        rm = f"{100 * r['rmse_m']:.2f}"
        rm = f"\\textbf{{{rm}}}" if r["rmse_m"] == best else rm
        nis = f"{r['mean_nis']:.3f}" if "mean_nis" in r else "--"
        cov = f"{100 * r['coverage_95']:.2f}\\%" if "coverage_95" in r else "--"
        area = f"{r['mean_fused_ellipse_area_95_cm2']:.1f}" if "mean_fused_ellipse_area_95_cm2" in r else "--"
        lines.append(f"        {label} & {rm} & {nis} & {cov} & {area} \\\\")
    return fusion["batches"] if "batches" in fusion else m["spatial"]["batches"], "\n".join(lines)


def fused_sq_errors(run):
    """Squared fused error, one per fusion decision (fused vs ground truth at the fused stamp)."""
    out = {}
    with (P.REPO / run / "fusion_observations.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["decision_seq"] in out:
                continue
            try:
                v = [float(r[k]) for k in ("fused_x", "fused_y", "gt_x_at_fused", "gt_y_at_fused")]
            except ValueError:
                continue
            if all(math.isfinite(x) for x in v):
                out[r["decision_seq"]] = (v[0] - v[2]) ** 2 + (v[1] - v[3]) ** 2
    return list(out.values())


def missed_updates(run, period_s=0.2):
    """Share of 0.2 s camera periods without an accepted update: k = round(gap / 0.2) periods per
    gap between accepted updates, k - 1 of them missed."""
    with (P.REPO / run / "correction_assimilations.csv").open(newline="") as handle:
        stamps = sorted(float(r["apply_stamp"]) for r in csv.DictReader(handle) if r["accepted"] in ("1", "True", "true"))
    k = [round(d / period_s) for d in (b - a for a, b in zip(stamps, stamps[1:]))]
    return sum(max(x - 1, 0) for x in k) / sum(k) if sum(k) else math.nan


def navigation_table(rows, summary):
    """Rows of tab:navigation-final: success, fused RMSE pooled over the condition, belief error
    and belief sigma (means over runs), missed updates (mean over runs)."""
    lines = []
    for model in P.MODELS:
        for state in ("intact", "removal"):
            arm = [r for r in rows if r["model"] == model and r["state"] == state and r.get("run_dir")]
            a = summary["arms"][f"{model}_{state}"]
            succ = f"{a['successes']}/{a['cells']}"
            fused = [e for r in arm for e in fused_sq_errors(r["run_dir"])]
            missed = [missed_updates(r["run_dir"]) for r in arm]
            lines.append(
                f"        {P.MODEL_LABEL[model]}, {'intact' if state == 'intact' else 'dropout':<7} & {succ}"
                f" & {100 * math.sqrt(sum(fused) / len(fused)):.2f} & {100 * a['belief_error_m']['mean']:.2f}"
                f" & {100 * a['belief_sigma_major_m']['mean']:.2f} & {100 * np_nanmean(missed):.1f} \\\\")
    return "\n".join(lines)


def np_nanmean(values):
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def route_changes(rows):
    by = {}
    for r in rows:
        by[(r["model"], r["state"], r["task"], r["seed"])] = r["route"]
    out = {}
    for model in P.MODELS:
        pairs = [(t, s) for (m, st, t, s) in by if m == model and st == "intact"]
        changed = [(t, s) for t, s in pairs if by[(model, "removal", t, s)] != by[(model, "intact", t, s)]]
        out[model] = (len(changed), len(pairs), sorted({t for t, _ in changed}))
    return out


def main():
    report = json.loads((T / "final_audit/report.json").read_text())
    fusion = json.loads((T / "final_audit_fusion/report.json").read_text())
    ddev = json.loads((T / "fits/ddev_evaluation/manifest.json").read_text())["D_dev_correction_metrics"]
    summary = json.loads((P.ANALYSIS / "summary.json").read_text())
    with (P.ANALYSIS / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    n_obs, n_pos, cov_rows = covariance_table(report)
    n_batches, fus_rows = fusion_table(fusion)
    nav_rows = navigation_table(rows, summary)
    changes = route_changes(rows)
    raw, struct, full = ddev["raw"], ddev["structured_only"], ddev["structured_plus_visibility"]
    fm = fusion["metrics"]
    diff = summary["removal_minus_intact"]
    vs = summary["spatial_minus_other_under_removal"]
    ci = lambda d, s=100.0, u="": f"{s * d['mean']:+.1f}{u} [{s * d['ci95'][0]:+.1f}, {s * d['ci95'][1]:+.1f}]"
    by_count = {c: 100 * fusion["by_camera_count"]["spatial"][c]["rmse_m"] for c in ("2", "3", "4")}

    text = f"""% ---------------------------------------------------------------------------
% Drop-ins for 05_results.tex, generated by figures/make_dropins.py from logs/thesis/.
% Nothing below has been pasted into the manuscript.
% ---------------------------------------------------------------------------

% === Table: covariance (replace the tabular body of tab:covariance-final) =====
% Sealed final audit, D_eval: {n_obs} admitted observations at {n_pos} positions.
% Every position counts once. Bold: best NLL and area, NIS closest to 2, C95 closest to 95.
%   Model & NLL & NIS & C$_{{50}}$ & C$_{{95}}$ & C$_{{99}}$ & Area$_{{95}}$ (cm$^2$) \\\\
{cov_rows}

% === Table: fusion (replace the tabular body of tab:localization-final) =========
% {n_batches} position--heading batches with at least two admitted cameras.
% Suggested header: Fusion rule & RMSE (cm) & NIS & Cov.~95 & Area$_{{95}}$ (cm$^2$) \\\\
{fus_rows}

% === Table: navigation (replace the tabular body of tab:navigation-final) =======
% 5 tasks x 3 matched seeds per arm; 90 of 90 runs evidence-valid. Success: stopped at the
% goal on the belief, true final distance < 0.30 m, never left the driveable region.
% Fused error: RMSE over fusion decisions, pooled over the condition. Belief error and sigma:
% means over runs. Missed updates: share of 0.2 s camera periods without an accepted update.
{nav_rows}

% === In-text numbers =============================================================
% Correction (D_dev, {raw['observations']} observations at {raw['positions']} positions; RMSE = mean over positions of per-position RMSE):
%   raw {100 * raw['equal_position_rmse_m']:.2f} cm -> structured {100 * struct['equal_position_rmse_m']:.2f} cm -> full {100 * full['equal_position_rmse_m']:.2f} cm
%   visibility branch removes a further {100 * (1 - full['equal_position_rmse_m'] / struct['equal_position_rmse_m']):.0f}% of the structured RMSE
%   median {100 * raw['pooled_median_m']:.2f} -> {100 * full['pooled_median_m']:.2f} cm; p95 {100 * raw['pooled_p95_m']:.2f} -> {100 * full['pooled_p95_m']:.2f} cm
%   raw along/across-ray bias {100 * raw['equal_position_signed_bias_ray_m'][0]:+.2f} / {100 * raw['equal_position_signed_bias_ray_m'][1]:+.2f} cm;
%   after correction {100 * full['equal_position_signed_bias_ray_m'][0]:+.2f} / {100 * full['equal_position_signed_bias_ray_m'][1]:+.2f} cm
% Covariance: spatial vs per-camera 95% area {100 * (1 - report['covariance']['R2_spatial_full']['equal_position_mean_ellipse_area_95_cm2'] / report['covariance']['R1_per_camera_full']['equal_position_mean_ellipse_area_95_cm2']):.1f}% smaller;
%   geometric baseline sigma_px = {report['rproj_sigma_px']:.3f} px (fitted on D_R)
%   cross-camera residual correlation: NOT regenerated (no current artifact); drop or recompute
% Fusion: spatial RMSE {100 * fm['spatial']['rmse_m']:.2f} cm, {100 * fm['spatial']['rmse_reduction_vs_equal']:.1f}% below equal weights and
%   {100 * fm['spatial']['rmse_reduction_vs_best_spatial_single']:.1f}% below the R2-selected single camera; by camera count 2/3/4:
%   {by_count['2']:.2f} / {by_count['3']:.2f} / {by_count['4']:.2f} cm
% Navigation, route changes under dropout (task-seed pairs): global {changes['global'][0]}/{changes['global'][1]},
%   per-camera {changes['per_camera'][0]}/{changes['per_camera'][1]}, spatial {changes['spatial'][0]}/{changes['spatial'][1]} (tasks: {', '.join(P.TASK_LABEL[t] for t in changes['spatial'][2])})
% Dropout minus intact, matched on (task, seed), mean [95% bootstrap CI]:
%   success (pp):        global {ci(diff['global']['success'])}, per-camera {ci(diff['per_camera']['success'])}, spatial {ci(diff['spatial']['success'])}
%   belief error (cm):   global {ci(diff['global']['belief_error_m'])}, per-camera {ci(diff['per_camera']['belief_error_m'])}, spatial {ci(diff['spatial']['belief_error_m'])}
%   belief sigma (cm):   global {ci(diff['global']['belief_sigma_major_m'])}, per-camera {ci(diff['per_camera']['belief_sigma_major_m'])}, spatial {ci(diff['spatial']['belief_sigma_major_m'])}
% Spatial minus other model, both with one camera dropped:
%   success (pp):        vs global {ci(vs['global']['success'])}, vs per-camera {ci(vs['per_camera']['success'])}
%   belief error (cm):   vs global {ci(vs['global']['belief_error_m'])}, vs per-camera {ci(vs['per_camera']['belief_error_m'])}
"""
    out = P.OUT / "dropins.tex"
    P.OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out)


if __name__ == "__main__":
    main()
