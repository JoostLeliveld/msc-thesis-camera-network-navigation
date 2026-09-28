#!/usr/bin/env python3
"""Run one predeclared two-phase detector-resolution trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    trial = next(
        (item for item in protocol["resolution_trials"] if int(item["imgsz"]) == args.imgsz),
        None,
    )
    if trial is None:
        raise RuntimeError(f"Resolution {args.imgsz} is not in the frozen protocol")
    output = args.out.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing trial: {output}")
    output.mkdir(parents=True)

    base = (REPO / protocol["base_model"]["path"]).resolve()
    data = (REPO / protocol["data"]["yaml"]).resolve()
    if sha256(base) != protocol["base_model"]["sha256"]:
        raise RuntimeError("COCO checkpoint hash drift")
    if sha256(data) != protocol["data"]["yaml_sha256"]:
        raise RuntimeError("Stage-04 dataset YAML hash drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen protocol requires GPU training, but CUDA is unavailable")

    common = protocol["common"]
    manifest = {
        "schema": "thesis_detector_training_trial.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "imgsz": args.imgsz,
        "batch": int(trial["batch"]),
        "base_model": str(base),
        "base_model_sha256": sha256(base),
        "dataset_yaml": str(data),
        "dataset_yaml_sha256": sha256(data),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "phases": [],
    }
    manifest_path = output / "trial_manifest.json"
    write_manifest(manifest_path, manifest)

    shared = dict(
        data=str(data), imgsz=args.imgsz, batch=int(trial["batch"]),
        device=str(common["device"]), workers=int(common["workers"]),
        seed=int(common["seed"]), deterministic=bool(common["deterministic"]),
        optimizer=common["optimizer"], amp=bool(common["amp"]),
        single_cls=bool(common["single_cls"]), cos_lr=bool(common["cos_lr"]),
        weight_decay=float(common["weight_decay"]),
        warmup_epochs=float(common["warmup_epochs"]),
        hsv_h=float(common["hsv_h"]), hsv_s=float(common["hsv_s"]),
        hsv_v=float(common["hsv_v"]), degrees=float(common["degrees"]),
        translate=float(common["translate"]), scale=float(common["scale"]),
        shear=float(common["shear"]), perspective=float(common["perspective"]),
        flipud=float(common["flipud"]), fliplr=float(common["fliplr"]),
        mosaic=float(common["mosaic"]), mixup=float(common["mixup"]),
        copy_paste=float(common["copy_paste"]), cache=False, plots=True,
        save=True, save_period=5, verbose=True,
    )

    weights = base
    for phase in protocol["phases"]:
        phase_name = phase["name"]
        model = YOLO(str(weights))
        result = model.train(
            **shared,
            project=str(output), name=phase_name, exist_ok=False,
            epochs=int(phase["epochs"]), freeze=int(phase["freeze"]),
            lr0=float(phase["learning_rate"]), lrf=float(phase["final_learning_rate_fraction"]),
            close_mosaic=int(phase["close_mosaic"]), patience=int(phase["early_stopping_patience"]),
        )
        phase_dir = Path(result.save_dir).resolve()
        best = phase_dir / "weights" / "best.pt"
        last = phase_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(f"Training phase {phase_name} did not produce checkpoints")
        record = {
            "name": phase_name,
            "directory": str(phase_dir),
            "initial_weights": str(weights),
            "best_weights": str(best),
            "best_weights_sha256": sha256(best),
            "last_weights": str(last),
            "last_weights_sha256": sha256(last),
            "results_csv": str(phase_dir / "results.csv"),
            "results_csv_sha256": sha256(phase_dir / "results.csv"),
        }
        manifest["phases"].append(record)
        write_manifest(manifest_path, manifest)
        weights = best
        del model
        torch.cuda.empty_cache()

    manifest["status"] = "complete_pending_cross_resolution_evaluation"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["candidate_weights"] = manifest["phases"][-1]["best_weights"]
    manifest["candidate_weights_sha256"] = manifest["phases"][-1]["best_weights_sha256"]
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
