"""Read-only YOLO CSV evidence; never unpickle checkpoints or launch training."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

METRICS = {
    "precision": ("metrics/precision(B)", "metrics/precision"),
    "recall": ("metrics/recall(B)", "metrics/recall"),
    "map50": ("metrics/mAP50(B)", "metrics/mAP_0.5"),
    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95"),
}
CONTEXT_KEYS = ("dataset_sha256", "split_sha256", "code_revision", "seed", "imgsz",
                "epochs", "batch", "optimizer", "task", "class_names", "device")


def number(value, name: str) -> float:
    try:
        v = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not math.isfinite(v):
        raise ValueError(f"non-finite {name}")
    return v


def summarize_csv(content: str, metadata: dict | None = None, name: str = "uploaded") -> dict:
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise ValueError("CSV has no header")
    names = {c.strip() for c in reader.fieldnames}
    selected = {k: next((c for c in options if c in names), None) for k, options in METRICS.items()}
    if "epoch" not in names or not selected["map50_95"]:
        raise ValueError("CSV requires epoch and validation mAP50-95 columns")
    valid, rejected = [], []
    for line, raw in enumerate(reader, 2):
        row = {(k or "").strip(): v for k, v in raw.items()}
        try:
            epoch = number(row.get("epoch"), "epoch")
            if epoch < 0 or not epoch.is_integer():
                raise ValueError("epoch must be a nonnegative integer")
            metrics = {k: number(row.get(col), k) if col else None for k, col in selected.items()}
            if any(v is not None and not 0 <= v <= 1 for v in metrics.values()):
                raise ValueError("metrics must be fractions in [0, 1], not percentages")
            valid.append(dict(epoch=int(epoch), **metrics, line=line))
        except ValueError as exc:
            rejected.append({"line": line, "reason": str(exc)})
    if not valid:
        raise ValueError("no valid metric rows")
    # Select all metrics from the same best row. Never combine independent maxima.
    best = max(valid, key=lambda r: r["map50_95"])
    epochs = [r["epoch"] for r in valid]
    warnings = []
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        warnings.append("non-monotonic or repeated epochs: resumed segments may be present")
    meta = dict(metadata or {})
    if meta.get("evidence_kind") != "measured":
        warnings.append("not verified as measured evidence")
    return {"name": name, "csv_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "best_validation_row": best, "valid_rows": len(valid), "rejected_rows": rejected,
            "metadata": meta, "warnings": warnings,
            "claim_boundary": "best validation row; not test performance, training completion or best.pt verification"}


def load_run(directory: Path) -> dict:
    csv_path = directory / "results.csv"
    meta_path = directory / "manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return summarize_csv(csv_path.read_text(encoding="utf-8-sig"), meta, directory.name)


def compare_runs(left: dict, right: dict, declared_change: str | None = None) -> dict:
    a, b = left.get("metadata", {}), right.get("metadata", {})
    missing, confounds = [], []
    for key in CONTEXT_KEYS:
        if a.get(key) is None or b.get(key) is None or a.get(key) == "" or b.get(key) == "":
            missing.append(key)
        elif a[key] != b[key]:
            confounds.append(key)
    for key in ("model_config_sha256", "evidence_kind"):
        if not a.get(key) or not b.get(key):
            missing.append(key)
    measured = a.get("evidence_kind") == b.get("evidence_kind") == "measured"
    delta = {k: (right["best_validation_row"][k] - left["best_validation_row"][k]) * 100
             if left["best_validation_row"][k] is not None and right["best_validation_row"][k] is not None else None
             for k in METRICS}
    return {"left": left["name"], "right": right["name"], "delta_percentage_points": delta,
            "missing_context": sorted(set(missing)), "confounded_fields": confounds,
            "controlled_comparison_candidate": not missing and not confounds and measured and bool(declared_change),
            "declared_change": declared_change, "causal_improvement_proven": False,
            "claim_boundary": "Observed validation differences only. Even controlled single-seed runs do not prove a causal improvement."}
