"""Structured diagnostic evidence, with explicit unknowns and no actuator API."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import platform
import statistics
import time


def diagnose(record: dict) -> dict:
    checks = []
    def add(code, state, evidence, suggestion):
        checks.append(dict(code=code, state=state, evidence=evidence, suggestion=suggestion))
    latency = record.get("inference_ms")
    target = record.get("target_ms")
    for key in ("inference_ms", "target_ms", "temperature_c", "free_memory_mb", "dropped_frame_ratio"):
        v = record.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
            raise ValueError(f"invalid telemetry: {key}")
    if record.get("dropped_frame_ratio", 0) > 1:
        raise ValueError("dropped_frame_ratio must be within [0, 1]")
    if latency is None or target is None:
        add("latency", "unknown", "latency or target is missing", "measure latency on the target device")
    else:
        add("latency", "warning" if latency > target else "pass", f"{latency} ms; target {target} ms",
            "profile preprocessing, inference and postprocessing separately")
    temp = record.get("temperature_c")
    add("temperature", "unknown" if temp is None else "warning" if temp >= 80 else "pass",
        f"temperature={temp}; 80 C is a demo alert threshold, not a device specification",
        "check the vendor limit and throttling logs before blaming temperature")
    memory = record.get("free_memory_mb")
    add("memory", "unknown" if memory is None else "warning" if memory < 512 else "pass",
        f"free_memory_mb={memory}; demo threshold=512", "check memory pressure and input queues")
    dropped = record.get("dropped_frame_ratio")
    add("frame_drop", "unknown" if dropped is None else "warning" if dropped > .05 else "pass",
        f"dropped_frame_ratio={dropped}; demo threshold=0.05", "inspect the capture rate, queue capacity and backpressure")
    supplied, expected = record.get("model_sha256"), record.get("expected_model_sha256")
    add("model_identity", "unknown" if not supplied or not expected else "pass" if supplied == expected else "warning",
        "hash not supplied" if not supplied or not expected else f"reported={supplied}; expected={expected}",
        "compare the deployed file hash against the release manifest")
    observed = record.get("evidence_kind", "unknown")
    return {"evidence_kind": observed, "checks": checks, "hardware_acceptance_proven": False,
            "status": "needs_attention" if any(c["state"] == "warning" for c in checks) else "insufficient_evidence" if any(c["state"] == "unknown" for c in checks) else "within_demo_thresholds",
            "claim_boundary": "Rule-based triage; thresholds are illustrative. No remote measurement or physical action was performed."}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * quantile
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def benchmark_onnx(model_path: Path, inputs_path: Path, iterations: int = 50, warmup: int = 5,
                   provider: str = "CPUExecutionProvider") -> dict:
    """Real optional benchmark, no fake fallback.

    Source: https://onnxruntime.ai/docs/api/python/api_summary.html
    NPZ inputs must match model input names; accuracy is deliberately not inferred.
    """
    if not 1 <= iterations <= 10_000 or not 0 <= warmup <= 1_000:
        raise ValueError("invalid benchmark iteration count")
    import numpy as np
    import onnxruntime as ort
    if provider not in ort.get_available_providers():
        raise ValueError(f"requested execution provider is unavailable: {provider}")
    session = ort.InferenceSession(str(model_path), providers=[provider])
    with np.load(inputs_path, allow_pickle=False) as archive:
        feed = {entry.name: archive[entry.name] for entry in session.get_inputs()}
    for _ in range(warmup):
        session.run(None, feed)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        session.run(None, feed)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return {"evidence_kind": "measured", "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
            "platform": platform.platform(), "processor": platform.processor(), "runtime_version": ort.__version__,
            "provider": provider, "active_providers": session.get_providers(), "iterations": iterations,
            "warmup": warmup, "p50_ms": percentile(timings, .5), "p95_ms": percentile(timings, .95),
            "mean_ms": statistics.mean(timings), "raw_timings_ms": timings,
            "claim_boundary": "Synchronous session.run wall time only, including provider transfers; not camera-to-actuator latency, sustained FPS or accuracy."}
