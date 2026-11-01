"""Retrieval and refusal regression on explicit fixtures, not a leaderboard."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import statistics
from . import __version__
from .copilot import EvidenceCopilot


def evaluate(copilot: EvidenceCopilot, cases_path: Path) -> dict:
    raw = cases_path.read_bytes()
    cases = json.loads(raw)
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("evaluation needs nonempty, uniquely identified cases")
    results, latencies = [], []
    answerable = refusal_cases = hit_count = correct_refusals = citation_matches = 0
    reciprocal_ranks = []
    for case in cases:
        r = copilot.ask(case["question"])
        latencies.append(r["elapsed_ms"])
        should_refuse = case.get("should_refuse", False)
        if should_refuse:
            refusal_cases += 1
            correct_refusals += int(r["refused"])
        else:
            answerable += 1
            expected = set(case["expected_sources"])
            rank = next((i for i, c in enumerate(r["citations"], 1) if c["source"] in expected), None)
            hit_count += int(rank is not None)
            reciprocal_ranks.append(1/rank if rank else 0)
            citation_matches += int(not r["refused"] and rank is not None)
        passed = r["refused"] if should_refuse else not r["refused"] and rank is not None
        results.append({"id": case["id"], "passed": passed, "refused": r["refused"], "mode": r["mode"],
                        "sources": [c["source"] for c in r["citations"]], "reason": r["reason"]})
    return {"version": __version__, "python": platform.python_version(), "case_set_sha256": hashlib.sha256(raw).hexdigest(),
            "corpus_fingerprint": hashlib.sha256(json.dumps([(c.id, c.sha256) for c in copilot.retriever.chunks]).encode()).hexdigest(),
            "case_count": len(cases), "passed": sum(r["passed"] for r in results),
            "retrieval_hit_at_3": hit_count / answerable if answerable else None,
            "mrr_at_3": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
            "correct_refusal_rate": correct_refusals / refusal_cases if refusal_cases else None,
            "answer_source_match_rate": citation_matches / answerable if answerable else None,
            "p50_ms": statistics.median(latencies), "results": results,
            "claim_boundary": "Small authored synthetic regression set; not independent field validation, answer factuality or generalization accuracy."}
