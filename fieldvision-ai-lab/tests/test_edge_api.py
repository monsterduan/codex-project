import json
import math
import pytest
from fastapi.testclient import TestClient
from fieldvision.edge import diagnose, percentile
from fieldvision.evaluation import evaluate
from fieldvision.service import SAMPLES, create_app


def test_missing_telemetry_is_unknown_not_pass():
    d = diagnose({})
    assert d["status"] == "insufficient_evidence" and not d["hardware_acceptance_proven"]
    assert all(c["state"] == "unknown" for c in d["checks"])


@pytest.mark.parametrize("v", [-1, float("nan"), float("inf"), "40", True])
def test_malformed_telemetry_rejected(v):
    with pytest.raises(ValueError):
        diagnose({"inference_ms": v})


def test_demo_thresholds_and_hash_mismatch():
    r = json.loads((SAMPLES / "telemetry.json").read_text())
    d = diagnose(r)
    assert d["evidence_kind"] == "synthetic_demo"
    assert all(c["state"] == "warning" for c in d["checks"])


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], .95) == pytest.approx(38.5)


def test_http_contracts():
    with TestClient(create_app()) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["model_mode"] == "extractive_baseline"
        r = client.post("/api/ask", json={"question": "单应性矩阵需要非共线对应点吗"})
        assert r.status_code == 200 and r.json()["citations"]
        assert client.post("/api/ask", json={"question": ""}).status_code == 422
        assert client.post("/api/runs/analyze", json={"csv_text": "bad"}).status_code == 400
        assert "split_sha256" in client.get("/api/demo/comparison").json()["confounded_fields"]
        assert client.post("/api/edge/diagnose", json={"inference_ms": -1}).status_code == 400
        assert client.post("/api/edge/diagnose", json={}).json()["status"] == "insufficient_evidence"


def test_authored_regression_set_not_generalization():
    app = create_app()
    report = evaluate(app.state.copilot, SAMPLES / "eval_cases.json")
    assert report["passed"] == report["case_count"] == 20
    assert "not independent" in report["claim_boundary"]


def test_duplicate_evaluation_ids_rejected(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text('[{"id":"x"},{"id":"x"}]')
    with pytest.raises(ValueError):
        evaluate(create_app().state.copilot, p)
