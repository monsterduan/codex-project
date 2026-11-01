import copy
import pytest
from fieldvision.runs import compare_runs, load_run, summarize_csv
from fieldvision.service import SAMPLES

HEADER = "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"


def test_metrics_come_from_same_row():
    r = load_run(SAMPLES / "runs" / "baseline")
    assert r["best_validation_row"]["epoch"] == 2
    assert r["best_validation_row"]["precision"] == .85


def test_nan_inf_rows_are_rejected():
    r = summarize_csv(HEADER + "1,0.8,0.7,0.8,nan\n2,0.9,0.8,inf,0.8\n3,0.8,0.7,0.8,0.6")
    assert r["valid_rows"] == 1 and len(r["rejected_rows"]) == 2


@pytest.mark.parametrize("text", ["", "epoch,other\n1,0.4", HEADER + "1,80,70,90,60", HEADER + "-1,.8,.7,.9,.6"])
def test_invalid_csv_rejected(text):
    with pytest.raises(ValueError):
        summarize_csv(text)


def test_whitespace_headers_supported():
    r = summarize_csv(" epoch , metrics/mAP50-95(B) \n 1 , .5 ")
    assert r["best_validation_row"]["map50_95"] == .5


def test_resumed_epochs_warn():
    r = summarize_csv(HEADER + "1,.8,.7,.8,.6\n2,.8,.7,.8,.7\n1,.8,.7,.8,.65")
    assert any("segments" in w for w in r["warnings"])


def test_confounded_demo_does_not_claim_improvement():
    a, b = [load_run(SAMPLES / "runs" / n) for n in ["baseline", "candidate"]]
    c = compare_runs(a, b, "model")
    assert c["delta_percentage_points"]["map50_95"] == pytest.approx(3)
    assert "split_sha256" in c["confounded_fields"]
    assert not c["controlled_comparison_candidate"]
    assert not c["causal_improvement_proven"]


def test_even_controlled_single_seed_does_not_prove_causality():
    a = load_run(SAMPLES / "runs" / "baseline")
    b = copy.deepcopy(a)
    a["metadata"]["evidence_kind"] = b["metadata"]["evidence_kind"] = "measured"
    b["metadata"]["model_config_sha256"] = "another-model"
    c = compare_runs(a, b, "architecture")
    assert c["controlled_comparison_candidate"] and not c["causal_improvement_proven"]


def test_empty_metadata_is_not_a_controlled_comparison():
    a = summarize_csv(HEADER + "1,.8,.7,.8,.6")
    c = compare_runs(a, a, "model")
    assert c["missing_context"] and not c["controlled_comparison_candidate"]
