from fieldvision.copilot import EvidenceCopilot
from fieldvision.retrieval import Retriever, load_corpus
from fieldvision.service import SAMPLES


def test_weak_matches_do_not_pad_the_model_context():
    r = EvidenceCopilot(Retriever(load_corpus(SAMPLES / "corpus"))).ask(
        "为什么不能把不同 epoch 的 precision 和 mAP 最大值拼在一起？")
    assert r["citations"]
    assert all(c["source"] == "experiment_metrics.md" for c in r["citations"])
