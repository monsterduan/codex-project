from pathlib import Path
import hashlib
import pytest

from fieldvision.retrieval import Retriever, load_corpus, tokens
from fieldvision.copilot import EvidenceCopilot
from fieldvision.service import SAMPLES


def test_chinese_bigrams_and_identifiers():
    assert "像素" in tokens("像素坐标与 mAP50")
    assert "map50" in tokens("mAP50")


def test_source_line_and_content_hash(tmp_path):
    p = tmp_path / "manual.md"
    p.write_text("# 标题\n\n像素坐标转换需要标定。\n原始记录必须保留。\n", encoding="utf-8")
    chunks = load_corpus(tmp_path)
    assert chunks[1].start_line == 3 and chunks[1].end_line == 4
    assert chunks[1].text == "像素坐标转换需要标定。\n原始记录必须保留。"
    assert chunks[1].sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


def test_missing_corpus_raises(tmp_path):
    with pytest.raises(ValueError):
        load_corpus(tmp_path / "missing")


def test_ranking_tracks_source():
    r = Retriever(load_corpus(SAMPLES / "corpus"))
    hits = r.search("单应性矩阵需要非共线对应点")
    assert hits[0]["source"] == "coordinate_calibration.md"
    assert hits[0]["start_line"] > 0


def test_empty_corpus_refuses():
    r = EvidenceCopilot(Retriever([])).ask("单应性矩阵需要对应点吗")
    assert r["refused"] and r["reason"] == "insufficient_evidence"
    assert r["trace"] == ["retrieve", "evidence_gate", "refuse"]


def test_local_baseline_does_not_claim_llm():
    r = EvidenceCopilot(Retriever(load_corpus(SAMPLES / "corpus"))).ask("单应性矩阵需要多少非共线对应点")
    assert r["mode"] == "extractive_baseline"
    assert "不是大模型生成" in r["answer"]
    assert r["usage"] == {}


def test_rrf_combines_ranks_without_confidence_claims():
    chunks = load_corpus(SAMPLES / "corpus")
    class FakeDense:
        def rank(self, query, chunks):
            return [(len(chunks) - 1, .8)]
    r = Retriever(chunks, FakeDense())
    assert r.search("RRF 融合词法检索语义检索")[0]["source"] == "rag_boundaries.md"
    assert r.search("RRF 融合词法检索语义检索")[0]["score"] < 1


@pytest.mark.parametrize("k", [0, -1, 21])
def test_retrieval_limit(k):
    with pytest.raises(ValueError):
        Retriever([]).search("query", k)
