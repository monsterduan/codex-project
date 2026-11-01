import httpx
import pytest
from fieldvision.copilot import EvidenceCopilot, ModelConfig, complete
from fieldvision.retrieval import Retriever, load_corpus
from fieldvision.service import SAMPLES

CONFIG = ModelConfig("http://127.0.0.1:9000/v1", "test-model")
QUESTION = "单应性矩阵需要多少非共线对应点"


def copilot(completion):
    return EvidenceCopilot(Retriever(load_corpus(SAMPLES / "corpus")), CONFIG, completion)


def test_valid_model_citation():
    r = copilot(lambda *_: {"text": "至少四组。[E1]", "usage": {"total_tokens": 30}}).ask(QUESTION)
    assert not r["refused"] and r["mode"] == "llm"
    assert r["usage"]["total_tokens"] == 30


@pytest.mark.parametrize("answer", ["无引用", "错误引用[E0]", "不存在的证据[E99]", "混用[E1][E999]"])
def test_invalid_citations_fail_closed(answer):
    r = copilot(lambda *_: {"text": answer}).ask(QUESTION)
    assert r["refused"] and r["reason"] == "invalid_citations"


def test_network_failure_is_not_a_fake_answer():
    def failure(*_):
        raise httpx.ReadTimeout("timeout")
    r = copilot(failure).ask(QUESTION)
    assert r["refused"] and r["reason"] == "provider_error"


def test_provider_wire_format_and_no_redirect(monkeypatch):
    original = httpx.Client
    observed = {}
    def handler(request):
        observed["request"] = request
        return httpx.Response(200, json={"choices": [{"message": {"content": "四组[E1]"}}]})
    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return original(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(httpx, "Client", client)
    assert complete(CONFIG, QUESTION, [{"source": "manual.md", "text": "四组"}])["text"] == "四组[E1]"
    assert str(observed["request"].url) == "http://127.0.0.1:9000/v1/chat/completions"
    assert b"untrusted_evidence" in observed["request"].content


@pytest.mark.parametrize("url", ["http://example.com", "https://user:secret@example.com", "file:///secret", "https://example.com/?key=secret"])
def test_unsafe_endpoint_config_rejected(url):
    with pytest.raises(ValueError):
        ModelConfig(url, "model")


def test_partial_environment_is_error(monkeypatch):
    monkeypatch.setenv("FIELDVISION_LLM_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.delenv("FIELDVISION_LLM_MODEL", raising=False)
    with pytest.raises(ValueError):
        ModelConfig.from_environment()


@pytest.mark.parametrize("question", [" ", "a" * 2001])
def test_question_limits(question):
    with pytest.raises(ValueError):
        copilot(lambda *_: {}).ask(question)
