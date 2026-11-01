"""Local-only API composition. Source: https://fastapi.tiangolo.com/tutorial/"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .copilot import EvidenceCopilot, ModelConfig
from .edge import diagnose
from .retrieval import LocalSentenceRanker, Retriever, load_corpus
from .runs import compare_runs, load_run, summarize_csv

SAMPLES = Path(__file__).parent / "sample_data"


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RunInput(BaseModel):
    name: str = Field(default="uploaded", min_length=1, max_length=100)
    csv_text: str = Field(min_length=1, max_length=1_000_000)
    metadata: dict = Field(default_factory=dict)


class Comparison(BaseModel):
    left: RunInput
    right: RunInput
    declared_change: str | None = Field(default=None, max_length=300)


def create_app(corpus_path: Path | None = None, config: ModelConfig | None = None) -> FastAPI:
    root = corpus_path or Path(os.getenv("FIELDVISION_CORPUS", str(SAMPLES / "corpus")))
    dense_path = os.getenv("FIELDVISION_EMBEDDING_PATH")
    dense = LocalSentenceRanker(dense_path) if dense_path else None
    copilot = EvidenceCopilot(Retriever(load_corpus(root), dense), config)
    app = FastAPI(title="FieldVision AI Lab", version=__version__)
    app.state.copilot = copilot

    @app.exception_handler(ValueError)
    async def invalid_data(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        return {"version": __version__, "model_mode": "llm" if config else "extractive_baseline",
                "retrieval": "hybrid_rrf" if dense else "bm25_lexical", "chunks": len(copilot.retriever.chunks)}

    @app.post("/api/ask")
    def ask(body: Question):
        return copilot.ask(body.question)

    @app.get("/api/demo/runs")
    def demo_runs():
        return [load_run(p) for p in sorted((SAMPLES / "runs").iterdir()) if p.is_dir()]

    @app.get("/api/demo/comparison")
    def demo_comparison():
        return compare_runs(load_run(SAMPLES / "runs" / "baseline"),
                            load_run(SAMPLES / "runs" / "candidate"), "demo model configuration")

    @app.post("/api/runs/analyze")
    def analyze(body: RunInput):
        return summarize_csv(body.csv_text, body.metadata, body.name)

    @app.post("/api/runs/compare")
    def compare(body: Comparison):
        return compare_runs(summarize_csv(body.left.csv_text, body.left.metadata, body.left.name),
                            summarize_csv(body.right.csv_text, body.right.metadata, body.right.name), body.declared_change)

    @app.get("/api/demo/edge")
    def demo_edge():
        data = json.loads((SAMPLES / "telemetry.json").read_text(encoding="utf-8"))
        return {"input": data, "diagnosis": diagnose(data)}

    @app.post("/api/edge/diagnose")
    def edge(body: dict):
        return diagnose(body)

    return app
