from __future__ import annotations
import argparse
import json
from pathlib import Path

from .copilot import EvidenceCopilot, ModelConfig
from .edge import benchmark_onnx, diagnose
from .evaluation import evaluate
from .retrieval import Retriever, load_corpus
from .runs import compare_runs, load_run
from .service import SAMPLES, create_app


def emit(value, destination: str | None = None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description="FieldVision AI Lab")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--corpus", type=Path, default=SAMPLES / "corpus")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--cases", type=Path, default=SAMPLES / "eval_cases.json")
    ev.add_argument("--corpus", type=Path, default=SAMPLES / "corpus")
    ev.add_argument("--output")
    run = commands.add_parser("run")
    run.add_argument("directory", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--declared-change")
    edge = commands.add_parser("diagnose")
    edge.add_argument("record", type=Path)
    bench = commands.add_parser("benchmark")
    bench.add_argument("model", type=Path)
    bench.add_argument("inputs", type=Path)
    bench.add_argument("--provider", default="CPUExecutionProvider")
    bench.add_argument("--iterations", type=int, default=50)
    bench.add_argument("--output")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run(create_app(config=ModelConfig.from_environment()), host="127.0.0.1", port=args.port)
    elif args.command == "ask":
        app = create_app(args.corpus, ModelConfig.from_environment())
        emit(app.state.copilot.ask(args.question))
    elif args.command == "evaluate":
        # Always evaluate reproducible offline baseline unless explicitly using Python API.
        cp = EvidenceCopilot(Retriever(load_corpus(args.corpus)))
        emit(evaluate(cp, args.cases), args.output)
    elif args.command == "run":
        emit(load_run(args.directory))
    elif args.command == "compare":
        emit(compare_runs(load_run(args.left), load_run(args.right), args.declared_change))
    elif args.command == "diagnose":
        emit(diagnose(json.loads(args.record.read_text(encoding="utf-8"))))
    elif args.command == "benchmark":
        emit(benchmark_onnx(args.model, args.inputs, args.iterations, provider=args.provider), args.output)


if __name__ == "__main__":
    main()
