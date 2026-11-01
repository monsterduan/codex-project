# Third-party components and attribution

Original project code and authored synthetic samples: MIT, copyright 2026 monsterduan.

The package uses dependencies rather than vendoring or renaming upstream code. Follow each component's license when distributing it. Model weights, datasets and training frameworks are not included.

| Component | Role | Upstream |
|---|---|---|
| LangGraph | State workflow | https://github.com/langchain-ai/langgraph (MIT) |
| FastAPI | Local API | https://github.com/fastapi/fastapi (MIT) |
| Uvicorn | Local server | https://github.com/Kludex/uvicorn (BSD-3-Clause) |
| HTTPX | Explicit model HTTP adapter | https://github.com/encode/httpx (BSD-3-Clause) |
| Pytest | Tests | https://github.com/pytest-dev/pytest (MIT) |
| Sentence Transformers | Optional local embedding adapter | https://github.com/huggingface/sentence-transformers (Apache-2.0) |
| NumPy | Optional tensor/input handling | https://github.com/numpy/numpy (BSD-3-Clause) |
| ONNX Runtime | Optional measured benchmark | https://github.com/microsoft/onnxruntime (MIT) |
| ONNX | Optional smoke-test fixture creation | https://github.com/onnx/onnx (Apache-2.0) |

Research references only: LlamaIndex, Qwen3-Embedding and LLaMA-Factory. Their models/data/tooling require their own licenses if adopted later. No Ultralytics source or pretrained weights are redistributed; CSV field-name compatibility is independently implemented. Dependency lock files also list transitive dependencies; this table is not a replacement for their license notices.
