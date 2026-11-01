# Contributing

Use a small issue or pull request with a reproducible input and a failing case. Keep evidence labels explicit: synthetic example, user-supplied measurement, locally measured benchmark, or unknown.

Before submitting:

```bash
pip install -e ".[dev]"
pytest
fieldvision evaluate
```

Optional CPU adapter check: install `.[benchmark]` and `onnx`, then run `tests/test_benchmark_optional.py`. The test creates an Identity model; do not present its latency as a computer-vision benchmark.

Do not include credentials, private documents, model checkpoints, or unauthorized images. Put experimental material under ignored `data/private/` and publish a license-compatible reproduction fixture instead.

For retrieval changes, keep the regression cases fixed and add held-out human-labeled data before claiming an improvement. For comparisons, preserve the same epoch and identify data/configuration differences. Explain what your check proves and what it cannot prove.
