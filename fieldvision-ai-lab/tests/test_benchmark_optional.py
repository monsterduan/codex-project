"""Run with pip install '.[benchmark]' onnx; real CPU identity-model smoke test."""
import pytest
from fieldvision.edge import benchmark_onnx


def test_invalid_counts_do_not_load_model(tmp_path):
    with pytest.raises(ValueError):
        benchmark_onnx(tmp_path / "x", tmp_path / "y", iterations=0)


def test_real_cpu_onnx_adapter(tmp_path):
    onnx = pytest.importorskip("onnx")
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    helper, tensor = onnx.helper, onnx.TensorProto
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "adapter-smoke",
                              [helper.make_tensor_value_info("x", tensor.FLOAT, [1, 3, 8, 8])],
                              [helper.make_tensor_value_info("y", tensor.FLOAT, [1, 3, 8, 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "identity.onnx"
    onnx.save(model, path)
    data = tmp_path / "input.npz"
    np.savez(data, x=np.zeros((1, 3, 8, 8), dtype=np.float32))
    r = benchmark_onnx(path, data, iterations=5, warmup=2)
    assert r["evidence_kind"] == "measured"
    assert r["provider"] == "CPUExecutionProvider"
    assert len(r["raw_timings_ms"]) == 5 and r["p95_ms"] > 0
