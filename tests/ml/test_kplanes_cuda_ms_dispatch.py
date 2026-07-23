"""CPU-only dispatch coverage for the optional multiscale K-Planes CUDA op."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from quantem.core import config
from quantem.core.ml.models.kplanes import interpolate_ms_features_tilted


def _inputs(num_grids=3):
    pts = SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(32, 3))
    rotations = SimpleNamespace(dtype=torch.float32, shape=(2, 3, 3))
    grids = [torch.empty((6, 4, 1, 1)) for _ in range(num_grids)]
    return pts, rotations, grids


@pytest.fixture
def mocked_cuda_ml(monkeypatch):
    calls = {"single": [], "ms": [], "ms_tv": []}
    cuda_ml = ModuleType("quantem.cuda.core.ml")

    def single(pts, rotations, grid):
        calls["single"].append((pts, rotations, grid))
        return torch.zeros((pts.shape[0], rotations.shape[0] * grid.shape[1]))

    def multiscale(*args):
        calls["ms"].append((args[2:5], args[5:8]))
        width = args[1].shape[0] * sum(grid.shape[1] for grid in args[2:5])
        return torch.zeros((args[0].shape[0], width))

    def multiscale_tv(*args):
        calls["ms_tv"].append((args[2:5], args[5:8]))
        width = args[1].shape[0] * sum(grid.shape[1] for grid in args[2:5])
        return torch.zeros((args[0].shape[0], width)), torch.tensor(2.5)

    cuda_ml.kplanes_tilted_fuse = single
    cuda_ml._kplanes_tilted_fuse_builtin = single
    cuda_ml.kplanes_tilted_fuse_ms = multiscale
    cuda_ml.kplanes_tilted_fuse_ms_tv = multiscale_tv

    cuda = ModuleType("quantem.cuda")
    cuda.__path__ = []
    cuda_core = ModuleType("quantem.cuda.core")
    cuda_core.__path__ = []
    cuda.core = cuda_core
    cuda_core.ml = cuda_ml
    monkeypatch.setattr(sys.modules["quantem"], "cuda", cuda, raising=False)
    monkeypatch.setitem(sys.modules, "quantem.cuda", cuda)
    monkeypatch.setitem(sys.modules, "quantem.cuda.core", cuda_core)
    monkeypatch.setitem(sys.modules, "quantem.cuda.core.ml", cuda_ml)
    monkeypatch.setattr(config, "get", lambda key, default=None: True)
    return cuda_ml, calls


def test_three_scales_dispatch_to_one_multiscale_op(mocked_cuda_ml):
    _, calls = mocked_cuda_ml
    pts, rotations, grids = _inputs()
    output = interpolate_ms_features_tilted(pts, grids, rotations, scale_gates=(0.25, 0.75, 1.0))

    assert output.shape == (pts.shape[0], rotations.shape[0] * 4 * 3)
    assert len(calls["ms"]) == 1
    called_grids, called_scales = calls["ms"][0]
    assert all(actual is expected for actual, expected in zip(called_grids, grids))
    assert called_scales == (0.25, 0.75, 1.0)
    assert calls["single"] == []


def test_none_scale_gates_dispatch_as_unit_gates(mocked_cuda_ml):
    _, calls = mocked_cuda_ml
    pts, rotations, grids = _inputs()

    interpolate_ms_features_tilted(pts, grids, rotations, scale_gates=None)

    assert len(calls["ms"]) == 1
    assert calls["ms"][0][1] == (1.0, 1.0, 1.0)
    assert calls["single"] == []


def test_plane_tv_defaults_to_combined_capability(mocked_cuda_ml, monkeypatch):
    _, calls = mocked_cuda_ml
    monkeypatch.delenv("QUANTEM_KPLANES_MS_TV_FUSED", raising=False)
    pts, rotations, grids = _inputs()

    features, tv = interpolate_ms_features_tilted(
        pts, grids, rotations, scale_gates=(0.25, 0.75, 1.0), include_plane_tv=True
    )

    assert features.shape == (pts.shape[0], rotations.shape[0] * 4 * 3)
    assert tv.item() == 2.5
    assert len(calls["ms_tv"]) == 1
    assert calls["ms"] == []


def test_plane_tv_opt_out_uses_multiscale_without_aux(mocked_cuda_ml, monkeypatch):
    _, calls = mocked_cuda_ml
    monkeypatch.setenv("QUANTEM_KPLANES_MS_TV_FUSED", "0")
    pts, rotations, grids = _inputs()

    interpolate_ms_features_tilted(pts, grids, rotations, include_plane_tv=True)

    assert len(calls["ms"]) == 1
    assert calls["ms_tv"] == []


@pytest.mark.parametrize("num_grids", [2, 4])
def test_non_three_scale_counts_use_per_level_fallback(mocked_cuda_ml, num_grids):
    _, calls = mocked_cuda_ml
    pts, rotations, grids = _inputs(num_grids)

    interpolate_ms_features_tilted(pts, grids, rotations)

    assert len(calls["single"]) == len(grids)
    assert all(call[2] is grid for call, grid in zip(calls["single"], grids))
    assert calls["ms"] == []


def test_missing_multiscale_op_uses_per_level_fallback(mocked_cuda_ml, monkeypatch):
    cuda_ml, calls = mocked_cuda_ml
    monkeypatch.setattr(cuda_ml, "kplanes_tilted_fuse_ms", None)
    pts, rotations, grids = _inputs()

    interpolate_ms_features_tilted(pts, grids, rotations)

    assert len(calls["single"]) == len(grids)
    assert all(call[2] is grid for call, grid in zip(calls["single"], grids))
    assert calls["ms"] == []


def test_overridden_single_level_op_uses_per_level_fallback(mocked_cuda_ml, monkeypatch):
    cuda_ml, calls = mocked_cuda_ml
    instrumented_calls = []
    builtin = cuda_ml.kplanes_tilted_fuse

    def instrumented(*args):
        instrumented_calls.append(args)
        return builtin(*args)

    monkeypatch.setattr(cuda_ml, "kplanes_tilted_fuse", instrumented)
    pts, rotations, grids = _inputs()

    interpolate_ms_features_tilted(pts, grids, rotations)

    assert len(instrumented_calls) == len(grids)
    assert all(call[2] is grid for call, grid in zip(instrumented_calls, grids))
    assert len(calls["single"]) == len(grids)
    assert calls["ms"] == []
