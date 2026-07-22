"""CPU-only capability and kill-switch coverage for fused plane TV."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from quantem.core import config
from quantem.tomography import object_models


def _eager_reference(grids, rotations):
    levels = []
    for grid in grids:
        dh = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
        dw = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
        levels.append((dh + dw).view(rotations, 3).sum(dim=1).mean())
    return torch.stack(levels).sum()


@pytest.fixture
def mocked_plane_tv(monkeypatch):
    calls = []
    cuda_ml = ModuleType("quantem.cuda.core.ml")

    def fused(*grids):
        calls.append(grids)
        return torch.tensor(7.0)

    cuda_ml.plane_tv_loss = fused
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
    monkeypatch.delenv("QUANTEM_PLANE_TV_FUSED", raising=False)
    return cuda_ml, calls


def _pretend_cuda_grids():
    return tuple(SimpleNamespace(is_cuda=True, dtype=torch.float32) for _ in range(3))


def test_three_cuda_levels_dispatch_to_fused_op(mocked_plane_tv):
    _, calls = mocked_plane_tv
    grids = _pretend_cuda_grids()
    loss = object_models._plane_tv_loss(grids, tilted=True, rotations=2)
    assert loss.item() == 7.0
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0], grids))


def test_cpu_levels_use_eager_fallback(mocked_plane_tv):
    _, calls = mocked_plane_tv
    generator = torch.Generator().manual_seed(1)
    grids = tuple(
        torch.rand((6, channels, height, width), generator=generator)
        for channels, height, width in ((2, 5, 7), (3, 9, 11), (4, 13, 17))
    )
    actual = object_models._plane_tv_loss(grids, tilted=True, rotations=2)
    torch.testing.assert_close(actual, _eager_reference(grids, rotations=2))
    assert calls == []


def test_missing_cuda_capability_uses_eager_fallback(mocked_plane_tv, monkeypatch):
    cuda_ml, calls = mocked_plane_tv
    del cuda_ml.plane_tv_loss
    expected = torch.tensor(3.5)
    eager_calls = []

    def eager(grids, tilted, rotations):
        eager_calls.append((grids, tilted, rotations))
        return expected

    monkeypatch.setattr(object_models, "_plane_tv_loss_eager", eager)
    actual = object_models._plane_tv_loss(_pretend_cuda_grids(), tilted=True, rotations=2)
    assert actual is expected
    assert len(eager_calls) == 1
    assert calls == []


def test_env_kill_switch_forces_eager_fallback(mocked_plane_tv, monkeypatch):
    _, calls = mocked_plane_tv
    monkeypatch.setenv("QUANTEM_PLANE_TV_FUSED", "0")
    expected = torch.tensor(2.25)
    eager_calls = []

    def eager(grids, tilted, rotations):
        eager_calls.append((grids, tilted, rotations))
        return expected

    monkeypatch.setattr(object_models, "_plane_tv_loss_eager", eager)
    actual = object_models._plane_tv_loss(_pretend_cuda_grids(), tilted=True, rotations=2)
    assert actual is expected
    assert len(eager_calls) == 1
    assert calls == []
