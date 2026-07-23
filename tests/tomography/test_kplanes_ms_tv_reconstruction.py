"""CPU integration coverage for reconstruction-scoped multiscale plane-TV fusion."""

import sys
from types import ModuleType

import pytest
import torch

from quantem.core import config
from quantem.core.ml.models.kplanes import KPlanesTILTED
from quantem.tomography.object_models import ObjectTensorDecomp
from quantem.tomography.tomography_context import ReconstructionContext


class _PretendCudaTensor(torch.Tensor):
    """CPU tensor subclass that reaches the mocked optional-CUDA dispatch seam."""

    @property
    def is_cuda(self):
        return True


@pytest.fixture(autouse=True)
def preserve_torch_rng_state():
    state = torch.random.get_rng_state()
    yield
    torch.random.set_rng_state(state)


@pytest.fixture
def mocked_cuda_ml(monkeypatch):
    calls = {"ms": 0, "ms_tv": 0}
    cuda_ml = ModuleType("quantem.cuda.core.ml")

    def multiscale(pts, rotations, grid0, grid1, grid2, *gates):
        calls["ms"] += 1
        width = rotations.shape[0] * sum(grid.shape[1] for grid in (grid0, grid1, grid2))
        return torch.zeros((pts.shape[0], width), dtype=grid0.dtype)

    def multiscale_tv(pts, rotations, grid0, grid1, grid2, *gates):
        calls["ms_tv"] += 1
        features = multiscale(pts, rotations, grid0, grid1, grid2, *gates)
        calls["ms"] -= 1
        return features, torch.tensor(2.5)

    cuda_ml.kplanes_tilted_fuse = lambda pts, rotations, grid: None
    cuda_ml._kplanes_tilted_fuse_builtin = cuda_ml.kplanes_tilted_fuse
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
    monkeypatch.delenv("QUANTEM_KPLANES_MS_TV_FUSED", raising=False)
    return calls


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason=(
        "CPU-only mock integration test; real GPU path covered by "
        "test_kplanes_ms_tv_optimizer.py and the three-level cuda_graphs arm"
    ),
)
def test_reconstruction_scope_requests_and_consumes_fused_plane_tv_once(
    mocked_cuda_ml, monkeypatch
):
    model = KPlanesTILTED(
        T=1,
        M_features=2,
        resolution=(4, 4, 4),
        multiscale_res_multipliers=(1, 2, 3),
    )
    obj = ObjectTensorDecomp.from_model(model, shape=(4, 4, 4), device="cpu")
    obj.constraints.tv_plane = 0.2
    coords = torch.zeros((3, 3)).as_subclass(_PretendCudaTensor)

    generic_density = obj.forward(coords)
    assert generic_density.shape == (3,)
    assert mocked_cuda_ml == {"ms": 1, "ms_tv": 0}

    with obj.reconstruction_forward_context():
        reconstruction_density = obj.forward(coords)
    assert reconstruction_density.shape == (3,)
    assert mocked_cuda_ml == {"ms": 1, "ms_tv": 1}
    assert model._plane_tv_fusion_requested is False

    def standalone_path_must_not_run():
        raise AssertionError("standalone plane TV was computed after the fused auxiliary")

    monkeypatch.setattr(obj, "_get_plane_tv_loss", standalone_path_must_not_run)
    loss = obj.get_tv_loss(ReconstructionContext(coords=coords, pred=reconstruction_density))
    torch.testing.assert_close(loss, torch.tensor(0.5))
    assert obj._fused_plane_tv_loss is None


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason=(
        "CPU-only mock integration test; real GPU path covered by "
        "test_kplanes_ms_tv_optimizer.py and the three-level cuda_graphs arm"
    ),
)
def test_reconstruction_scope_clears_fused_plane_tv_after_exception(mocked_cuda_ml):
    model = KPlanesTILTED(
        T=1,
        M_features=2,
        resolution=(4, 4, 4),
        multiscale_res_multipliers=(1, 2, 3),
    )
    obj = ObjectTensorDecomp.from_model(model, shape=(4, 4, 4), device="cpu")

    with pytest.raises(RuntimeError, match="failed reconstruction"):
        with obj.reconstruction_forward_context():
            obj._fused_plane_tv_loss = torch.tensor(2.5, requires_grad=True)
            raise RuntimeError("failed reconstruction")

    assert obj._fused_plane_tv_loss is None
    assert model._plane_tv_fusion_requested is False
