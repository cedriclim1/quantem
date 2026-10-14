"""BF16 autocast / FP32 precision contract of tomography INR reconstruction (PR #280 design)."""

import numpy as np
import pytest
import torch
from torch import nn

from quantem.core import config
from quantem.core.ml.inr import HSiren
from quantem.core.ml.models.kplanes import (
    CPTilted,
    KPlanesTILTED,
    _rotate_tilted_points_fp32,
    gaussian_blur_planes,
    interpolate_ms_features_tilted,
)
from quantem.core.ml.optimizer_mixin import OptimizerParams
from quantem.tomography import object_models
from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjectINR,
    ObjectTensorDecomp,
    _to_fp32,
    _tomography_autocast,
)

from .conftest import requires_gpu

requires_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="requires a bf16-capable CUDA device",
)
requires_quantem_cuda = pytest.mark.skipif(
    not config.get("has_quantem_cuda"), reason="requires quantem-cuda"
)


class _LinearDtypes:
    """Record the output dtype of every nn.Linear call while active."""

    def __init__(self):
        self.dtypes: list[torch.dtype] = []
        self._handle = None

    def __enter__(self):
        def hook(module, args, output):
            if isinstance(module, nn.Linear):
                self.dtypes.append(output.dtype)

        self._handle = nn.modules.module.register_module_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()


def _tilted(T: int = 2, n: int = 8) -> KPlanesTILTED:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return KPlanesTILTED(
            M_features=2, resolution=(n, n, n), multiscale_res_multipliers=[1, 2], T=T
        )


def _cptilted(T: int = 2, n: int = 8) -> CPTilted:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return CPTilted(C=2, resolution=(n, n, n), T=T)


def _points(n: int = 128, seed: int = 1) -> torch.Tensor:
    return torch.rand(n, 3, generator=torch.Generator().manual_seed(seed)) * 2 - 1


class TestHelpers:
    def test_rotation_keeps_float64_and_upcasts_bf16(self):
        R = torch.eye(3).repeat(2, 1, 1)
        pts = _points(8)
        assert _rotate_tilted_points_fp32(pts.double(), R.double()).dtype == torch.float64
        assert _rotate_tilted_points_fp32(pts.bfloat16(), R).dtype == torch.float32
        assert _rotate_tilted_points_fp32(pts, R).dtype == torch.float32

    def test_float64_features_stay_float64_on_cpu(self):
        model = _tilted().double()
        feats = interpolate_ms_features_tilted(
            _points().double(), model.grids, model.so3.as_matrix()
        )
        assert feats.dtype == torch.float64

    def test_to_fp32(self):
        assert _to_fp32(torch.ones(2, dtype=torch.bfloat16)).dtype == torch.float32
        x32, x64 = torch.ones(2), torch.ones(2, dtype=torch.float64)
        assert _to_fp32(x32) is x32 and _to_fp32(x64) is x64

    @pytest.mark.parametrize("flag", [True, False])
    def test_autocast_is_off_on_cpu(self, flag):
        lin = nn.Linear(4, 2)
        with _tomography_autocast("cpu", flag):
            assert lin(torch.ones(1, 4)).dtype == torch.float32


@requires_gpu
class TestGPUPrecision:
    def test_setup_distributed_disables_tf32(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ObjectTensorDecomp.from_model(_tilted(), shape=(8, 8, 8), device="cuda:0")
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False

    @requires_bf16
    @pytest.mark.parametrize("kind", ["kplanes_tilted", "cptilted", "hsiren"])
    @pytest.mark.parametrize("flag", [True, False])
    def test_forward_runs_model_in_bf16_returns_fp32(self, kind, flag):
        if kind == "hsiren":
            obj = ObjectINR.from_model(
                HSiren(hidden_layers=1, hidden_features=16), shape=(8, 8, 8), device="cuda:0"
            )
        else:
            model = _tilted() if kind == "kplanes_tilted" else _cptilted()
            obj = ObjectTensorDecomp.from_model(model, shape=(8, 8, 8), device="cuda:0")
        coords = _points().cuda()
        with _LinearDtypes() as rec, _tomography_autocast("cuda", flag):
            out = obj.forward(coords)
        assert out.dtype == torch.float32
        expected = torch.bfloat16 if flag else torch.float32
        assert rec.dtypes and all(d == expected for d in rec.dtypes)

    @requires_bf16
    def test_blur_stays_fp32_under_autocast(self):
        grid = torch.rand(6, 2, 16, 16, device="cuda:0")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = gaussian_blur_planes(grid, 1.5)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, gaussian_blur_planes(grid, 1.5))

    @requires_bf16
    def test_tilted_features_fp32_under_autocast(self):
        model = _tilted().cuda()
        pts = _points().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = interpolate_ms_features_tilted(pts, model.grids, model.so3.as_matrix())
        assert feats.dtype == torch.float32
        torch.testing.assert_close(
            feats, interpolate_ms_features_tilted(pts, model.grids, model.so3.as_matrix())
        )

    @requires_quantem_cuda
    def test_kernel_casts_bf16_inputs_to_fp32(self, monkeypatch):
        import quantem.cuda.core.ml

        seen = []
        real = quantem.cuda.core.ml.kplanes_tilted_fuse

        def spy(pts, rotations, plane):
            seen.append((pts.dtype, rotations.dtype, plane.dtype))
            return real(pts, rotations, plane)

        monkeypatch.setattr(quantem.cuda.core.ml, "kplanes_tilted_fuse", spy)
        model = _tilted().cuda()
        pts, R = _points().cuda(), model.so3.as_matrix().detach()
        grids_bf16 = [g.detach().bfloat16().requires_grad_(True) for g in model.grids]
        out = interpolate_ms_features_tilted(pts, grids_bf16, R)
        assert out.dtype == torch.float32
        assert seen and all(d == (torch.float32,) * 3 for d in seen)
        reference = interpolate_ms_features_tilted(pts, [g.float() for g in grids_bf16], R)
        torch.testing.assert_close(out, reference)
        out.sum().backward()
        assert all(g.grad is not None and g.grad.dtype == torch.bfloat16 for g in grids_bf16)


def _dataset(n: int = 8, nang: int = 5) -> TomographyINRDataset:
    rng = np.random.default_rng(0)
    stack = (rng.random((nang, n, n)) * 5.0 + 0.5).astype(np.float32)
    return TomographyINRDataset.from_data(stack, np.linspace(-60, 60, nang, dtype="f4"))


def _optimizer_params() -> dict:
    return {
        "object": {
            "grids": OptimizerParams.Adam(lr=1e-2),
            "sigma_net": OptimizerParams.Adam(lr=1e-3),
            "so3": OptimizerParams.Adam(lr=1e-2),
        },
        "pose": OptimizerParams.Adam(lr=1e-3),
    }


ALL_TERMS = ObjConstraintParams.ObjTensorDecompConstraints(
    tv_vol=1e-4, tv_plane=1e-3, sparsity=1e-5, s3im_weight=0.1, l21_plane=1e-3, distortion=1e-3
)
RECON_KWARGS = dict(batch_size=64, num_samples_per_ray=9, num_workers=0, holdout_fraction=0.1)


def _tomo(c2f: bool = False):
    from quantem.tomography.tomography import Tomography

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        extra = dict(c2f_warmup_frac=0.5, c2f_blur_sigma0=1.0, c2f_blur_frac=0.5) if c2f else {}
        model = KPlanesTILTED(
            M_features=2, resolution=(8, 8, 8), multiscale_res_multipliers=[1, 2], T=2, **extra
        )
    obj = ObjectTensorDecomp.from_model(model, shape=(8, 8, 8), device="cuda:0")
    return Tomography.from_models(dset=_dataset(), obj_model=obj, device="cuda:0", verbose=False)


def _spy_fp32_paths(monkeypatch, tomo):
    """Record dtypes reaching ray integration and every recorded soft term."""
    seen = {"integrate": [], "soft": []}
    integrate = tomo.dset.integrate_rays

    def spy_integrate(rays, *args, **kwargs):
        seen["integrate"].append(rays.dtype)
        return integrate(rays, *args, **kwargs)

    record = object_models.record_soft_term

    def spy_record(ctx, name, value):
        seen["soft"].append((name, value.dtype))
        record(ctx, name, value)

    monkeypatch.setattr(tomo.dset, "integrate_rays", spy_integrate)
    monkeypatch.setattr(object_models, "record_soft_term", spy_record)
    return seen


@requires_bf16
@pytest.mark.slow
class TestReconstructPrecision:
    @pytest.mark.parametrize("flag", [True, False])
    def test_reconstruct(self, monkeypatch, flag):
        tomo = _tomo(c2f=True)
        seen = _spy_fp32_paths(monkeypatch, tomo)
        with _LinearDtypes() as rec:
            tomo.reconstruct(
                num_iter=2,
                optimizer_params=_optimizer_params(),
                obj_constraints=ALL_TERMS.copy(),
                use_bfloat16=flag,
                **RECON_KWARGS,
            )
        expected = torch.bfloat16 if flag else torch.float32
        assert rec.dtypes and all(d == expected for d in rec.dtypes)
        assert seen["integrate"] and all(d == torch.float32 for d in seen["integrate"])
        names = {name for name, _ in seen["soft"]}
        assert {"tv_vol", "tv_plane", "sparsity", "s3im", "l21_plane", "distortion"} <= names
        assert all(d == torch.float32 for _, d in seen["soft"])
        assert all(np.isfinite(tomo.epoch_losses))
        assert len(tomo._val_losses) == 2 and all(np.isfinite(tomo._val_losses))
        for p in tomo.dset.parameters():
            assert p.dtype == torch.float32

    def test_two_phase_bf16(self, monkeypatch):
        tomo = _tomo()
        seen = _spy_fp32_paths(monkeypatch, tomo)
        with _LinearDtypes() as rec:
            summary = tomo.reconstruct_two_phase(
                phase1={"num_iter": 1, "optimizer_params": _optimizer_params()},
                num_iter=2,
                optimizer_params=_optimizer_params(),
                obj_constraints=ALL_TERMS.copy(),
                use_bfloat16=True,
                **RECON_KWARGS,
            )
        assert summary["phase1_grad_steps"] > 0 and summary["phase2_grad_steps"] > 0
        assert rec.dtypes and all(d == torch.bfloat16 for d in rec.dtypes)
        assert all(d == torch.float32 for d in seen["integrate"])
        assert all(d == torch.float32 for _, d in seen["soft"])
        assert all(np.isfinite(tomo.epoch_losses))
