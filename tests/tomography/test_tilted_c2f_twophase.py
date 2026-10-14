"""TILTED coarse-to-fine schedules, rotation transfer and two-phase optimisation."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.core.ml.models.kplanes import (
    CPTilted,
    KPlanesTILTED,
    c2f_blur_sigma,
    c2f_scale_gates,
    gaussian_blur_planes,
)
from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.object_models import ObjConstraintParams, ObjectTensorDecomp
from quantem.tomography.tomography_context import ReconstructionContext

from .conftest import requires_gpu

SO3_TYPES = ["quat", "r9svd"]


def _tilted(seed: int = 0, **kwargs) -> KPlanesTILTED:
    params = dict(M_features=2, resolution=(8, 8, 8), multiscale_res_multipliers=[1, 2], T=2)
    params.update(kwargs)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return KPlanesTILTED(**params)


def _points(n: int = 256, seed: int = 1) -> torch.Tensor:
    return torch.rand(n, 3, generator=torch.Generator().manual_seed(seed)) * 2 - 1


class TestScaleGates:
    def test_schedule_values(self):
        assert c2f_scale_gates(0.0, 0.5, 3) == [1.0, 0.0, 0.0]
        np.testing.assert_allclose(c2f_scale_gates(0.125, 0.5, 3), [1.0, 0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(c2f_scale_gates(0.25, 0.5, 3), [1.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(c2f_scale_gates(0.375, 0.5, 3), [1.0, 1.0, 0.5], atol=1e-12)
        assert c2f_scale_gates(0.5, 0.5, 3) == [1.0, 1.0, 1.0]
        assert c2f_scale_gates(0.9, 0.5, 3) == [1.0, 1.0, 1.0]

    def test_cosine_ramp_shape(self):
        gate = c2f_scale_gates(0.1, 1.0, 2)[1]
        assert gate == pytest.approx(0.5 * (1 - math.cos(math.pi * 0.1)))

    def test_disabled_and_single_level_are_noops(self):
        assert c2f_scale_gates(0.0, 0.0, 3) == [1.0, 1.0, 1.0]
        assert c2f_scale_gates(0.0, 0.5, 1) == [1.0]
        model = _tilted(multiscale_res_multipliers=[1], c2f_warmup_frac=0.5)
        assert model._active_scale_gates() is None

    def test_model_gates_follow_progress_and_coarsest_trains_at_start(self):
        model = _tilted(c2f_warmup_frac=0.5)
        assert model.scale_gates == [1.0, 0.0]
        model(_points()).sum().backward()
        assert model.grids[0].grad.abs().sum() > 0
        assert model.grids[1].grad.abs().sum() == 0
        model.set_progress(0.25)
        assert model.scale_gates == pytest.approx([1.0, 0.5])
        model.set_progress(1.0)
        assert model._active_scale_gates() is None

    def test_fully_open_gates_match_default_model(self):
        gated = _tilted(c2f_warmup_frac=0.5)
        gated.set_progress(0.5)
        pts = _points()
        assert torch.equal(gated(pts), _tilted()(pts))


class TestPlaneBlur:
    def test_sigma_schedule(self):
        assert c2f_blur_sigma(0.0, 2.0, 0.5) == 2.0
        assert c2f_blur_sigma(0.25, 2.0, 0.5) == pytest.approx(1.0)
        assert c2f_blur_sigma(0.5, 2.0, 0.5) == 0.0
        assert c2f_blur_sigma(0.3, 0.0, 0.5) == 0.0

    def test_invalid_configuration_rejected(self):
        with pytest.raises(ValueError):
            _tilted(c2f_blur_sigma0=1.0)
        with pytest.raises(ValueError):
            _tilted(c2f_blur_frac=0.5)
        with pytest.raises(ValueError):
            _tilted(c2f_warmup_frac=-0.1)

    def test_sigma_zero_is_bit_identical(self):
        blurred = _tilted(c2f_blur_sigma0=2.0, c2f_blur_frac=0.5)
        blurred.set_progress(0.5)
        assert blurred.blur_sigma == 0.0
        assert blurred._active_grids() is blurred.grids
        pts = _points()
        assert torch.equal(blurred(pts), _tilted()(pts))

    def test_blur_active_changes_densities(self):
        blurred = _tilted(c2f_blur_sigma0=2.0, c2f_blur_frac=0.5)
        assert blurred.blur_sigma == 2.0
        pts = _points()
        assert not torch.equal(blurred(pts), _tilted()(pts))

    def test_blur_reduces_high_frequency_energy_and_is_differentiable(self):
        grid = torch.rand(6, 2, 32, 32, generator=torch.Generator().manual_seed(3))
        grid.requires_grad_(True)
        out = gaussian_blur_planes(grid, 1.5)
        assert out.shape == grid.shape

        def high_freq_energy(x):
            spec = torch.fft.fftshift(torch.fft.fft2(x - x.mean(dim=(-2, -1), keepdim=True)))
            yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
            mask = ((yy - 16) ** 2 + (xx - 16) ** 2) > 8**2
            return float(spec.abs().pow(2)[..., mask].sum())

        assert high_freq_energy(out.detach()) < 0.05 * high_freq_energy(grid.detach())
        torch.testing.assert_close(
            out.detach().mean(dim=(-2, -1)), grid.detach().mean(dim=(-2, -1)), rtol=0, atol=2e-2
        )
        out.square().sum().backward()
        assert grid.grad is not None and grid.grad.abs().sum() > 0

    def test_blur_zero_sigma_returns_input(self):
        grid = torch.rand(3, 2, 8, 8)
        assert gaussian_blur_planes(grid, 0.0) is grid

    def test_per_level_sigma_scales_with_resolution(self, monkeypatch):
        from quantem.core.ml.models import kplanes

        seen = []

        def spy(grid, sigma):
            seen.append(sigma)
            return grid

        monkeypatch.setattr(kplanes, "gaussian_blur_planes", spy)
        model = _tilted(c2f_blur_sigma0=2.0, c2f_blur_frac=0.5)
        model._active_grids()
        assert seen == [1.0, 2.0]


def _rotation_model(kind: str, so3: str, seed: int, T: int = 3):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "cp":
            return CPTilted(C=2, resolution=(8, 8, 8), T=T, so3_param_type=so3)
        return KPlanesTILTED(M_features=2, resolution=(8, 8, 8), T=T, so3_param_type=so3)


class TestTauTransfer:
    @pytest.mark.parametrize("src_so3", SO3_TYPES)
    @pytest.mark.parametrize("dst_so3", SO3_TYPES)
    @pytest.mark.parametrize(("src_kind", "dst_kind"), [("cp", "kplanes"), ("kplanes", "cp")])
    def test_round_trip(self, src_so3, dst_so3, src_kind, dst_kind):
        src = _rotation_model(src_kind, src_so3, seed=0)
        dst = _rotation_model(dst_kind, dst_so3, seed=1)
        R = src.extract_tau_state()
        assert R.shape == (3, 3, 3) and R.device.type == "cpu"
        torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(3).expand(3, 3, 3), rtol=0, atol=1e-5)
        torch.testing.assert_close(torch.linalg.det(R), torch.ones(3), rtol=0, atol=1e-5)
        assert not torch.allclose(dst.extract_tau_state(), R, atol=1e-3)
        dst.load_tau_state(R)
        torch.testing.assert_close(dst.extract_tau_state(), R, rtol=0, atol=1e-5)
        torch.testing.assert_close(dst.so3.as_matrix().detach(), R, rtol=0, atol=1e-5)

    @pytest.mark.parametrize("dst_so3", SO3_TYPES)
    def test_raw_r9_matrix_is_projected(self, dst_so3):
        src = _rotation_model("cp", "r9svd", seed=0)
        dst = _rotation_model("kplanes", dst_so3, seed=1)
        dst.load_tau_state(src.so3.M.detach())
        torch.testing.assert_close(
            dst.extract_tau_state(), src.extract_tau_state(), rtol=0, atol=1e-5
        )

    @pytest.mark.parametrize("src_so3", SO3_TYPES)
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=requires_gpu)])
    def test_r9svd_load_has_finite_gradients(self, src_so3, device):
        src = _rotation_model("cp", src_so3, seed=0)
        dst = _rotation_model("kplanes", "r9svd", seed=1).to(device)
        R = src.extract_tau_state()
        dst.load_tau_state(R)
        torch.testing.assert_close(dst.extract_tau_state(), R, rtol=0, atol=1e-5)
        s = torch.linalg.svdvals(dst.so3.M.detach().double())
        assert (s[:, :-1] - s[:, 1:]).abs().min() > 0.05
        weights = torch.randn(3, 3, 3, generator=torch.Generator().manual_seed(2)).to(device)
        (dst.so3.as_matrix() * weights).sum().backward()
        assert torch.isfinite(dst.so3.M.grad).all()

    @pytest.mark.parametrize("so3", SO3_TYPES)
    def test_T_mismatch_rejected(self, so3):
        src = _rotation_model("cp", so3, seed=0, T=2)
        dst = _rotation_model("kplanes", so3, seed=1, T=3)
        with pytest.raises(ValueError):
            dst.load_tau_state(src.extract_tau_state())


def _box_dataset(n: int = 8, nang: int = 5, seed: int = 0) -> TomographyINRDataset:
    rng = np.random.default_rng(seed)
    stack = (rng.random((nang, n, n)) * 5.0 + 0.5).astype(np.float32)
    return TomographyINRDataset.from_data(stack, np.linspace(-60, 60, nang, dtype="f4"))


def _optimizer_params(pose: bool = False) -> dict:
    from quantem.core.ml.optimizer_mixin import OptimizerParams

    params = {
        "object": {
            "grids": OptimizerParams.Adam(lr=1e-2),
            "sigma_net": OptimizerParams.Adam(lr=1e-3),
            "so3": OptimizerParams.Adam(lr=1e-2),
        }
    }
    if pose:
        params["pose"] = OptimizerParams.Adam(lr=1e-3)
    return params


def _tomo(device: str, n: int = 8, **model_kwargs):
    from quantem.tomography.tomography import Tomography

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        params = dict(M_features=2, resolution=(n, n, n), T=2)
        params.update(model_kwargs)
        model = KPlanesTILTED(**params)
    obj = ObjectTensorDecomp.from_model(model, shape=(n, n, n), device=device)
    return Tomography.from_models(dset=_box_dataset(n=n), obj_model=obj, device=device, verbose=False)


RECON_KWARGS = dict(batch_size=64, num_samples_per_ray=9, num_workers=0)


class _RecordingLogger:
    """Stand-in for LoggerTomography recording the epoch and gradient-step axes."""

    log_images_every = 0

    def __init__(self):
        self.iters, self.grad_steps, self.snapshot_steps, self.phases = [], [], [], []

    def log_iter(self, *, iter, grad_step=None, **kwargs):
        self.iters.append(iter)
        self.grad_steps.append(grad_step)

    def log_scalar(self, tag, value, step, **kwargs):
        if tag == "snapshots/last_grad_step":
            self.snapshot_steps.append(step)
        elif tag == "two_phase/phase":
            self.phases.append(value)

    def flush(self):
        pass


@requires_gpu
@pytest.mark.slow
class TestTrainingLoopProgress:
    def test_set_progress_driven_per_gradient_step(self, monkeypatch, torch_device):
        tomo = _tomo(torch_device, multiscale_res_multipliers=[1, 2], c2f_warmup_frac=0.5)
        model = tomo.obj_model.model
        calls = []
        original = model.set_progress

        def spy(frac):
            calls.append(frac)
            original(frac)

        monkeypatch.setattr(model, "set_progress", spy)
        tomo.reconstruct(num_iter=2, optimizer_params=_optimizer_params(), **RECON_KWARGS)
        planned = 2 * len(tomo.dataloader)
        assert calls[0] == 0.0
        assert len(calls) == planned + 1
        np.testing.assert_allclose(calls[1:], np.arange(1, planned + 1) / planned)
        assert model.scale_gates == [1.0, 1.0]
        assert tomo._grad_steps == planned

    def test_gpu_matches_cpu_with_gates_and_blur(self):
        kwargs = dict(c2f_warmup_frac=0.5, c2f_blur_sigma0=1.5, c2f_blur_frac=0.5)
        cpu, gpu = _tilted(**kwargs), _tilted(**kwargs).cuda()
        for m in (cpu, gpu):
            m.set_progress(0.2)
        pts = _points()
        out_cpu, out_gpu = cpu(pts), gpu(pts.cuda())
        torch.testing.assert_close(out_gpu.cpu(), out_cpu, rtol=1e-4, atol=1e-5)
        out_cpu.square().sum().backward()
        out_gpu.square().sum().backward()
        for gc, gg in zip(cpu.grids, gpu.grids):
            torch.testing.assert_close(gg.grad.cpu(), gc.grad, rtol=1e-3, atol=1e-5)


@requires_gpu
@pytest.mark.slow
class TestTwoPhase:
    @pytest.mark.parametrize(("p1_so3", "p2_so3"), [("r9svd", "quat"), ("quat", "r9svd")])
    def test_end_to_end(self, monkeypatch, torch_device, p1_so3, p2_so3):
        tomo = _tomo(torch_device, so3_param_type=p2_so3)
        transferred = {}
        cp_extract = CPTilted.extract_tau_state

        def spy_extract(self):
            transferred["phase1"] = cp_extract(self)
            return transferred["phase1"]

        monkeypatch.setattr(CPTilted, "extract_tau_state", spy_extract)
        phase2_model = tomo.obj_model.model
        kp_load = phase2_model.load_tau_state

        def spy_load(R):
            kp_load(R)
            transferred["loaded"] = phase2_model.extract_tau_state()

        monkeypatch.setattr(phase2_model, "load_tau_state", spy_load)

        snapshots, evals = [], []
        logger = _RecordingLogger()
        tomo._logger = logger
        summary = tomo.reconstruct_two_phase(
            phase1={
                "num_iter": 2,
                "optimizer_params": _optimizer_params(),
                "model_kwargs": {"C": 2, "so3_param_type": p1_so3},
            },
            num_iter=5,
            optimizer_params=_optimizer_params(pose=True),
            obj_constraints=ObjConstraintParams.ObjTensorDecompConstraints(
                tv_plane=1e-3, l21_plane=1e-3
            ),
            holdout_fraction=0.1,
            snapshot_every=1,
            snapshot_callback=lambda step, vol: snapshots.append(step),
            eval_callback=evals.append,
            eval_every=1,
            **RECON_KWARGS,
        )
        steps_per_epoch = len(tomo.dataloader)
        torch.testing.assert_close(transferred["loaded"], transferred["phase1"], rtol=0, atol=1e-5)
        assert summary["phase1_grad_steps"] == 2 * steps_per_epoch
        assert summary["phase2_grad_steps"] == 3 * steps_per_epoch
        assert summary["total_grad_steps"] == tomo._grad_steps == 5 * steps_per_epoch
        assert summary["phase1_epochs"] == 2 and summary["phase2_epochs"] == 3
        assert snapshots == list(range(1, 5 * steps_per_epoch + 1))
        assert evals == [1, 2, 3, 4, 5]
        assert len(tomo.epoch_losses) == 5
        assert len(tomo.obj_model.soft_constraint_losses) == 5
        assert isinstance(tomo.obj_model.model, KPlanesTILTED)
        assert tomo.obj_model.constraints.l21_plane == 1e-3
        assert {"tv_plane", "l21_plane"} <= set(tomo.soft_term_losses[0])
        assert {"tv_plane", "l21_plane"} <= set(tomo.soft_term_losses[-1])
        assert tomo.dset.has_optimizer()
        assert logger.iters == [1, 2, 3, 4, 5]
        assert logger.grad_steps == [k * steps_per_epoch for k in range(1, 6)]
        assert logger.snapshot_steps == snapshots
        assert logger.phases == [1.0, 1.0, 2.0, 2.0, 2.0]

    @pytest.mark.parametrize("phase1_iters", [0, 3, 4])
    def test_rejects_phase1_outside_budget(self, torch_device, phase1_iters):
        tomo = _tomo(torch_device)
        with pytest.raises(ValueError):
            tomo.reconstruct_two_phase(
                phase1={"num_iter": phase1_iters, "optimizer_params": _optimizer_params()},
                num_iter=3,
                optimizer_params=_optimizer_params(),
                **RECON_KWARGS,
            )
        assert tomo._grad_steps == 0

    def test_rejects_T_mismatch(self, torch_device):
        tomo = _tomo(torch_device)
        with pytest.raises(ValueError):
            tomo.reconstruct_two_phase(
                phase1={
                    "num_iter": 1,
                    "optimizer_params": _optimizer_params(),
                    "model_kwargs": {"T": 3},
                },
                num_iter=2,
                optimizer_params=_optimizer_params(),
                **RECON_KWARGS,
            )


class TestPhase1Constraints:
    def _host(self, **weights):
        constraints = ObjConstraintParams.ObjTensorDecompConstraints(**weights)
        return SimpleNamespace(obj_model=SimpleNamespace(constraints=constraints))

    def test_default_copies_every_phase2_soft_weight(self):
        from quantem.tomography.tomography import Tomography

        weights = dict(tv_plane=1e-3, l21_plane=2e-3, tv_vol=1e-4, sparsity=1e-5, s3im_weight=0.5)
        host = self._host(**weights)
        got = Tomography._phase1_constraints(host, None)
        assert got is not host.obj_model.constraints
        for key, value in weights.items():
            assert getattr(got, key) == value

    def test_explicit_plane_terms_are_accepted(self):
        from quantem.tomography.tomography import Tomography

        got = Tomography._phase1_constraints(
            self._host(), {"name": "obj_tensor_decomp", "tv_plane": 5e-3, "l21_plane": 1e-3}
        )
        assert got.tv_plane == 5e-3 and got.l21_plane == 1e-3

    def test_cptilted_soft_terms_recorded(self, torch_device):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = CPTilted(C=2, resolution=(8, 8, 8), T=2)
        obj = ObjectTensorDecomp.from_model(model, shape=(8, 8, 8), device=torch_device)
        obj.constraints = ObjConstraintParams.ObjTensorDecompConstraints(
            tv_plane=1e-3, l21_plane=1e-3
        )
        coords = _points(64).to(torch_device)
        densities = obj.forward(coords)
        terms: dict[str, torch.Tensor] = {}
        loss = obj.apply_soft_constraints(
            ReconstructionContext(
                coords=coords,
                pred=densities[:8],
                all_densities=densities,
                target=torch.zeros(8, device=torch_device),
                soft_terms=terms,
            )
        )
        assert set(terms) == {"tv_plane", "l21_plane"}
        assert float(terms["tv_plane"]) > 0 and float(terms["l21_plane"]) > 0
        torch.testing.assert_close(loss.detach(), terms["tv_plane"] + terms["l21_plane"])
