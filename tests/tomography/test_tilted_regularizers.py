"""TILTED l2,1 plane sparsity and Mip-NeRF 360 ray distortion regularizers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import default_collate

from quantem.core.ml.models.kplanes import KPlanes, KPlanesTILTED
from quantem.tomography.dataset_models import (
    DatasetConstraintParams,
    TomographyINRDataset,
    padded_ray_distortion,
)
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import ObjConstraintParams, ObjectTensorDecomp
from quantem.tomography.tomography_context import ReconstructionContext

from .conftest import requires_gpu


def _brute_force_distortion(densities: np.ndarray, step: float) -> float:
    weights = densities * step
    positions = np.arange(len(densities)) * step
    pairwise = np.abs(positions[:, None] - positions[None, :])
    inter = float(np.sum(weights[:, None] * weights[None, :] * pairwise))
    intra = float(np.sum(weights**2) * step / 3.0)
    return inter + intra


def _random_ragged(seed: int, num_rays: int = 7, max_samples: int = 13):
    rng = np.random.default_rng(seed)
    counts = rng.integers(2, max_samples + 1, size=num_rays)
    width = int(counts.max())
    densities = np.zeros((num_rays, width))
    for r, count in enumerate(counts):
        densities[r, :count] = rng.random(count) * 3.0
    steps = rng.uniform(0.05, 0.3, size=num_rays)
    mask = np.arange(width)[None, :] < counts[:, None]
    return densities, steps, mask


def _box_dataset(n: int = 8, nang: int = 3, seed: int = 0) -> TomographyINRDataset:
    rng = np.random.default_rng(seed)
    stack = (rng.random((nang, n, n)) * 5.0 + 0.5).astype(np.float32)
    dset = TomographyINRDataset.from_data(stack, np.linspace(-50, 50, nang, dtype="f4"))
    dset.to("cpu")
    return dset


def _fake_meta(densities: np.ndarray, steps: np.ndarray, mask: np.ndarray, valid: np.ndarray):
    mask_t = torch.as_tensor(mask)
    return {
        "valid": torch.as_tensor(valid),
        "local_ray_ids": mask_t.nonzero(as_tuple=False)[:, 0],
        "step_sizes": torch.as_tensor(steps, dtype=torch.float64),
        "num_valid": int(mask.shape[0]),
        "sample_mask": mask_t,
    }


class TestDistortionKernel:
    def test_matches_brute_force_on_ragged_rays(self):
        densities, steps, mask = _random_ragged(seed=1)
        got = padded_ray_distortion(
            torch.as_tensor(densities), torch.as_tensor(steps)
        ).numpy()
        expected = np.array(
            [
                _brute_force_distortion(densities[r, mask[r]], steps[r])
                for r in range(densities.shape[0])
            ]
        )
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

    def test_matches_continuous_integral_of_step_function(self):
        rng = np.random.default_rng(2)
        sigma = rng.random(5) * 2.0
        step = 0.2
        sub = 400
        fine_sigma = np.repeat(sigma, sub)
        fine_positions = (np.arange(len(fine_sigma)) + 0.5) * (step / sub)
        cell = step / sub
        numeric = float(
            np.sum(
                fine_sigma[:, None]
                * fine_sigma[None, :]
                * np.abs(fine_positions[:, None] - fine_positions[None, :])
            )
            * cell**2
        )
        got = float(
            padded_ray_distortion(
                torch.as_tensor(sigma)[None, :], torch.tensor([step], dtype=torch.float64)
            )[0]
        )
        np.testing.assert_allclose(got, numeric, rtol=1e-5)

    def test_gradcheck(self):
        densities, steps, _ = _random_ragged(seed=3, num_rays=4, max_samples=6)
        sigma = torch.as_tensor(densities).requires_grad_(True)
        step = torch.as_tensor(steps)
        assert torch.autograd.gradcheck(lambda x: padded_ray_distortion(x, step), (sigma,))

    def test_two_deltas_grow_with_separation(self):
        values = []
        for separation in range(1, 9):
            sigma = torch.zeros(1, 10, dtype=torch.float64)
            sigma[0, 0] = 1.0
            sigma[0, separation] = 1.0
            values.append(
                float(padded_ray_distortion(sigma, torch.tensor([0.1], dtype=torch.float64))[0])
            )
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_long_float32_ray_matches_float64_reference(self):
        rng = np.random.default_rng(12)
        sigma32 = torch.as_tensor(rng.random((2, 1000)) * 4.0, dtype=torch.float32)
        step32 = torch.tensor([0.002, 0.0015], dtype=torch.float32)
        got = padded_ray_distortion(sigma32, step32)
        assert got.dtype == torch.float32
        expected = np.array(
            [
                _brute_force_distortion(
                    sigma32[r].double().numpy(), float(step32[r].double())
                )
                for r in range(2)
            ]
        )
        np.testing.assert_allclose(got.double().numpy(), expected, rtol=1e-6)


class TestRayDistortionDataset:
    def test_box_mode_ragged_matches_brute_force_and_keeps_metadata(self):
        densities, steps, mask = _random_ragged(seed=4)
        num_rays = densities.shape[0]
        valid = np.ones(num_rays + 2, dtype=bool)
        valid[[1, 5]] = False
        dset = _box_dataset()
        dset._ray_meta = _fake_meta(densities, steps, mask, valid)
        flat = torch.as_tensor(densities[mask])
        target = torch.rand(num_rays + 2, dtype=torch.float64)
        got = dset.ray_distortion(flat, target, norm="none")
        expected = np.mean(
            [_brute_force_distortion(densities[r, mask[r]], steps[r]) for r in range(num_rays)]
        )
        np.testing.assert_allclose(float(got), expected, rtol=1e-10)
        assert dset._ray_meta is not None

    @pytest.mark.parametrize("norm", ["target", "pred"])
    def test_box_mode_normalisations(self, norm):
        densities, steps, mask = _random_ragged(seed=5)
        num_rays = densities.shape[0]
        dset = _box_dataset()
        dset._ray_meta = _fake_meta(densities, steps, mask, np.ones(num_rays, dtype=bool))
        target = np.random.default_rng(6).random(num_rays) + 0.1
        eps_rel = 0.05
        got = dset.ray_distortion(
            torch.as_tensor(densities[mask]), torch.as_tensor(target), norm=norm, eps_rel=eps_rel
        )
        raw = np.array(
            [_brute_force_distortion(densities[r, mask[r]], steps[r]) for r in range(num_rays)]
        )
        eps = eps_rel**2 * np.mean(target**2)
        if norm == "target":
            reference = target**2
        else:
            reference = (densities.sum(axis=1) * steps) ** 2
        np.testing.assert_allclose(float(got), np.mean(raw / (reference + eps)), rtol=1e-10)

    @pytest.mark.parametrize("norm", ["none", "target"])
    def test_box_mode_gradcheck(self, norm):
        densities, steps, mask = _random_ragged(seed=7, num_rays=3, max_samples=5)
        dset = _box_dataset()
        dset._ray_meta = _fake_meta(densities, steps, mask, np.ones(3, dtype=bool))
        flat = torch.as_tensor(densities[mask]).requires_grad_(True)
        target = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
        assert torch.autograd.gradcheck(
            lambda x: dset.ray_distortion(x, target, norm=norm, eps_rel=0.05), (flat,)
        )

    def test_pred_norm_gradient_treats_denominator_as_constant(self):
        densities, steps, mask = _random_ragged(seed=11, num_rays=3, max_samples=5)
        dset = _box_dataset()
        dset._ray_meta = _fake_meta(densities, steps, mask, np.ones(3, dtype=bool))
        flat = torch.as_tensor(densities[mask]).requires_grad_(True)
        target = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
        dset.ray_distortion(flat, target, norm="pred", eps_rel=0.05).backward()

        reference_flat = flat.detach().clone().requires_grad_(True)
        padded = torch.zeros(mask.shape, dtype=torch.float64).masked_scatter(
            torch.as_tensor(mask), reference_flat
        )
        step = torch.as_tensor(steps)
        eps = 0.05**2 * target.pow(2).mean()
        frozen = (padded.detach().sum(dim=1) * step).pow(2) + eps
        (padded_ray_distortion(padded, step) / frozen).mean().backward()
        torch.testing.assert_close(flat.grad, reference_flat.grad, rtol=1e-12, atol=0.0)

    def test_all_zero_rays_give_zero_under_target_norm(self):
        dset = _box_dataset()
        mask = np.ones((3, 4), dtype=bool)
        dset._ray_meta = _fake_meta(np.zeros((3, 4)), np.full(3, 0.1), mask, np.ones(3, dtype=bool))
        flat = torch.zeros(12, dtype=torch.float64, requires_grad=True)
        loss = dset.ray_distortion(flat, torch.zeros(3, dtype=torch.float64), norm="target")
        assert float(loss) == 0.0
        loss.backward()
        assert torch.isfinite(flat.grad).all()

    @pytest.mark.parametrize("norm", ["target", "pred"])
    def test_all_zero_targets_with_nonzero_density_give_graph_carrying_zero(self, norm):
        densities, steps, mask = _random_ragged(seed=13, num_rays=4, max_samples=6)
        dset = _box_dataset()
        dset._ray_meta = _fake_meta(densities, steps, mask, np.ones(4, dtype=bool))
        flat = torch.as_tensor(densities[mask]).requires_grad_(True)
        loss = dset.ray_distortion(flat, torch.zeros(4, dtype=torch.float64), norm=norm)
        assert float(loss) == 0.0
        assert loss.requires_grad
        loss.backward()
        assert torch.isfinite(flat.grad).all()
        assert float(flat.grad.abs().max()) == 0.0

    @pytest.mark.parametrize("norm", ["target", "pred"])
    def test_zero_eps_rel_rejected(self, norm):
        dset = _box_dataset()
        with pytest.raises(ValueError):
            dset.ray_distortion(torch.ones(4), torch.ones(2), norm=norm, eps_rel=0.0)

    def test_zero_ray_contributes_zero_in_mixed_batch(self):
        sigma = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]], dtype=torch.float64)
        per_ray = padded_ray_distortion(sigma, torch.tensor([0.1, 0.1], dtype=torch.float64))
        assert float(per_ray[0]) == 0.0
        assert float(per_ray[1]) > 0.0

    def test_no_valid_ray_returns_graph_carrying_zero(self):
        dset = _box_dataset()
        dset._ray_meta = {
            "valid": torch.zeros(4, dtype=torch.bool),
            "local_ray_ids": torch.zeros(0, dtype=torch.long),
            "step_sizes": torch.zeros(0),
            "num_valid": 0,
        }
        flat = torch.zeros(0, dtype=torch.float64, requires_grad=True)
        loss = dset.ray_distortion(flat, torch.ones(4, dtype=torch.float64))
        assert float(loss) == 0.0
        assert loss.requires_grad and loss.dtype == torch.float64
        loss.backward()

    def test_requires_metadata_in_box_mode(self):
        dset = _box_dataset()
        dset._ray_meta = None
        with pytest.raises(RuntimeError):
            dset.ray_distortion(torch.ones(4), torch.ones(2))

    def test_rejects_unknown_norm(self):
        dset = _box_dataset()
        with pytest.raises(ValueError):
            dset.ray_distortion(torch.ones(4), torch.ones(2), norm="bogus")

    def test_real_get_coords_then_integrate(self):
        dset = _box_dataset(n=8)
        batch = default_collate([dset[i] for i in range(0, 3 * 64, 5)])
        coords = dset.get_coords(batch, 8, 9).detach()
        densities = (
            1.0
            + 0.5 * torch.sin(3.0 * coords[:, 0].double()) * torch.cos(2.0 * coords[:, 1].double())
            + coords[:, 2].double() ** 2
        )
        meta = dset._ray_meta
        valid = meta["valid"]
        valid_ids = meta["local_ray_ids"]
        steps = meta["step_sizes"].double()
        num_valid = int(meta["num_valid"])
        target = batch["target_value"].double()
        raw = np.array(
            [
                _brute_force_distortion(densities[valid_ids == r].numpy(), float(steps[r]))
                for r in range(num_valid)
            ]
        )
        weight_sums = np.array(
            [float(densities[valid_ids == r].sum() * steps[r]) for r in range(num_valid)]
        )
        got = dset.ray_distortion(densities, target, norm="none")
        got_pred_norm = dset.ray_distortion(densities, target, norm="pred", eps_rel=0.05)
        np.testing.assert_allclose(float(got), raw.mean(), rtol=1e-10)
        assert dset._ray_meta is meta

        pred = dset.integrate_rays(densities, 9, len(target))
        assert pred.shape == (len(target),)
        assert dset._ray_meta is None
        integrated = pred[valid].detach().numpy()
        np.testing.assert_allclose(weight_sums, integrated, rtol=1e-12)
        eps = 0.05**2 * float(target[valid].pow(2).mean())
        np.testing.assert_allclose(
            float(got_pred_norm), np.mean(raw / (integrated**2 + eps)), rtol=1e-10
        )

    def test_legacy_fixed_n_path(self):
        rng = np.random.default_rng(8)
        stack = (rng.random((3, 6, 6)) + 0.5).astype(np.float32)
        dset = TomographyINRDataset.from_data(
            stack, np.linspace(-50, 50, 3, dtype="f4"), ray_sampling="legacy"
        )
        num_rays, num_samples = 5, 7
        densities = rng.random((num_rays, num_samples)) * 2.0
        step = 2.0 / (num_samples - 1)
        got = dset.ray_distortion(
            torch.as_tensor(densities.reshape(-1)),
            torch.ones(num_rays, dtype=torch.float64),
            norm="none",
        )
        expected = np.mean([_brute_force_distortion(densities[r], step) for r in range(num_rays)])
        np.testing.assert_allclose(float(got), expected, rtol=1e-10)


def _l12_cost_numpy(grids: list[np.ndarray]) -> float:
    per_group = []
    for grid in grids:
        flat = grid.reshape(grid.shape[0], -1)
        per_group.append(np.sqrt(np.mean(flat**2, axis=1) + 1e-12))
    return float(np.mean(np.concatenate(per_group)))


class _GridModule(nn.Module):
    def __init__(self, grids: list[torch.Tensor]):
        super().__init__()
        self.grids = nn.ParameterList([nn.Parameter(g) for g in grids])




def _seeded_plane_model(kind: str) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if kind == "kplanes_tilted":
            return KPlanesTILTED(
                M_features=2, resolution=(8, 8, 8), multiscale_res_multipliers=[1, 2], T=2
            )
        return KPlanes(M_features=2, resolution=(8, 8, 8), multiscale_res_multipliers=[1, 2])


class TestPlaneL21:
    def test_matches_numpy_l12_cost_on_random_grids(self):
        rng = np.random.default_rng(9)
        grids = [rng.standard_normal((6, 3, 5, 5)), rng.standard_normal((6, 3, 10, 10))]
        fake = SimpleNamespace(
            model=_GridModule([torch.as_tensor(g) for g in grids]),
            constraints=SimpleNamespace(l21_plane=0.7),
        )
        got = float(ObjectTensorDecomp.get_plane_l21_loss(fake))
        np.testing.assert_allclose(got, 0.7 * _l12_cost_numpy(grids), rtol=1e-12)

    @pytest.mark.parametrize("kind", ["kplanes_tilted", "kplanes"])
    def test_real_kplanes_grids(self, kind):
        model = _seeded_plane_model(kind)
        fake = SimpleNamespace(model=model, constraints=SimpleNamespace(l21_plane=1.0))
        got = ObjectTensorDecomp.get_plane_l21_loss(fake)
        expected = _l12_cost_numpy([g.detach().double().numpy() for g in model.grids])
        np.testing.assert_allclose(float(got), expected, rtol=1e-6)
        got.backward()
        assert all(g.grad is not None for g in model.grids)

    def test_line_grids_not_supported(self):
        fake = SimpleNamespace(
            model=_GridModule([torch.rand(6, 3, 8)]),
            constraints=SimpleNamespace(l21_plane=1.0),
        )
        with pytest.raises(NotImplementedError):
            ObjectTensorDecomp.get_plane_l21_loss(fake)


def _tensor_decomp_obj(device: str, n: int = 8) -> ObjectTensorDecomp:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = KPlanesTILTED(
            M_features=2, resolution=(n, n, n), multiscale_res_multipliers=[1], T=2
        )
    return ObjectTensorDecomp.from_model(model, shape=(n, n, n), device=device)


class TestSoftConstraintParity:
    def test_new_fields_default_off(self):
        for cls in (
            ObjConstraintParams.ObjINRConstraints,
            ObjConstraintParams.ObjTensorDecompConstraints,
        ):
            c = cls()
            assert c.distortion == 0.0
            assert c.distortion_norm == "target"
            assert c.distortion_eps_rel == 0.05
            assert "distortion" in c.soft_constraint_keys
        tensor_decomp = ObjConstraintParams.ObjTensorDecompConstraints()
        assert tensor_decomp.l21_plane == 0.0
        assert "l21_plane" in tensor_decomp.soft_constraint_keys
        assert tensor_decomp.sparsity == 0.0

    @pytest.mark.parametrize(
        "cls",
        [ObjConstraintParams.ObjINRConstraints, ObjConstraintParams.ObjTensorDecompConstraints],
    )
    def test_dataclass_validates_distortion_params(self, cls):
        with pytest.raises(ValueError):
            cls(distortion_norm="bogus")
        with pytest.raises(ValueError):
            cls(distortion=0.1, distortion_eps_rel=0.0)
        assert cls(distortion_norm="none", distortion_eps_rel=0.0).distortion_eps_rel == 0.0

    def test_zero_new_weights_reproduce_legacy_total_bit_exactly(self, torch_device):
        obj = _tensor_decomp_obj(torch_device)
        obj.constraints.tv_plane = 1.0e-3
        obj.constraints.sparsity = 1.0e-4
        device = torch.device(torch_device)
        generator = torch.Generator().manual_seed(10)
        coords = (torch.rand(64, 3, generator=generator) * 2 - 1).to(device)
        pred = torch.rand(16, generator=generator).to(device)
        densities = torch.rand(64, generator=generator).to(device)

        legacy = torch.tensor(0.0, device=device)
        legacy += torch.tensor(0.0, device=device) + obj._get_plane_tv_loss()
        legacy += 1.0e-4 * densities.abs().mean()

        plain = obj.apply_soft_constraints(
            ReconstructionContext(coords=coords, pred=pred, all_densities=densities)
        )
        terms: dict[str, torch.Tensor] = {}
        logged = obj.apply_soft_constraints(
            ReconstructionContext(
                coords=coords, pred=pred, all_densities=densities, soft_terms=terms
            )
        )
        assert torch.equal(plain, legacy)
        assert torch.equal(logged, legacy)
        assert set(terms) == {"tv_plane", "sparsity"}
        assert torch.equal(terms["tv_plane"] + terms["sparsity"], legacy)

    def test_new_terms_added_and_recorded(self, torch_device):
        obj = _tensor_decomp_obj(torch_device)
        obj.constraints.sparsity = 0.0
        obj.constraints.l21_plane = 0.5
        obj.constraints.distortion = 2.0
        device = torch.device(torch_device)
        terms: dict[str, torch.Tensor] = {}
        total = obj.apply_soft_constraints(
            ReconstructionContext(
                coords=torch.zeros(4, 3, device=device),
                pred=torch.zeros(2, device=device),
                all_densities=torch.zeros(4, device=device),
                distortion=torch.tensor(0.25, device=device),
                soft_terms=terms,
            )
        )
        assert set(terms) == {"l21_plane", "distortion"}
        assert float(terms["distortion"]) == 0.5
        assert torch.allclose(total, terms["l21_plane"] + terms["distortion"])

    def test_distortion_without_context_value_raises(self, torch_device):
        obj = _tensor_decomp_obj(torch_device)
        obj.constraints.sparsity = 0.0
        obj.constraints.distortion = 1.0
        with pytest.raises(ValueError):
            obj.apply_soft_constraints(
                ReconstructionContext(pred=torch.zeros(2, device=torch_device))
            )


@requires_gpu
@pytest.mark.slow
class TestTrainingLoop:
    @pytest.mark.parametrize("distortion", [0.0, 0.1])
    def test_ray_distortion_called_only_when_enabled(self, monkeypatch, torch_device, distortion):
        from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams
        from quantem.tomography.tomography import Tomography

        n = 8
        obj = _tensor_decomp_obj(torch_device, n)
        dset = _box_dataset(n=n)
        tomo = Tomography.from_models(
            dset=dset, obj_model=obj, device=torch_device, verbose=False
        )
        calls = []
        original = tomo.dset.ray_distortion

        def spy(*args, **kwargs):
            calls.append(kwargs.get("norm"))
            return original(*args, **kwargs)

        monkeypatch.setattr(tomo.dset, "ray_distortion", spy)
        pose_tv = 1.0e-3 if distortion > 0 else 0.0
        tomo.reconstruct(
            optimizer_params={
                "object": {
                    "grids": OptimizerParams.Adam(lr=1e-2),
                    "sigma_net": OptimizerParams.Adam(lr=1e-3),
                    "so3": OptimizerParams.Adam(lr=1e-2),
                },
                "pose": OptimizerParams.Adam(lr=1e-3),
            },
            scheduler_params={
                "object": SchedulerParams.CosineAnnealing(T_max=1),
                "pose": SchedulerParams.CosineAnnealing(T_max=1),
            },
            obj_constraints=ObjConstraintParams.ObjTensorDecompConstraints(
                tv_plane=1.0e-3, sparsity=1.0e-4, distortion=distortion
            ),
            dset_constraints=DatasetConstraintParams.BaseTomographyDatasetConstraints(
                tv_shifts=pose_tv
            ),
            num_iter=1,
            batch_size=64,
            num_samples_per_ray=9,
            num_workers=0,
        )
        num_batches = len(tomo.dataloader)
        epoch_terms = tomo.soft_term_losses[-1]
        if distortion == 0.0:
            assert calls == []
            assert set(epoch_terms) == {"tv_plane", "sparsity"}
        else:
            assert calls == ["target"] * num_batches
            assert set(epoch_terms) == {"tv_plane", "sparsity", "distortion", "pose_tv"}
        assert all(np.isfinite(v) for v in epoch_terms.values())
        np.testing.assert_allclose(
            sum(epoch_terms.values()), tomo.obj_model.soft_constraint_losses[-1], rtol=1e-5
        )


def test_logger_writes_per_term_scalars():
    tags = []
    fake = SimpleNamespace(log_scalar=lambda tag, value, step, extra_steps=None: tags.append(tag))
    LoggerTomography.log_iter(
        fake,
        object_model=SimpleNamespace(_soft_constraint_losses=[0.3]),
        iter=1,
        consistency_loss=0.5,
        total_loss=0.8,
        learning_rates={},
        num_samples_per_ray=8,
        soft_terms={"tv_plane": 0.1, "distortion": 0.2},
    )
    assert "loss/soft" in tags
    assert "loss/soft/tv_plane" in tags
    assert "loss/soft/distortion" in tags
