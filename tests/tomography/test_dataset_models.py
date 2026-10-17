"""Tests for ``quantem.tomography.dataset_models``.

Covers constraint parsing, the pixelated dataset (validation, normalisation, tilt-angle
convention, pose-parameter materialisation) and the INR / pretrain datasets.
"""

import numpy as np
import pytest
import torch

from quantem.core.ml.optimizer_mixin import OptimizerParams
from quantem.tomography.dataset_models import (
    DatasetConstraintParams,
    DatasetValue,
    TomographyINRDataset,
    TomographyINRPretrainDataset,
    TomographyPixDataset,
)
from quantem.tomography.tomography_opt import _pose_optimizer_specs

from .conftest import requires_torch


class TestDatasetConstraintParse:
    def test_parse_base_by_name(self):
        c = DatasetConstraintParams.parse_dict({"name": "base_tomography_dataset", "tv_zs": 0.1})
        assert isinstance(c, DatasetConstraintParams.BaseTomographyDatasetConstraints)
        assert c.tv_zs == 0.1

    def test_parse_base_by_type_key(self):
        c = DatasetConstraintParams.parse_dict(
            {"type": "base_tomography_dataset", "tv_shifts": 0.2}
        )
        assert c.tv_shifts == 0.2

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            DatasetConstraintParams.parse_dict({"name": "nope"})


def _stack(nang=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((nang, n, n)) * 10).astype(np.float32)


class TestTomographyPixDataset:
    def test_wrong_projection_axis_raises(self):
        # projections must live on axis 0, matching the number of tilt angles.
        # here the projections are on the last axis, so axis 0 (12) != n_angles (5).
        bad = np.zeros((12, 12, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            TomographyPixDataset.from_data(bad, np.linspace(-60, 60, 5).astype(np.float32))

    def test_tilt_angles_are_negated(self):
        angles = np.linspace(-40, 60, 5).astype(np.float32)
        d = TomographyPixDataset.from_data(_stack(), angles)
        np.testing.assert_allclose(d.tilt_angles.numpy(), -angles, atol=1e-5)

    def test_normalised_by_95th_quantile(self):
        d = TomographyPixDataset.from_data(_stack(), np.linspace(-60, 60, 5).astype(np.float32))
        q95 = torch.quantile(d.tilt_stack, 0.95)
        assert torch.isclose(q95, torch.tensor(1.0), atol=1e-4)

    def test_sparse_stack_normalisation_stays_finite(self):
        # >95% zeros makes the 95th quantile 0; normalisation must not divide
        # by it (inf/NaN targets poison every parameter on the first backward).
        stack = np.zeros((5, 12, 12), dtype=np.float32)
        stack[:, 5:7, 5:7] = 10.0
        d = TomographyPixDataset.from_data(stack, np.linspace(-60, 60, 5).astype(np.float32))
        assert torch.isfinite(d.tilt_stack).all()
        assert d.tilt_stack.max() > 0

    def test_all_zero_stack_raises(self):
        stack = np.zeros((5, 12, 12), dtype=np.float32)
        with pytest.raises(ValueError):
            TomographyPixDataset.from_data(stack, np.linspace(-60, 60, 5).astype(np.float32))

    def test_reference_idx_and_learnable_tilts(self):
        # negated angles -> [40, 15, -10, -35, -60]; smallest |angle| is index 2.
        angles = np.linspace(-40, 60, 5).astype(np.float32)
        d = TomographyPixDataset.from_data(_stack(), angles)
        assert d.reference_tilt_idx == 2
        assert d.learnable_tilts == 4

    def test_forward_returns_dataset_value(self):
        angles = np.linspace(-40, 60, 5).astype(np.float32)
        d = TomographyPixDataset.from_data(_stack(nang=5, n=12), angles)
        out = d.forward(0)
        assert isinstance(out, DatasetValue)
        assert out.target.shape == (12, 12)
        assert out.tilt_angle == pytest.approx(float(-angles[0]))

    def test_to_materialises_pose_parameters(self):
        d = TomographyPixDataset.from_data(_stack(), np.linspace(-60, 60, 5).astype(np.float32))
        d.to("cpu")
        assert isinstance(d.z1_params, torch.nn.Parameter)
        assert d.shifts_params.shape == (d.learnable_tilts, 2)


@requires_torch
class TestTomographyINRDataset:
    def test_len_is_projections_times_pixels(self):
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12), np.linspace(-60, 60, 5, dtype="f4")
        )
        assert len(d) == 5 * 12 * 12

    def test_getitem_keys(self):
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12), np.linspace(-60, 60, 5, dtype="f4")
        )
        item = d[0]
        assert {"phi", "pixel_i", "pixel_j", "projection_idx", "target_value"} <= set(item.keys())

    @pytest.mark.parametrize(
        "learn_shift,learn_tilt_axis,shift_lr,tilt_axis_lr,expected_lrs",
        [
            (True, True, 2e-2, 3e-3, {"pose_shift": 2e-2, "pose_tilt_axis": 3e-3}),
            (True, True, None, None, {"pose_shift": 1e-2, "pose_tilt_axis": 1e-2}),
            (True, False, 2e-2, 3e-3, {"pose_shift": 2e-2}),
            (False, True, 2e-2, 3e-3, {"pose_tilt_axis": 3e-3}),
        ],
    )
    def test_pose_optimizer_groups_and_lrs(
        self, learn_shift, learn_tilt_axis, shift_lr, tilt_axis_lr, expected_lrs
    ):
        angles = np.linspace(-60, 60, 5, dtype="f4")
        d = TomographyINRDataset.from_data(
            _stack(nang=len(angles), n=12),
            angles,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
        )
        d.to("cpu")
        specs = _pose_optimizer_specs(
            d, pose_lr=1e-2, shift_lr=shift_lr, tilt_axis_lr=tilt_axis_lr
        )
        d.set_optimizer(specs)

        groups = {group["name"]: group for group in d.optimizer.param_groups}
        assert set(groups) == set(expected_lrs)
        for name, lr in expected_lrs.items():
            assert groups[name]["lr"] == pytest.approx(lr)

        if learn_shift:
            shift_params = groups["pose_shift"]["params"]
            assert len(shift_params) == 1 and shift_params[0] is d._shifts_params
            assert d._shifts_params.shape == (len(angles) - 1, 2)
        if learn_tilt_axis:
            tilt_params = groups["pose_tilt_axis"]["params"]
            assert len(tilt_params) == 2
            assert tilt_params[0] is d._z1_params and tilt_params[1] is d._z3_params
            assert d._z1_params.shape == d._z3_params.shape == (len(angles) - 1,)

        optimized = [param for group in groups.values() for param in group["params"]]
        assert all(
            ref is not param
            for ref in (d._shifts_ref, d._z1_ref, d._z3_ref)
            for param in optimized
        )

    def test_legacy_shared_pose_optimizer_expands_to_active_groups(self):
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12), np.linspace(-60, 60, 5, dtype="f4")
        )
        d.to("cpu")
        d.set_optimizer(OptimizerParams.Adam(lr=1e-2))
        assert {group["name"]: group["lr"] for group in d.optimizer.param_groups} == {
            "pose_shift": 1e-2,
            "pose_tilt_axis": 1e-2,
        }

    def test_full_tomography_pose_optimizer_groups_and_lrs(self):
        from quantem.core.ml.inr import HSiren
        from quantem.tomography.object_models import ObjectINR
        from quantem.tomography.tomography import Tomography

        model = HSiren(in_features=3, out_features=1, hidden_layers=1, hidden_features=8)
        obj = ObjectINR.from_model(model, shape=(12, 12, 12), device="cpu")
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12),
            np.linspace(-60, 60, 5, dtype="f4"),
            learn_shift=True,
            learn_tilt_axis=True,
        )
        tomo = Tomography.from_models(dset=d, obj_model=obj, device="cpu", verbose=False)
        tomo.optimizer_params = {
            "pose": {
                "pose_shift": {"name": "adam", "lr": 2e-2},
                "pose_tilt_axis": {"name": "adam", "lr": 3e-3},
            }
        }
        tomo.set_optimizers()

        groups = {group["name"]: group for group in tomo.dset.optimizer.param_groups}
        assert {name: group["lr"] for name, group in groups.items()} == {
            "pose_shift": 2e-2,
            "pose_tilt_axis": 3e-3,
        }
        optimized = [param for group in groups.values() for param in group["params"]]
        assert all(
            ref is not param
            for ref in (tomo.dset._shifts_ref, tomo.dset._z1_ref, tomo.dset._z3_ref)
            for param in optimized
        )

    @pytest.mark.parametrize(
        "learn_shift,learn_tilt_axis",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_forward_gates_shift_and_tilt(self, learn_shift, learn_tilt_axis):
        """``forward`` zeros the disabled component and passes the enabled one through.

        Guards the gating after removing the unreachable duplicate branch: shifts are
        controlled by ``learn_shift``; the z1/z3 Euler angles by ``learn_tilt_axis``.
        """
        d = TomographyINRDataset.from_data(
            _stack(nang=5, n=12),
            np.linspace(-60, 60, 5, dtype="f4"),
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
        )
        d.to("cpu")
        # Make every pose parameter non-zero so the gating is observable by value.
        for p in (d.z1_params, d.z3_params, d.shifts_params):
            p.data.fill_(1.0)
        d._z1_ref = torch.ones_like(d._z1_ref)
        d._z3_ref = torch.ones_like(d._z3_ref)
        d._shifts_ref = torch.ones_like(d._shifts_ref)

        shifts, z1, z3 = d.forward(None)
        assert bool(shifts.any()) == learn_shift
        assert bool(z1.any()) == learn_tilt_axis
        assert bool(z3.any()) == learn_tilt_axis


def _write_pose_init(path, n, z1=None, z3=None, tilt=None, shifts=None):
    zeros = np.zeros(n, dtype=np.float32)
    np.savez(
        path,
        z1_deg=zeros if z1 is None else z1,
        z3_deg=zeros if z3 is None else z3,
        tilt_offset_deg=zeros if tilt is None else tilt,
        shifts_xy_px=np.zeros((n, 2), dtype=np.float32) if shifts is None else shifts,
    )
    return str(path)


def _coords(d, idxs, N=12, S=4):
    batch = torch.utils.data.default_collate([d[i] for i in idxs])
    return d.get_coords(batch, N=N, num_samples_per_ray=S)


@requires_torch
class TestINRPoseZ1Z3TiltAngle:
    """Separate z1 / z3 groups, the per-view tilt-angle offset and pose_init."""

    def test_learn_tilt_axis_with_z1_raises(self):
        with pytest.raises(ValueError):
            TomographyINRDataset.from_data(
                _stack(), np.linspace(-60, 60, 5, dtype="f4"), learn_tilt_axis=True, learn_z1=True
            )

    def test_separate_groups_from_shared_spec(self):
        d = TomographyINRDataset.from_data(
            _stack(),
            np.linspace(-60, 60, 5, dtype="f4"),
            learn_shift=True,
            learn_tilt_axis=False,
            learn_z1=True,
            learn_z3=True,
            learn_tilt_angle=True,
        )
        d.to("cpu")
        d.set_optimizer(OptimizerParams.Adam(lr=1e-2))
        groups = {group["name"]: group["params"] for group in d.optimizer.param_groups}
        assert set(groups) == {"pose_shift", "pose_z1", "pose_z3", "pose_tilt_angle"}
        assert groups["pose_z1"][0] is d._z1_params
        assert groups["pose_z3"][0] is d._z3_params
        assert groups["pose_tilt_angle"][0] is d._tilt_offset_params
        assert d._tilt_offset_params.shape == (4,)

    def test_forward_gates_z1_and_z3_separately(self):
        d = TomographyINRDataset.from_data(
            _stack(), np.linspace(-60, 60, 5, dtype="f4"), learn_tilt_axis=False, learn_z1=True
        )
        d.to("cpu")
        d.z1_params.data.fill_(1.0)
        d.z3_params.data.fill_(1.0)
        _, z1, z3 = d.forward(None)
        assert bool(z1.any()) and not bool(z3.any())

    def test_tilt_offset_equals_shifted_nominal_angles(self, tmp_path):
        # The INR dataset uses the input angles as phi; an offset of +delta on every view
        # must equal reconstructing with input angles + delta and no offset.
        angles, delta = np.linspace(-60, 60, 5, dtype="f4"), 1.7
        n = len(angles)
        init = _write_pose_init(tmp_path / "p.npz", n, tilt=np.full(n, delta, dtype="f4"))
        kw = dict(learn_shift=False, learn_tilt_axis=False)
        d = TomographyINRDataset.from_data(_stack(), angles, pose_init=init, **kw)
        ref = TomographyINRDataset.from_data(_stack(), angles + delta, **kw)
        plain = TomographyINRDataset.from_data(_stack(), angles, **kw)
        for ds in (d, ref, plain):
            ds.to("cpu")
        idxs = [0, 150, 300, 450, 700]
        cd, cp = _coords(d, idxs), _coords(plain, idxs)
        assert torch.allclose(cd, _coords(ref, idxs), atol=1e-5)
        # box sampling drops out-of-volume samples, so a changed angle may change the count
        assert cd.shape != cp.shape or not torch.allclose(cd, cp, atol=1e-3)

    def test_pose_init_loads_all_views_and_fixes_reference(self, tmp_path):
        angles = np.linspace(-40, 60, 5, dtype="f4")  # reference view is index 2
        n = len(angles)
        vals = np.arange(n, dtype="f4") + 1.0
        init = _write_pose_init(
            tmp_path / "p.npz",
            n,
            z1=vals,
            z3=-vals,
            tilt=0.1 * vals,
            shifts=np.stack([vals, -vals], 1),
        )
        d = TomographyINRDataset.from_data(
            _stack(), angles, learn_tilt_axis=False, learn_tilt_angle=True, pose_init=init
        )
        d.to("cpu")
        shifts, z1, z3 = d.forward(None)
        np.testing.assert_allclose(z1.detach().numpy(), vals)
        np.testing.assert_allclose(z3.detach().numpy(), -vals)
        np.testing.assert_allclose(shifts.detach().numpy(), np.stack([vals, -vals], 1))
        offsets = d.tilt_angle_offsets().detach().numpy()
        np.testing.assert_allclose(offsets, 0.1 * vals, rtol=1e-6)
        assert d._tilt_offset_ref.item() == pytest.approx(0.3)
        d.set_optimizer(OptimizerParams.Adam(lr=1e-2))
        optimized = [p for g in d.optimizer.param_groups for p in g["params"]]
        assert all(p is not d._tilt_offset_ref for p in optimized)

    def test_pose_init_wrong_shape_raises(self, tmp_path):
        init = _write_pose_init(tmp_path / "p.npz", 4)
        with pytest.raises(ValueError):
            TomographyINRDataset.from_data(
                _stack(), np.linspace(-60, 60, 5, dtype="f4"), pose_init=init
            )

    def test_tilt_offsets_survive_to_and_save_load(self, tmp_path):
        angles = np.linspace(-60, 60, 5, dtype="f4")
        d = TomographyINRDataset.from_data(_stack(), angles, learn_tilt_angle=True)
        d.to("cpu")
        d._tilt_offset_params.data.fill_(0.25)
        d.to("cpu")
        assert torch.allclose(d._tilt_offset_params.detach(), torch.full((4,), 0.25))
        path = str(tmp_path / "params.pt")
        d.save_parameters(path)
        d2 = TomographyINRDataset.from_data(_stack(), angles, learn_tilt_angle=True)
        d2.to("cpu")
        d2.load_parameters(path)
        assert torch.allclose(d2._tilt_offset_params.detach(), torch.full((4,), 0.25))


class TestTomographyINRPretrainDataset:
    def test_len_and_getitem(self):
        vol = torch.rand(1, 8, 8, 8)
        ds = TomographyINRPretrainDataset(pretrain_target=vol)
        assert len(ds) == 8**3
        item = ds[0]
        assert set(item.keys()) == {"coords", "target"}
        assert item["coords"].shape == (3,)


@requires_torch
class TestINRRayMath:
    """The static ray helpers are pure tensor math (CPU), exercised here without a recon."""

    def test_create_batch_rays_shape_and_endpoints(self):
        N, S = 8, 5
        rays = TomographyINRDataset.create_batch_rays(
            torch.tensor([0, N - 1]), torch.tensor([0, N - 1]), N=N, num_samples_per_ray=S
        )
        assert rays.shape == (2, S, 3)
        # pixel 0 maps to -1, pixel N-1 maps to +1 on both x (j) and y (i).
        assert torch.allclose(rays[0, :, 0], torch.full((S,), -1.0))
        assert torch.allclose(rays[1, :, 0], torch.full((S,), 1.0))
        # z spans the full -1..1 sampling range.
        assert torch.isclose(rays[0, 0, 2], torch.tensor(-1.0))
        assert torch.isclose(rays[0, -1, 2], torch.tensor(1.0))

    def test_transform_batch_rays_identity_at_zero_pose(self):
        rays = torch.rand(4, 6, 3)
        zero = torch.zeros(4)
        out = TomographyINRDataset.transform_batch_rays(
            rays, z1=zero, x=zero, z3=zero, shifts=torch.zeros(4, 2), N=8, sampling_rate=1.0
        )
        # No rotation and no shift -> rays pass through unchanged.
        assert torch.allclose(out, rays, atol=1e-5)

    def test_integrate_rays_sums_with_step_size(self):
        # integrate_rays became an instance method dispatching on ray_sampling
        # (box_fixed_ds vs legacy); the default legacy path keeps the original
        # step-size summation this test pins down.
        B, S = 3, 5
        stack = _stack(nang=3, n=4)
        d = TomographyINRDataset.from_data(
            stack, np.linspace(-60, 60, 3, dtype="f4"), ray_sampling="legacy"
        )
        out = d.integrate_rays(torch.ones(B * S), num_samples_per_ray=S, target_values_len=B)
        step = 2.0 / (S - 1)
        assert out.shape == (B,)
        assert torch.allclose(out, torch.full((B,), S * step))

    def test_getitem_index_mapping(self):
        # n=4 -> H*W=16 per projection. idx=21 -> projection 1, remaining 5 -> i=1, j=1.
        stack = _stack(nang=3, n=4)
        d = TomographyINRDataset.from_data(stack, np.linspace(-60, 60, 3, dtype="f4"))
        item = d[21]
        assert int(item["projection_idx"]) == 1
        assert int(item["pixel_i"]) == 1
        assert int(item["pixel_j"]) == 1
        assert torch.isclose(item["target_value"], d.tilt_stack[1, 1, 1])

    def test_len_and_getitem_non_square(self):
        # Regression: pixel_i/pixel_j must decompose by the width (shape[2]) and __len__
        # by H*W, not max(shape)^2 -- both were wrong for rectangular tilt images.
        rng = np.random.default_rng(0)
        stack = rng.random((3, 4, 6)).astype(np.float32)  # nang=3, H=4, W=6
        d = TomographyINRDataset.from_data(stack, np.linspace(-60, 60, 3, dtype="f4"))
        assert len(d) == 3 * 4 * 6
        # idx = proj*(H*W) + i*W + j = 1*24 + 2*6 + 3 = 39
        item = d[39]
        assert int(item["projection_idx"]) == 1
        assert int(item["pixel_i"]) == 2
        assert int(item["pixel_j"]) == 3
        assert torch.isclose(item["target_value"], d.tilt_stack[1, 2, 3])
        # every index in range maps to a valid pixel
        last = d[len(d) - 1]
        assert int(last["projection_idx"]) == 2
        assert int(last["pixel_i"]) == 3
        assert int(last["pixel_j"]) == 5

    def test_save_load_parameters_roundtrip(self, tmp_path):
        angles = np.linspace(-60, 60, 5, dtype="f4")
        d = TomographyINRDataset.from_data(_stack(), angles)
        d.to("cpu")
        d.z1_params.data.fill_(0.37)
        path = str(tmp_path / "params.pt")
        d.save_parameters(path)

        d2 = TomographyINRDataset.from_data(_stack(), angles)
        d2.to("cpu")
        d2.load_parameters(path)
        assert torch.allclose(d2.z1_params.detach(), d.z1_params.detach())


@pytest.mark.parametrize("cls", [TomographyPixDataset, TomographyINRDataset])
def test_to_preserves_trained_pose_parameters(cls):
    """Regression: to() rebuilt the pose parameters from the initial-value buffers,
    so any device move after training (e.g. from_file(...).to(device)) silently
    reset the learned pose to zero."""
    d = cls.from_data(_stack(), np.linspace(-60, 60, 5, dtype="f4"))
    d.to("cpu")
    d.z1_params.data.fill_(0.37)
    d.z3_params.data.fill_(-0.21)
    d.shifts_params.data.fill_(1.5)

    d.to("cpu")

    assert torch.allclose(d.z1_params.detach(), torch.full_like(d.z1_params, 0.37))
    assert torch.allclose(d.z3_params.detach(), torch.full_like(d.z3_params, -0.21))
    assert torch.allclose(d.shifts_params.detach(), torch.full_like(d.shifts_params, 1.5))


def test_learnable_tilts_is_read_only():
    """Regression: the old setter wrote a private attribute the getter never read,
    so assignment appeared to succeed while silently doing nothing."""
    d = TomographyPixDataset.from_data(_stack(), np.linspace(-60, 60, 5).astype(np.float32))
    with pytest.raises(AttributeError):
        d.learnable_tilts = 3
    assert d.learnable_tilts == 4
