"""Tests for the ``Tomography`` / ``TomographyConventional`` orchestrators and the shared
``TomographyBase`` plumbing, without running a reconstruction.

The ``TomographyConventional`` path uses ``ObjectPixelated`` (no DDP setup), so it builds on
CPU and is always-on -- this exercises the bulk of ``tomography_base.py`` (factory, property
setters/validation, loss accessors). The ``Tomography`` (INR) factory and ``save_volume`` go
through ``setup_distributed`` and so follow the ``torch_device`` fixture under
``requires_torch`` (build on CUDA when present; see conftest).
"""

import time

import numpy as np
import pytest
import torch

from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams
from quantem.tomography.dataset_models import TomographyINRDataset, TomographyPixDataset
from quantem.tomography.object_models import ObjConstraintParams, ObjectINR, ObjectPixelated
from quantem.tomography.tomography import Tomography, TomographyConventional
from quantem.tomography.tomography_lite import TomographyLiteINR

from .conftest import requires_torch


def _stack(nang=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((nang, n, n)) * 10).astype(np.float32)


def _conventional(n=12):
    angles = np.linspace(-60, 60, 5).astype(np.float32)
    dset = TomographyPixDataset.from_data(_stack(nang=5, n=n), angles)
    obj = ObjectPixelated.from_uniform(shape=(n, n, n), device="cpu")
    return TomographyConventional.from_models(
        dset=dset, obj_model=obj, device="cpu", verbose=False
    )


class TestConventionalFactory:
    def test_from_models_builds(self):
        tomo = _conventional()
        assert isinstance(tomo, TomographyConventional)
        assert isinstance(tomo.obj_model, ObjectPixelated)
        assert isinstance(tomo.dset, TomographyPixDataset)
        assert tomo.num_epochs == 0

    def test_direct_init_requires_token(self):
        with pytest.raises(RuntimeError):
            TomographyConventional(
                dset=TomographyPixDataset.from_data(
                    _stack(), np.linspace(-60, 60, 5).astype(np.float32)
                ),
                obj_model=ObjectPixelated.from_uniform(shape=(12, 12, 12), device="cpu"),
            )


class TestBaseProperties:
    def test_constraints_setter_dict_and_object(self):
        tomo = _conventional()
        tomo.constraints = {"name": "obj_pixelated", "tv_vol": 0.02}
        assert isinstance(tomo.constraints, ObjConstraintParams.ObjPixelatedConstraints)
        assert tomo.constraints.tv_vol == 0.02
        obj_c = ObjConstraintParams.ObjPixelatedConstraints(positivity=True)
        tomo.constraints = obj_c
        assert tomo.constraints is obj_c

    def test_constraints_setter_none_is_noop(self):
        tomo = _conventional()
        before = tomo.constraints
        tomo.constraints = None
        assert tomo.constraints is before

    def test_constraints_setter_invalid_raises(self):
        tomo = _conventional()
        with pytest.raises(ValueError):
            tomo.constraints = 1.0

    def test_logger_setter_rejects_wrong_type(self):
        tomo = _conventional()
        with pytest.raises(TypeError):
            tomo.logger = "not a logger"

    def test_dset_setter_rejects_wrong_type(self):
        tomo = _conventional()
        with pytest.raises(TypeError):
            tomo.dset = object()

    def test_loss_accessors_start_empty(self):
        tomo = _conventional()
        assert tomo.epoch_losses.shape == (0,)
        assert tomo.consistency_losses.shape == (0,)
        assert tomo.learning_rates == {}

    def test_append_learning_rates_accumulates(self):
        tomo = _conventional()
        tomo.append_learning_rates({"object": 1e-3, "pose": 1e-2})
        tomo.append_learning_rates({"object": 5e-4, "pose": 5e-3})
        assert tomo.learning_rates["object"] == [1e-3, 5e-4]
        assert tomo.learning_rates["pose"] == [1e-2, 5e-3]

    def test_to_updates_device(self):
        tomo = _conventional()
        tomo.to("cpu")
        assert torch.device(tomo.device) == torch.device("cpu")

    def test_plot_losses_runs(self):
        tomo = _conventional()
        tomo._epoch_losses.extend([1.0, 0.5, 0.25])
        tomo.plot_losses()  # Agg backend; plt.show() is a no-op


@requires_torch
class TestInrFactory:
    def _inr_tomo(self, device, n=16):
        from quantem.core.ml.inr import HSiren

        model = HSiren(in_features=3, out_features=1, hidden_layers=1, hidden_features=8)
        obj = ObjectINR.from_model(model, shape=(n, n, n), device=device)
        dset = TomographyINRDataset.from_data(
            _stack(nang=5, n=n), np.linspace(-60, 60, 5).astype(np.float32)
        )
        return Tomography.from_models(dset=dset, obj_model=obj, device=device, verbose=False)

    def test_from_models_builds(self, torch_device):
        tomo = self._inr_tomo(torch_device)
        assert isinstance(tomo, Tomography)
        assert isinstance(tomo.obj_model, ObjectINR)

    def test_plot_losses_runs(self, torch_device):
        tomo = self._inr_tomo(torch_device)
        tomo._epoch_losses.extend([1.0, 0.5])
        tomo._lrs["object"] = [1e-3, 5e-4]
        tomo.plot_losses()

    def test_save_volume_overwrite_guard(self, torch_device, tmp_path):
        tomo = self._inr_tomo(torch_device)
        path = str(tmp_path / "vol.npz")
        tomo.save_volume(path)
        assert (tmp_path / "vol.npz").exists()
        with pytest.raises(FileExistsError):
            tomo.save_volume(path)
        tomo.save_volume(path, overwrite=True)  # must not raise
        with np.load(path) as data:
            assert "volume" in data

    def test_reconstruct_pose_warmup_activation_and_reference_exclusion(self, monkeypatch):
        tomo = self._inr_tomo("cpu", n=4)
        pose_active_at_epoch_start = []
        pose_scheduler_at_epoch_start = []
        object_active_at_epoch_start = []
        initial_pose = [
            param.detach().clone()
            for param in (tomo.dset._shifts_params, tomo.dset._z1_params, tomo.dset._z3_params)
        ]
        pose_at_epoch_end = []
        original_train = tomo.dset.train

        def record_optimizer_state(mode=True):
            pose_active_at_epoch_start.append(tomo.dset.has_optimizer())
            pose_scheduler_at_epoch_start.append(tomo.dset.scheduler is not None)
            object_active_at_epoch_start.append(tomo.obj_model.has_optimizer())
            return original_train(mode)

        def record_pose(_epoch):
            pose_at_epoch_end.append(
                [
                    param.detach().clone()
                    for param in (
                        tomo.dset._shifts_params,
                        tomo.dset._z1_params,
                        tomo.dset._z3_params,
                    )
                ]
            )

        monkeypatch.setattr(tomo.dset, "train", record_optimizer_state)
        tomo.reconstruct(
            num_iter=2,
            batch_size=len(tomo.dset),
            num_workers=0,
            num_samples_per_ray=2,
            optimizer_params={
                "object": OptimizerParams.Adam(lr=1e-3),
                "pose": {
                    "pose_shift": OptimizerParams.Adam(lr=2e-2),
                    "pose_tilt_axis": OptimizerParams.Adam(lr=3e-3),
                },
            },
            scheduler_params={
                "object": SchedulerParams.Exponential(gamma=0.9),
                "pose": SchedulerParams.Exponential(gamma=0.9),
            },
            pose_warmup_epochs=1,
            eval_callback=record_pose,
            eval_every=1,
        )

        assert pose_active_at_epoch_start == [False, True]
        assert pose_scheduler_at_epoch_start == [False, True]
        assert object_active_at_epoch_start == [True, True]
        assert all(
            torch.equal(before, after)
            for before, after in zip(initial_pose, pose_at_epoch_end[0])
        )
        assert any(
            not torch.equal(before, after)
            for before, after in zip(pose_at_epoch_end[0], pose_at_epoch_end[1])
        )
        assert {group["name"] for group in tomo.dset.optimizer.param_groups} == {
            "pose_shift",
            "pose_tilt_axis",
        }
        optimized = [
            param for group in tomo.dset.optimizer.param_groups for param in group["params"]
        ]
        assert all(
            ref is not param
            for ref in (tomo.dset._shifts_ref, tomo.dset._z1_ref, tomo.dset._z3_ref)
            for param in optimized
        )

        default_tomo = self._inr_tomo("cpu", n=4)
        default_tomo.reconstruct(
            num_iter=0,
            batch_size=len(default_tomo.dset),
            num_workers=0,
            optimizer_params={"pose": OptimizerParams.Adam(lr=1e-2)},
        )
        assert default_tomo.dset.has_optimizer()

        with pytest.raises(ValueError, match="pose_warmup_epochs must be >= 0"):
            default_tomo.reconstruct(num_iter=0, pose_warmup_epochs=-1)

    def test_overhead_window_nested_counts_once(self):
        tomo = self._inr_tomo("cpu", n=4)
        assert tomo.overhead_s == 0.0
        with tomo._overhead_window():
            with tomo._overhead_window():
                time.sleep(0.02)
            assert tomo.overhead_s >= 0.02
        assert 0.02 <= tomo.overhead_s < 1.0
        assert tomo._overhead_t0 is None

    def test_reconstruct_counts_snapshot_and_callback_time_as_overhead(self):
        tomo = self._inr_tomo("cpu", n=4)
        seen = []

        def slow_snapshot(_step, _volume):
            time.sleep(0.05)

        def record(_epoch):
            seen.append(tomo.overhead_s)
            time.sleep(0.05)

        tomo.reconstruct(
            num_iter=2,
            batch_size=len(tomo.dset),
            num_workers=0,
            num_samples_per_ray=2,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
            eval_callback=record,
            eval_every=1,
            snapshot_every=1,
            snapshot_callback=slow_snapshot,
        )

        assert tomo._grad_steps == 2
        assert seen[0] >= 0.05
        assert seen[1] >= seen[0] + 0.1
        assert tomo.overhead_s >= 0.2
        assert tomo._overhead_t0 is None

    def _exact_kwargs(self, tomo, batch_size=16, **kwargs):
        out = dict(
            batch_size=batch_size,
            num_workers=0,
            num_samples_per_ray=2,
            optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
        )
        out.update(kwargs)
        return out

    def test_max_grad_steps_stops_mid_epoch_and_averages_the_partial_epoch(self, monkeypatch):
        from quantem.tomography import tomography as tomography_module

        tomo = self._inr_tomo("cpu", n=4)
        assert tomo.grad_steps_per_epoch(16) == 5
        batch_losses = []
        original = tomography_module.get_loss_module

        def recording_loss(**kwargs):
            loss = original(**kwargs)

            def wrapped(pred, target):
                value = loss(pred, target)
                batch_losses.append(float(value.detach()))
                return value

            return wrapped

        monkeypatch.setattr(tomography_module, "get_loss_module", recording_loss)
        snapshots, evals = [], []
        tomo.reconstruct(
            num_iter=3,
            max_grad_steps=7,
            snapshot_every=5,
            snapshot_steps={7},
            snapshot_callback=lambda step, vol: snapshots.append(step),
            eval_callback=evals.append,
            eval_every=2,
            **self._exact_kwargs(tomo),
        )
        assert tomo._grad_steps == 7
        assert tomo.num_epochs == 2
        assert snapshots == [5, 7]
        assert evals == [2]
        assert len(batch_losses) == 7
        assert tomo._consistency_losses[0] == pytest.approx(np.mean(batch_losses[:5]), rel=1e-5)
        assert tomo._consistency_losses[1] == pytest.approx(np.mean(batch_losses[5:]), rel=1e-5)

    def test_max_grad_steps_validates_the_stopping_epoch(self):
        tomo = self._inr_tomo("cpu", n=4)
        tomo.reconstruct(
            num_iter=4,
            max_grad_steps=6,
            holdout_fraction=0.2,
            holdout_every=10,
            **self._exact_kwargs(tomo),
        )
        assert tomo._grad_steps == 6
        assert tomo.num_epochs == 2
        assert len(tomo._val_losses) == 1

    def test_max_grad_steps_beyond_num_iter_runs_every_epoch(self):
        tomo = self._inr_tomo("cpu", n=4)
        tomo.reconstruct(num_iter=2, max_grad_steps=100, **self._exact_kwargs(tomo))
        assert tomo._grad_steps == 10
        assert tomo.num_epochs == 2

    def test_max_grad_steps_counts_from_the_start_of_each_call(self):
        tomo = self._inr_tomo("cpu", n=4)
        tomo.reconstruct(num_iter=1, **self._exact_kwargs(tomo))
        tomo.reconstruct(num_iter=2, max_grad_steps=3, **self._exact_kwargs(tomo))
        assert tomo._grad_steps == 8

    @pytest.mark.parametrize("value", [0, -1])
    def test_max_grad_steps_rejects_non_positive(self, value):
        tomo = self._inr_tomo("cpu", n=4)
        with pytest.raises(ValueError, match="max_grad_steps"):
            tomo.reconstruct(num_iter=1, max_grad_steps=value, **self._exact_kwargs(tomo))

    def test_grad_step_cosine_schedule_decays_every_step_to_eta_min(self):
        tomo = self._inr_tomo("cpu", n=4)
        lrs = []
        tomo.reconstruct(
            num_iter=3,
            max_grad_steps=7,
            scheduler_params={"object": {"name": "cosine_annealing", "eta_min": 0.0}},
            scheduler_step="grad_step",
            snapshot_every=1,
            snapshot_callback=lambda step, vol: lrs.append(tomo.obj_model.get_current_lr()),
            **self._exact_kwargs(tomo),
        )
        expected = [0.5e-3 * (1 + np.cos(np.pi * k / 7)) for k in range(1, 8)]
        np.testing.assert_allclose(lrs, expected, rtol=1e-6, atol=1e-12)
        assert all(b < a for a, b in zip(lrs, lrs[1:]))
        assert tomo.learning_rates["object"][-1] == pytest.approx(0.0, abs=1e-12)

    def test_epoch_cosine_schedule_is_unchanged_by_default(self):
        tomo = self._inr_tomo("cpu", n=4)
        lrs = []
        tomo.reconstruct(
            num_iter=2,
            scheduler_params={"object": {"name": "cosine_annealing", "eta_min": 0.0}},
            snapshot_every=1,
            snapshot_callback=lambda step, vol: lrs.append(tomo.obj_model.get_current_lr()),
            **self._exact_kwargs(tomo),
        )
        assert lrs == [pytest.approx(1e-3)] * 5 + [pytest.approx(0.5e-3)] * 5

    def test_grad_step_schedule_rejects_plateau(self):
        tomo = self._inr_tomo("cpu", n=4)
        with pytest.raises(ValueError, match="plateau"):
            tomo.reconstruct(
                num_iter=1,
                scheduler_params={"object": {"name": "plateau"}},
                scheduler_step="grad_step",
                **self._exact_kwargs(tomo),
            )

    def test_scheduler_step_rejects_unknown_value(self):
        tomo = self._inr_tomo("cpu", n=4)
        with pytest.raises(ValueError, match="scheduler_step"):
            tomo.reconstruct(num_iter=1, scheduler_step="batch", **self._exact_kwargs(tomo))

    def test_snapshot_steps_add_explicit_steps(self):
        tomo = self._inr_tomo("cpu", n=4)
        snapshots = []
        tomo.reconstruct(
            num_iter=2,
            snapshot_steps=[1, 3, 10, 50],
            snapshot_callback=lambda step, vol: snapshots.append(step),
            **self._exact_kwargs(tomo),
        )
        assert snapshots == [1, 3, 10]


@requires_torch
class TestLiteINRReconstructBranch:
    """``TomographyLiteINR.reconstruct`` bundles optimizer/scheduler params only on the first
    epoch and passes ``None`` afterwards. Stub out the heavy ``Tomography.reconstruct`` to
    assert the branch without running a reconstruction."""

    def _lite(self, device, n=12):
        return TomographyLiteINR.from_dataset(
            tilt_series=_stack(nang=5, n=n),
            tilt_angles=np.linspace(-60, 60, 5).astype(np.float32),
            device=device,
        )

    def test_param_bundling_first_then_subsequent(self, torch_device, monkeypatch):
        tomo = self._lite(torch_device)
        captured = {}

        def fake_reconstruct(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            self._epoch_losses.append(1.0)  # mark an epoch as having run

        monkeypatch.setattr(Tomography, "reconstruct", fake_reconstruct)

        # First call (num_epochs == 0): object + pose params are assembled.
        tomo.reconstruct(num_iter=1, num_workers=0, learn_pose=True)
        assert set(captured["optimizer_params"].keys()) == {"object", "pose"}
        assert set(captured["scheduler_params"].keys()) == {"object", "pose"}

        # Second call (num_epochs > 0): params are passed through as None.
        tomo.reconstruct(num_iter=1, num_workers=0)
        assert captured["optimizer_params"] is None
        assert captured["scheduler_params"] is None
