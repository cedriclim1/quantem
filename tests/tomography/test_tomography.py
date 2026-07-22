"""Tests for the ``Tomography`` / ``TomographyConventional`` orchestrators and the shared
``TomographyBase`` plumbing, without running a reconstruction.

The ``TomographyConventional`` path uses ``ObjectPixelated`` (no DDP setup), so it builds on
CPU and is always-on -- this exercises the bulk of ``tomography_base.py`` (factory, property
setters/validation, loss accessors). The ``Tomography`` (INR) factory and ``save_volume`` go
through ``setup_distributed`` and so follow the ``torch_device`` fixture under
``requires_torch`` (build on CUDA when present; see conftest).
"""

import numpy as np
import pytest
import torch

from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams
from quantem.tomography.dataset_models import TomographyINRDataset, TomographyPixDataset
from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjectINR,
    ObjectPixelated,
    ObjectTensorDecomp,
)
from quantem.tomography.tomography import Tomography, TomographyConventional
from quantem.tomography.tomography_lite import TomographyLiteINR

from .conftest import requires_gpu, requires_torch


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
            torch.equal(before, after) for before, after in zip(initial_pose, pose_at_epoch_end[0])
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

    def test_reconstruct_default_autocast_keeps_parameters_fp32(self):
        tomo = self._inr_tomo("cpu", n=4)
        tomo.reconstruct(
            num_iter=1,
            batch_size=len(tomo.dset),
            num_workers=0,
            num_samples_per_ray=2,
            autocast_dtype=None,
        )
        assert len(tomo.epoch_losses) == 1
        assert all(param.dtype == torch.float32 for param in tomo.obj_model.model.parameters())

    def test_reconstruct_rejects_invalid_autocast_dtype(self):
        tomo = self._inr_tomo("cpu", n=4)
        with pytest.raises(ValueError, match="autocast_dtype"):
            tomo.reconstruct(autocast_dtype="float32")

    def test_reconstruct_bf16_autocast_runs_on_cpu(self):
        from quantem.core.ml.models.kplanes import KPlanes

        n = 4
        model = KPlanes(M_features=2, resolution=(n, n, n))
        obj = ObjectTensorDecomp.from_model(model, shape=(n, n, n), device="cpu")
        dset = TomographyINRDataset.from_data(
            _stack(nang=5, n=n), np.linspace(-60, 60, 5).astype(np.float32)
        )
        tomo = Tomography.from_models(dset=dset, obj_model=obj, device="cpu", verbose=False)
        tomo.reconstruct(
            num_iter=2,
            batch_size=len(dset),
            num_workers=0,
            num_samples_per_ray=2,
            autocast_dtype="bf16",
        )
        assert len(tomo.epoch_losses) == 2

    @requires_gpu
    def test_reconstruct_fp16_autocast_runs_on_cuda(self, monkeypatch):
        grad_scalers = []
        grad_scaler_cls = torch.amp.GradScaler

        def capture_grad_scaler(*args, **kwargs):
            grad_scaler = grad_scaler_cls(*args, **kwargs)
            grad_scalers.append(grad_scaler)
            return grad_scaler

        monkeypatch.setattr(torch.amp, "GradScaler", capture_grad_scaler)
        tomo = self._inr_tomo("cuda:0", n=4)
        tomo.reconstruct(
            num_iter=2,
            batch_size=len(tomo.dset),
            num_workers=0,
            num_samples_per_ray=2,
            autocast_dtype="fp16",
        )

        assert len(tomo.epoch_losses) == 2
        assert np.isfinite(tomo.epoch_losses).all()
        assert len(grad_scalers) == 1
        scale = grad_scalers[0].get_scale()
        assert isinstance(scale, float)
        assert np.isfinite(scale) and scale > 0

    @requires_gpu
    @pytest.mark.parametrize("model_kind", ["inr", "kplanes_tilted"])
    def test_reconstruct_cuda_graphs_matches_eager_fp32(self, model_kind):
        seed = 7

        def build_tomography():
            if model_kind == "inr":
                return self._inr_tomo("cuda:0", n=4)

            from quantem.core.ml.models.kplanes import KPlanesTILTED

            model = KPlanesTILTED(
                M_features=2,
                resolution=(4, 4, 4),
                multiscale_res_multipliers=[1],
                T=2,
                so3_param_type="r9svd",
            )
            obj = ObjectTensorDecomp.from_model(
                model,
                shape=(4, 4, 4),
                device="cuda:0",
            )
            dset = TomographyINRDataset.from_data(
                _stack(nang=5, n=4),
                np.linspace(-60, 60, 5).astype(np.float32),
            )
            return Tomography.from_models(
                dset=dset,
                obj_model=obj,
                device="cuda:0",
                verbose=False,
            )

        torch.manual_seed(seed)
        eager = build_tomography()
        torch.manual_seed(seed)
        graphed = build_tomography()

        eager_initial_params = [
            parameter.detach().clone() for parameter in eager.obj_model.model.parameters()
        ]
        graphed_initial_params = [
            parameter.detach().clone() for parameter in graphed.obj_model.model.parameters()
        ]

        reconstruct_kwargs = {
            "num_iter": 40,
            "batch_size": len(eager.dset),
            "num_workers": 0,
            "num_samples_per_ray": 4,
        }
        if model_kind == "inr":
            reconstruct_kwargs["optimizer_params"] = {
                "object": OptimizerParams.Adam(lr=1e-3)
            }
        else:
            optimizer_params = {
                "grids": OptimizerParams.Adam(lr=1e-3),
                "sigma_net": OptimizerParams.Adam(lr=1e-3),
                "so3": OptimizerParams.Adam(lr=1e-3),
            }
            eager.obj_model.set_optimizer(optimizer_params)
            graphed.obj_model._cuda_graphs_optimizer = True
            try:
                graphed.obj_model.set_optimizer(optimizer_params)
            finally:
                graphed.obj_model._cuda_graphs_optimizer = False
            eager.obj_model.model.so3.requires_grad_(False)
            graphed.obj_model.model.so3.requires_grad_(False)

        def adam_steps(tomo):
            return [
                int(state["step"].item())
                for state in tomo.obj_model.optimizer.state.values()
                if "step" in state
            ]

        graphed_step_history = []

        def record_graphed_step():
            steps = adam_steps(graphed)
            assert steps and len(set(steps)) == 1
            graphed_step_history.append(steps[0])

        graphed._cuda_graph_step_callback = record_graphed_step

        torch.manual_seed(seed)
        eager.reconstruct(**reconstruct_kwargs)
        torch.manual_seed(seed)
        graphed.reconstruct(**reconstruct_kwargs, cuda_graphs=True)
        if model_kind == "kplanes_tilted":
            assert graphed.obj_model.model._rotation_matrices_override is None

        eager_param_delta = max(
            (parameter - initial).abs().max().item()
            for parameter, initial in zip(
                eager.obj_model.model.parameters(), eager_initial_params
            )
        )
        graphed_param_delta = max(
            (parameter - initial).abs().max().item()
            for parameter, initial in zip(
                graphed.obj_model.model.parameters(), graphed_initial_params
            )
        )
        assert eager_param_delta > 0.0
        assert graphed_param_delta > 0.0
        assert graphed_step_history == list(range(1, reconstruct_kwargs["num_iter"] + 1))

        eager_steps = adam_steps(eager)
        graphed_steps = adam_steps(graphed)
        assert eager_steps
        assert graphed_steps
        assert set(eager_steps) == {reconstruct_kwargs["num_iter"]}
        assert graphed_steps == eager_steps

        # Padded rays change reduction order, so Adam compounds small graph/eager
        # differences without a bound in principle. The full trajectory is only a smoke
        # bound; the prefix check and Adam-step/parameter-delta assertions establish
        # faithfulness.
        np.testing.assert_allclose(
            graphed.epoch_losses[:10],
            eager.epoch_losses[:10],
            rtol=1e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            graphed.epoch_losses,
            eager.epoch_losses,
            rtol=2e-3,
            atol=0.0,
        )


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
