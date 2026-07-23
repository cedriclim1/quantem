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

    def test_reconstruct_float32_dtype_matches_disabled_autocast(self):
        def run(autocast_dtype):
            torch.manual_seed(17)
            tomo = self._inr_tomo("cpu", n=4)
            tomo.reconstruct(
                num_iter=1,
                batch_size=len(tomo.dset),
                num_workers=0,
                num_samples_per_ray=2,
                autocast_dtype=autocast_dtype,
            )
            return tomo

        disabled = run(None)
        float32 = run(torch.float32)

        np.testing.assert_allclose(float32.epoch_losses, disabled.epoch_losses)
        for actual, expected in zip(
            float32.obj_model.model.parameters(), disabled.obj_model.model.parameters()
        ):
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("disabled_value", [None, 0.0])
    def test_reconstruct_can_skip_inactive_gradient_clipping(self, monkeypatch, disabled_value):
        original_clip = torch.nn.utils.clip_grad_norm_
        clip_calls = []

        def tracking_clip(parameters, max_norm, *args, **kwargs):
            result = original_clip(parameters, max_norm, *args, **kwargs)
            clip_calls.append((float(result), float(max_norm)))
            return result

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", tracking_clip)

        def run(grad_clip_max_norm):
            torch.manual_seed(19)
            tomo = self._inr_tomo("cpu", n=4)
            torch.manual_seed(23)
            tomo.reconstruct(
                num_iter=1,
                batch_size=len(tomo.dset),
                num_workers=0,
                num_samples_per_ray=2,
                grad_clip_max_norm=grad_clip_max_norm,
            )
            return (
                tomo,
                [
                    parameter.grad.detach().clone()
                    for parameter in tomo.obj_model.model.parameters()
                ],
                [parameter.detach().clone() for parameter in tomo.obj_model.model.parameters()],
            )

        clipped, clipped_grads, clipped_params = run(1e6)
        unclipped, unclipped_grads, unclipped_params = run(disabled_value)

        assert len(clip_calls) == 1
        total_norm, max_norm = clip_calls[0]
        assert total_norm < max_norm
        np.testing.assert_allclose(unclipped.epoch_losses, clipped.epoch_losses)
        for actual, expected in zip(unclipped_grads, clipped_grads):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(unclipped_params, clipped_params):
            torch.testing.assert_close(actual, expected)

    def test_reconstruct_rejects_negative_gradient_clip_norm(self):
        tomo = self._inr_tomo("cpu", n=4)
        with pytest.raises(ValueError, match="grad_clip_max_norm"):
            tomo.reconstruct(num_iter=0, grad_clip_max_norm=-1.0)

    def test_reconstruct_rejects_invalid_autocast_dtype(self):
        tomo = self._inr_tomo("cpu", n=4)
        for invalid_dtype in ("float32", torch.float64):
            with pytest.raises(ValueError, match="autocast_dtype"):
                tomo.reconstruct(autocast_dtype=invalid_dtype)

    def test_reconstruct_bfloat16_dtype_matches_bf16_string_on_cpu(self):
        from quantem.core.ml.models.kplanes import KPlanes

        def run(autocast_dtype):
            torch.manual_seed(17)
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
                autocast_dtype=autocast_dtype,
            )
            return tomo

        string_dtype = run("bf16")
        torch_dtype = run(torch.bfloat16)

        np.testing.assert_allclose(torch_dtype.epoch_losses, string_dtype.epoch_losses)
        for actual, expected in zip(
            torch_dtype.obj_model.model.parameters(),
            string_dtype.obj_model.model.parameters(),
        ):
            torch.testing.assert_close(actual, expected)

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
    @pytest.mark.parametrize(
        ("model_kind", "multiscale_res_multipliers"),
        [
            pytest.param("inr", None, id="inr"),
            pytest.param("kplanes_tilted", [1], id="kplanes-single-level"),
            pytest.param(
                "kplanes_tilted",
                [0.25, 0.75, 1.0],
                id="kplanes-three-level-ms-tv",
            ),
        ],
    )
    def test_reconstruct_cuda_graphs_matches_eager_fp32(
        self, model_kind, multiscale_res_multipliers, monkeypatch
    ):
        seed = 7
        monkeypatch.delenv("QUANTEM_KPLANES_MS_TV_FUSED", raising=False)

        def build_tomography():
            if model_kind == "inr":
                return self._inr_tomo("cuda:0", n=4)

            from quantem.core.ml.models.kplanes import KPlanesTILTED

            model = KPlanesTILTED(
                M_features=2,
                resolution=(4, 4, 4),
                multiscale_res_multipliers=multiscale_res_multipliers,
                T=2,
                so3_param_type="r9svd",
            )
            obj = ObjectTensorDecomp.from_model(
                model,
                shape=(4, 4, 4),
                device="cuda:0",
            )
            if len(multiscale_res_multipliers) == 3:
                obj.constraints.tv_plane = 0.05
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
            reconstruct_kwargs["optimizer_params"] = {"object": OptimizerParams.Adam(lr=1e-3)}
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
            for parameter, initial in zip(eager.obj_model.model.parameters(), eager_initial_params)
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

    @requires_gpu
    @pytest.mark.parametrize("num_steps", [1, 10])
    def test_pred_fork_matches_single_stream_state(self, monkeypatch, num_steps):
        def build_tomography():
            tomo = self._inr_tomo("cuda:0", n=4)
            tomo.obj_model.constraints = ObjConstraintParams.ObjINRConstraints(
                s3im_weight=0.05,
                s3im_repeat_time=2,
                s3im_kernel=2,
                s3im_value_range=10.0,
            )
            return tomo

        def run(tomo, enabled):
            if enabled:
                monkeypatch.setenv("QUANTEM_RECON_PRED_FORK", "1")
            else:
                monkeypatch.setenv("QUANTEM_RECON_PRED_FORK", "0")
            torch.manual_seed(29)
            tomo.reconstruct(
                num_iter=num_steps,
                batch_size=len(tomo.dset),
                num_workers=0,
                num_samples_per_ray=2,
                optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
                grad_clip_max_norm=None,
            )
            torch.cuda.synchronize()

        torch.manual_seed(17)
        reference = build_tomography()
        torch.manual_seed(17)
        forked = build_tomography()
        run(reference, enabled=False)
        run(forked, enabled=True)

        # The 100-step arm was removed because intrinsic atomic-order chaos
        # (1.5e-2 ref-vs-ref) exceeds any assertable bound; the standing measurement is
        # test_pred_fork_reference_intrinsic_drift.
        np.testing.assert_allclose(forked.epoch_losses, reference.epoch_losses, rtol=1e-4)
        np.testing.assert_allclose(
            forked.consistency_losses, reference.consistency_losses, rtol=1e-4
        )
        np.testing.assert_allclose(
            forked.obj_model.soft_constraint_losses,
            reference.obj_model.soft_constraint_losses,
            rtol=1e-4,
        )
        for forked_parameter, reference_parameter in zip(
            forked.obj_model.model.parameters(), reference.obj_model.model.parameters()
        ):
            torch.testing.assert_close(
                forked_parameter.grad,
                reference_parameter.grad,
                rtol=1e-4,
                atol=1e-7,
            )
            torch.testing.assert_close(
                forked_parameter,
                reference_parameter,
                rtol=1e-4,
                atol=1e-7,
            )
            forked_state = forked.obj_model.optimizer.state[forked_parameter]
            reference_state = reference.obj_model.optimizer.state[reference_parameter]
            assert forked_state.keys() == reference_state.keys()
            for key in forked_state:
                if isinstance(forked_state[key], torch.Tensor):
                    torch.testing.assert_close(
                        forked_state[key],
                        reference_state[key],
                        rtol=1e-4,
                        atol=1e-7,
                    )
                else:
                    assert forked_state[key] == reference_state[key]

    @requires_gpu
    @pytest.mark.parametrize("num_steps", [10, 100])
    def test_pred_fork_reference_intrinsic_drift(self, monkeypatch, num_steps):
        def build_tomography():
            tomo = self._inr_tomo("cuda:0", n=4)
            tomo.obj_model.constraints = ObjConstraintParams.ObjINRConstraints(
                s3im_weight=0.05,
                s3im_repeat_time=2,
                s3im_kernel=2,
                s3im_value_range=10.0,
            )
            return tomo

        def run(tomo):
            monkeypatch.setenv("QUANTEM_RECON_PRED_FORK", "0")
            torch.manual_seed(29)
            tomo.reconstruct(
                num_iter=num_steps,
                batch_size=len(tomo.dset),
                num_workers=0,
                num_samples_per_ray=2,
                optimizer_params={"object": OptimizerParams.Adam(lr=1e-3)},
                grad_clip_max_norm=None,
            )
            torch.cuda.synchronize()

        torch.manual_seed(17)
        reference_a = build_tomography()
        torch.manual_seed(17)
        reference_b = build_tomography()
        run(reference_a)
        run(reference_b)

        losses_a = reference_a.epoch_losses
        losses_b = reference_b.epoch_losses
        d = np.max(np.abs(losses_a - losses_b) / np.abs(losses_a))
        print(f"intrinsic drift @{num_steps} steps: {d:.3e}")
        assert np.isfinite(d)


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
