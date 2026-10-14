import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Self, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import nvtx
from tqdm.auto import tqdm

from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.profiling import nsys_capture_tick
from quantem.core.ml.models.kplanes import CPTilted, KPlanesTILTED
from quantem.core.utils.filter import gaussian_filter_2d_stack, gaussian_kernel_1d
from quantem.core.utils.tomography_utils import torch_phase_cross_correlation
from quantem.tomography.dataset_models import (
    DatasetConstraintParams,
    DatasetConstraintsType,
    DatasetModelType,
    DeviceBatchSampler,
    TomographyINRDataset,
    TomographyPixDataset,
    build_pixel_holdout_split,
)
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjConstraintsType,
    ObjectINR,
    ObjectPixelated,
    ObjectTensorDecomp,
    _tomography_autocast,
    _unwrap,
)
from quantem.tomography.radon.radon import iradon_torch, radon_torch
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_context import ReconstructionContext
from quantem.tomography.tomography_opt import TomographyOpt


def _should_take_grad_step_snapshot(grad_step: int, snapshot_every: int) -> bool:
    """Return whether a gradient-update snapshot should fire at ``grad_step``."""
    return snapshot_every > 0 and grad_step > 0 and grad_step % snapshot_every == 0


def _take_grad_step_snapshot(
    *,
    obj_model: ObjectINR | ObjectTensorDecomp,
    grad_step: int,
    global_rank: int,
    logger: LoggerTomography | None,
    snapshot_dir: str | Path | None,
    snapshot_callback: Callable[[int, np.ndarray | None], None] | None,
) -> None:
    volume = obj_model.obj_view
    volume_or_none = volume if global_rank == 0 else None

    if global_rank == 0:
        if snapshot_dir is not None:
            snapshot_path = Path(snapshot_dir) / f"step_{grad_step}.npy"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(snapshot_path, np.asarray(volume_or_none).astype(np.float32, copy=False))

        if logger is not None:
            logger.log_scalar(
                "snapshots/last_grad_step",
                float(grad_step),
                grad_step,
                step_domain="grad_step",
            )

    if snapshot_callback is not None:
        snapshot_callback(grad_step, volume_or_none)


PHASE1_UNSUPPORTED_CONSTRAINTS = ("tv_plane", "l21_plane")


@dataclass
class TwoPhaseParams:
    """Phase-1 (CPTilted bottleneck) settings for ``Tomography.reconstruct_two_phase``.

    ``model_kwargs`` go to ``CPTilted`` (e.g. ``C``, ``multiscale_res_multipliers``,
    ``so3_param_type``, ``tau_init``); ``T``, ``resolution`` and ``density_activation``
    default to the phase-2 model's and ``T`` must equal it. ``optimizer_params`` and
    ``scheduler_params`` use the ``reconstruct`` format; the object group keys are
    ``grids``, ``sigma_net`` and ``so3``, and pose is frozen in phase 1 unless a
    ``"pose"`` entry is given. ``obj_constraints`` of None reuses the phase-2
    constraints with ``tv_plane`` and ``l21_plane`` set to 0 (CPTilted has line grids,
    on which neither is defined); an explicit value with either > 0 raises.
    ``num_samples_per_ray`` of None reuses the phase-2 value unless that is a schedule.
    """

    num_iter: int
    optimizer_params: dict
    scheduler_params: dict | None = None
    model_kwargs: dict = field(default_factory=dict)
    obj_constraints: dict | ObjConstraintsType | None = None
    num_samples_per_ray: int | list[tuple[int, int]] | None = None

    @classmethod
    def parse(cls, params: "TwoPhaseParams | dict") -> "TwoPhaseParams":
        """Return params as a TwoPhaseParams, building it from a dict if needed."""
        return params if isinstance(params, cls) else cls(**params)


class Tomography(TomographyOpt, TomographyBase):
    """
    Class for handling all ML tomography reconstruction methods.
    Automatic handling between AD and INR-based tomography.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectINR | ObjectTensorDecomp,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
            verbose=verbose,
            _token=cls._token,
        )

    def _sync_pose_gradients_ddp(self) -> None:
        """Average otherwise-unsynced pose gradients that live outside DDP.

        Averaging rank-local shard gradients gives the full-batch direction and keeps
        pose updates in lockstep across ranks.
        """
        if self.world_size <= 1 or not self.dset.has_optimizer():
            return
        if not dist.is_available() or not dist.is_initialized():
            return

        groups = self.dset.get_optimization_parameters()
        for key in sorted(groups.keys()):
            for p in groups[key]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def reconstruct(
        self,
        num_iter: int = 10,
        batch_size: int = 1024,
        num_workers: int = 32,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        obj_constraints: dict | ObjConstraintsType | None = None,
        dset_constraints: dict | DatasetConstraintsType | None = None,
        num_samples_per_ray: int | list[tuple[int, int]] | None = None,
        profiling_mode: bool = False,
        val_fraction: float = 0.0,
        holdout_fraction: float = 0.0,
        holdout_seed: int = 0,
        holdout_every: int = 1,
        loss_type: Literal[
            "l2",
            "l1",
            "smooth_l1",
            "charbonnier",
            "llmse",
            "mse_log_mse",
        ] = "l2",
        loss_func_kwargs: dict = {},
        reset_dset: DatasetModelType | None = None,
        show_metrics: bool = False,
        eval_callback: Callable[[int], None] | None = None,
        eval_every: int = 0,
        snapshot_every: int = 0,
        snapshot_dir: str | Path | None = None,
        snapshot_callback: Callable[[int, np.ndarray | None], None] | None = None,
        pose_warmup_epochs: int = 0,
        use_bfloat16: bool = True,
    ):
        """
        This function should be able to handle both AD and INR-based tomography reconstruction methods.
        I.e, auto-detection through the obj model type, while both share the same pose optimization.
        """
        if snapshot_every < 0:
            raise ValueError("snapshot_every must be >= 0.")
        if pose_warmup_epochs < 0:
            raise ValueError("pose_warmup_epochs must be >= 0.")
        if holdout_every < 1:
            raise ValueError("holdout_every must be >= 1.")
        if val_fraction > 0.0 and holdout_fraction > 0.0:
            raise ValueError("Use either val_fraction or holdout_fraction, not both.")
        snapshots_enabled = snapshot_every > 0

        # Check device consistency
        self.obj_model.to(self.device)

        previous_batch_size = getattr(self, "batch_size", None)
        previous_num_workers = getattr(self, "num_workers", None)
        previous_val_fraction = getattr(self, "val_fraction", None)
        previous_holdout_fraction = getattr(self, "holdout_fraction", None)
        previous_holdout_seed = getattr(self, "holdout_seed", None)

        # Saving batch size, num workers, and validation split settings for reloading
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed
        self.holdout_every = holdout_every

        if profiling_mode:
            if self.global_rank == 0:
                print("Profiling mode enabled.")

        if reset:
            raise NotImplementedError("Reset is not implemented yet.")

        new_scheduler = reset
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()
            new_scheduler = True

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True

        if new_scheduler:
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)

        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)

            self.obj_model.constraints = obj_constraints

        if dset_constraints is not None:
            if isinstance(dset_constraints, dict):
                dset_constraints = DatasetConstraintParams.parse_dict(dset_constraints)

            self.dset.constraints = dset_constraints
        dataloader_needs_rebuild = (
            not hasattr(self, "dataloader")
            or reset_dset is not None
            or previous_batch_size != batch_size
            or previous_num_workers != num_workers
            or previous_val_fraction != val_fraction
            or previous_holdout_fraction != holdout_fraction
            or previous_holdout_seed != holdout_seed
        )

        # Setting up DDP
        if dataloader_needs_rebuild:
            if reset_dset is not None:
                print("Resetting Dataloader")
                print("Putting in params from previous dataset.")

                self.dset = reset_dset
                self.dset.to(self.device)

                if optimizer_params is not None:
                    self.optimizer_params = optimizer_params
                    self.set_optimizers()
                if scheduler_params is not None:
                    self.scheduler_params = scheduler_params
                    self.set_schedulers(self.scheduler_params, num_iter=num_iter)

            self._setup_recon_dataloaders(
                batch_size, num_workers, val_fraction, holdout_fraction, holdout_seed
            )

        # Type check for INR-based reconstruction
        if not isinstance(self.dset, TomographyINRDataset):
            raise NotImplementedError(
                "Only TomographyINRDataset is supported for this reconstruction method."
            )

        pose_warmup_pending = pose_warmup_epochs > 0 and self.dset.has_optimizer()
        if pose_warmup_pending:
            # remove_optimizer also clears the stored specs; restore those specs without
            # rebuilding the optimizer/scheduler until the warmup finishes.
            pose_optimizer_params = self.optimizer_params["pose"]
            pose_scheduler_params = self.scheduler_params["pose"]
            self.dset.remove_optimizer()
            self.dset.optimizer_params = pose_optimizer_params
            self.dset.scheduler_params = pose_scheduler_params

        N = max(self.obj_model.shape)

        if num_samples_per_ray is None:
            num_samples_per_ray = max(self.obj_model.shape)
        else:
            if isinstance(num_samples_per_ray, int):
                num_samples_per_ray = num_samples_per_ray
            else:
                if len(num_samples_per_ray) != num_iter:
                    raise ValueError(
                        "num_samples_per_ray schedule must have the same length as num_iter"
                    )
                if self.global_rank == 0:
                    print("num_samples_per_ray schedule provided.")

        loss_func = get_loss_module(name=loss_type, dtype=self.obj_model.dtype, **loss_func_kwargs)

        set_progress = (
            getattr(_unwrap(self.obj_model.model), "set_progress", None)
            if isinstance(self.obj_model, ObjectINR)
            else None
        )
        planned_steps = max(1, num_iter * len(self.dataloader))
        call_steps = 0
        if set_progress is not None:
            set_progress(0.0)

        pbar = tqdm(range(num_iter), disable=not self.verbose)
        for a0 in pbar:
            if pose_warmup_pending and a0 == pose_warmup_epochs:
                self.dset.set_optimizer(self.optimizer_params["pose"])
                self.dset.set_scheduler(self.scheduler_params["pose"], num_iter=num_iter)
                pose_warmup_pending = False
            nvtx.range_push(f"epoch_{a0}")
            consistency_loss = torch.tensor(0.0, device=self.device)
            total_loss = torch.tensor(0.0, device=self.device)
            epoch_soft_constraint_loss = torch.tensor(0.0, device=self.device)
            epoch_soft_terms: dict[str, torch.Tensor] = {}
            active_obj_constraints = self.obj_model.constraints
            distortion_enabled = getattr(active_obj_constraints, "distortion", 0.0) > 0
            active_dset_constraints = getattr(self.dset, "constraints", None)
            pose_tv_enabled = (
                getattr(active_dset_constraints, "tv_zs", 0.0) > 0
                or getattr(active_dset_constraints, "tv_shifts", 0.0) > 0
            )
            if isinstance(self.obj_model, ObjectINR) or isinstance(
                self.obj_model, ObjectTensorDecomp
            ):
                self.obj_model.model.train()
            else:
                raise NotImplementedError(
                    "AD Pixelated reconstruction is not yet implemented. Use ObjectINR instead."
                )
            self.dset.train()
            # self._reset_iter_constraints()

            if self.sampler is not None:
                self.sampler.set_epoch(a0)

            if isinstance(num_samples_per_ray, list):
                curr_num_samples_per_ray = num_samples_per_ray[a0][1]
            else:
                curr_num_samples_per_ray = num_samples_per_ray

            for batch_idx, batch in enumerate(self.dataloader):
                nvtx.range_push("batch")
                self.zero_grad_all()
                with _tomography_autocast(self.device, use_bfloat16):
                    nvtx.range_push("get_coords")
                    all_coords = self.dset.get_coords(batch, N, curr_num_samples_per_ray)
                    nvtx.range_pop()

                    nvtx.range_push("obj_forward")
                    tap_coords = self.obj_model.sample_tv_tap_coords(all_coords)
                    if tap_coords is not None:
                        all_densities, tv_tap_raw = self.obj_model.forward_with_tv_taps(
                            all_coords, tap_coords
                        )
                    else:
                        all_densities = self.obj_model.forward(all_coords)
                        tv_tap_raw = None
                    nvtx.range_pop()

                    batch_distortion = None
                    target = None
                    if distortion_enabled:
                        nvtx.range_push("ray_distortion")
                        target = batch["target_value"].to(self.device, non_blocking=True).float()
                        batch_distortion = self.dset.ray_distortion(
                            all_densities,
                            target,
                            norm=active_obj_constraints.distortion_norm,
                            eps_rel=active_obj_constraints.distortion_eps_rel,
                        )
                        nvtx.range_pop()

                    nvtx.range_push("integrate_rays")
                    integrated_densities = self.dset.integrate_rays(
                        all_densities,
                        curr_num_samples_per_ray,
                        len(batch["target_value"]),
                    )
                    nvtx.range_pop()

                pred = integrated_densities.float()

                if target is None:
                    target = batch["target_value"].to(self.device, non_blocking=True).float()

                nvtx.range_push("soft_constraints")
                batch_soft_terms: dict[str, torch.Tensor] = {}
                soft_constraints_loss = self.obj_model.apply_soft_constraints(
                    ctx=ReconstructionContext(
                        coords=all_coords,
                        pred=pred,
                        all_densities=all_densities,
                        target=target,
                        tv_tap_densities=tv_tap_raw,
                        distortion=batch_distortion,
                        soft_terms=batch_soft_terms,
                    )
                )
                nvtx.range_pop()

                nvtx.range_push("consistency_loss")
                batch_consistency_loss = loss_func(pred, target)

                dataset_soft_loss = self.dset.apply_soft_constraints()
                soft_constraints_loss += dataset_soft_loss
                if pose_tv_enabled:
                    batch_soft_terms["pose_tv"] = dataset_soft_loss.detach()

                epoch_soft_constraint_loss += soft_constraints_loss.detach()
                for term_name, term_value in batch_soft_terms.items():
                    if term_name in epoch_soft_terms:
                        epoch_soft_terms[term_name] += term_value.float()
                    else:
                        epoch_soft_terms[term_name] = term_value.float().clone()

                batch_loss = batch_consistency_loss.float() + soft_constraints_loss.float()
                nvtx.range_pop()

                nvtx.range_push("backward")
                batch_loss.backward()
                nvtx.range_pop()
                # Clip gradients
                nvtx.range_push("clip_and_optim_step")
                self._sync_pose_gradients_ddp()
                torch.nn.utils.clip_grad_norm_(self.obj_model.model.parameters(), max_norm=1.0)
                self.step_optimizers()
                nvtx.range_pop()
                self._grad_steps = getattr(self, "_grad_steps", 0) + 1
                call_steps += 1
                if set_progress is not None:
                    set_progress(call_steps / planned_steps)
                nsys_capture_tick(self._grad_steps)
                if snapshots_enabled and _should_take_grad_step_snapshot(
                    self._grad_steps, snapshot_every
                ):
                    _take_grad_step_snapshot(
                        obj_model=self.obj_model,
                        grad_step=self._grad_steps,
                        global_rank=self.global_rank,
                        logger=self.logger,
                        snapshot_dir=snapshot_dir,
                        snapshot_callback=snapshot_callback,
                    )
                total_loss += batch_loss.detach()
                consistency_loss += batch_consistency_loss.detach()
                nvtx.range_pop()  # batch

            if isinstance(self.obj_model.model, CPTilted):
                if a0 == 0:
                    prev_R = self.obj_model.model.so3.as_matrix().detach().clone()
                elif (a0 + 1) % 20 == 0:
                    R_now = self.obj_model.model.so3.as_matrix().detach()
                    # Cumulative angular change per rotation over the last 20 iters.
                    # trace(R_prev^T R_now) = 1 + 2*cos(theta), so theta = acos((trace - 1) / 2).
                    rel_trace = torch.einsum("tij,tij->t", prev_R, R_now)
                    angle = torch.acos(((rel_trace - 1) / 2).clamp(-1, 1))  # (T,) radians
                    angle_deg = torch.rad2deg(angle)
                    per_tau_str = ", ".join(f"{a:.2f}°" for a in angle_deg.tolist())
                    print(
                        f"iter {a0}: 20-iter τ change "
                        f"max={angle_deg.max().item():.2f}°, "
                        f"mean={angle_deg.mean().item():.2f}°, "
                        f"per-τ=[{per_tau_str}]"
                    )
                    prev_R = R_now.clone()

            # One stacked all_reduce and one host sync instead of three of each.
            losses = torch.stack([total_loss, consistency_loss, epoch_soft_constraint_loss])
            if self.world_size > 1:
                nvtx.range_push("ddp_allreduce_epoch_metrics")
                dist.all_reduce(losses, dist.ReduceOp.AVG)
                nvtx.range_pop()
            total_loss, consistency_loss, epoch_soft_constraint_loss = (
                losses / len(self.dataloader)
            ).tolist()

            soft_term_names = sorted(epoch_soft_terms)
            epoch_soft_term_values: dict[str, float] = {}
            if soft_term_names:
                soft_term_losses = torch.stack([epoch_soft_terms[name] for name in soft_term_names])
                if self.world_size > 1:
                    dist.all_reduce(soft_term_losses, dist.ReduceOp.AVG)
                epoch_soft_term_values = dict(
                    zip(soft_term_names, (soft_term_losses / len(self.dataloader)).tolist())
                )

            self.step_schedulers(loss=total_loss)
            # TODO: Maybe reorganize the losses so that the order makes sense lol.

            avg_val_loss = None
            avg_val_fg_loss = None
            avg_val_bg_loss = None
            validate_this_epoch = self.val_dataloader is not None and (
                holdout_fraction <= 0.0 or (a0 + 1) % holdout_every == 0
            )
            if validate_this_epoch:
                print("Validating...")
                nvtx.range_push("validation")
                avg_val_loss = self._evaluate_validation_loss(
                    dataloader=self.val_dataloader,
                    num_samples_per_ray=curr_num_samples_per_ray,
                    object_extent=N,
                    loss_func=loss_func,
                    use_bfloat16=use_bfloat16,
                )
                if getattr(self, "val_fg_dataloader", None) is not None:
                    avg_val_fg_loss = self._evaluate_validation_loss(
                        dataloader=self.val_fg_dataloader,
                        num_samples_per_ray=curr_num_samples_per_ray,
                        object_extent=N,
                        loss_func=loss_func,
                        use_bfloat16=use_bfloat16,
                    )
                if getattr(self, "val_bg_dataloader", None) is not None:
                    avg_val_bg_loss = self._evaluate_validation_loss(
                        dataloader=self.val_bg_dataloader,
                        num_samples_per_ray=curr_num_samples_per_ray,
                        object_extent=N,
                        loss_func=loss_func,
                        use_bfloat16=use_bfloat16,
                    )
                nvtx.range_pop()  # validation

            # The three losses were already rank-averaged (and batch-normalized) right
            # after the batch loop; re-reducing identical values here was a redundant
            # all_reduce plus an extra host sync per epoch.
            pbar.set_description(
                f"Reconstruction | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e}"
            )

            self._epoch_losses.append(total_loss)
            self._consistency_losses.append(consistency_loss)
            self.append_learning_rates(self.get_current_lrs())
            self.obj_model._soft_constraint_losses.append(epoch_soft_constraint_loss)
            if not hasattr(self, "_soft_term_losses"):
                self._soft_term_losses = []
            self._soft_term_losses.append(epoch_soft_term_values)
            if avg_val_loss is not None:
                self._val_losses.append(avg_val_loss)

            if self.logger is not None:
                if (
                    self.logger.log_images_every > 0
                    and self.num_epochs % self.logger.log_images_every == 0
                ):
                    pred_full = self.obj_model.obj_view

                    if self.global_rank == 0:
                        self.logger.log_iter_images(
                            pred_volume=pred_full,
                            dataset_model=self.dset,
                            iter=self.num_epochs,
                        )
                    pbar.set_description(
                        f"Reconstruction | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e} | Images Logged"
                    )

                if self.global_rank == 0:
                    self.logger.log_iter(
                        object_model=self.obj_model,
                        iter=self.num_epochs,
                        consistency_loss=consistency_loss,
                        total_loss=total_loss,
                        learning_rates=self.get_current_lrs(),
                        num_samples_per_ray=curr_num_samples_per_ray,
                        val_loss=avg_val_loss if validate_this_epoch else None,
                        val_fg_loss=avg_val_fg_loss,
                        val_bg_loss=avg_val_bg_loss,
                        grad_step=self._grad_steps,
                        soft_terms=epoch_soft_term_values,
                    )
                    phase = getattr(self, "_recon_phase", None)
                    if phase is not None:
                        self.logger.log_scalar(
                            "two_phase/phase",
                            float(phase),
                            self.num_epochs,
                            extra_steps={"grad_step": self._grad_steps},
                        )

                self.logger.flush()
            if not self.verbose:
                if self.global_rank == 0:
                    print(
                        f"Reconstruction Epoch {self.num_epochs} | Loss: {total_loss:.5e}, Consistency Loss: {consistency_loss:.5e}, Soft Constraint Loss: {epoch_soft_constraint_loss:.5e}"
                    )

            # Opt-in per-epoch evaluation hook (e.g. SSIM-vs-ground-truth convergence
            # curves). Note `obj_view` is collective, so the callback must be invoked on
            # every rank; it is the caller's responsibility to guard rank-0-only work.
            if eval_callback is not None and eval_every > 0 and (a0 + 1) % eval_every == 0:
                eval_callback(a0 + 1)
            nvtx.range_pop()  # epoch

        if show_metrics and self.world_size == 1:
            self.plot_losses()

    def reconstruct_two_phase(
        self,
        phase1: TwoPhaseParams | dict,
        num_iter: int,
        optimizer_params: dict,
        scheduler_params: dict | None = None,
        obj_constraints: dict | ObjConstraintsType | None = None,
        eval_callback: Callable[[int], None] | None = None,
        **reconstruct_kwargs,
    ) -> dict[str, int]:
        """TILTED two-phase optimisation (Yi et al. 2023, arXiv 2308.15461, Sec. 4.2).

        Phase 1 trains a fresh CPTilted bottleneck built from ``phase1`` on the same
        dataset and train/holdout split; everything but its rotations is then
        discarded. The rotations are broadcast from rank 0 and loaded into this
        object's KPlanesTILTED, which phase 2 trains with fresh optimizers and
        schedulers from the remaining arguments (as in ``reconstruct``).

        Histories, logger epochs and ``_grad_steps`` run on across both phases;
        snapshots in phase 1 come from the CPTilted model, epochs are logged with a
        ``two_phase/phase`` tag, and ``eval_callback`` receives the epoch index
        counted over both phases. Returns (and stores as ``two_phase_summary``) the
        epoch and gradient-step counts per phase.
        """
        phase1 = TwoPhaseParams.parse(phase1)
        if not isinstance(self.obj_model, ObjectTensorDecomp):
            raise TypeError("reconstruct_two_phase needs an ObjectTensorDecomp object model.")
        phase2_model = _unwrap(self.obj_model.model)
        if not isinstance(phase2_model, KPlanesTILTED):
            raise TypeError(
                f"reconstruct_two_phase needs a KPlanesTILTED model, got {type(phase2_model).__name__}."
            )
        if phase1.num_iter < 1:
            raise ValueError("phase1.num_iter must be >= 1.")
        if optimizer_params is None or phase1.optimizer_params is None:
            raise ValueError("Both phases need optimizer_params; each phase starts fresh optimizers.")
        if reconstruct_kwargs.get("reset_dset") is not None:
            raise ValueError("reset_dset is not supported: both phases must share one dataset.")

        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)
            self.obj_model.constraints = obj_constraints
        phase1_constraints = self._phase1_constraints(phase1.obj_constraints)

        phase1_samples = phase1.num_samples_per_ray
        if phase1_samples is None:
            phase1_samples = reconstruct_kwargs.get("num_samples_per_ray")
            if isinstance(phase1_samples, (list, tuple)):
                raise ValueError(
                    "num_samples_per_ray is a phase-2 schedule; set phase1.num_samples_per_ray."
                )

        model_kwargs = dict(phase1.model_kwargs)
        model_kwargs.setdefault("T", phase2_model.T)
        model_kwargs.setdefault("resolution", list(phase2_model.resolution))
        model_kwargs.setdefault("density_activation", phase2_model.density_activation)
        if model_kwargs["T"] != phase2_model.T:
            raise ValueError(
                f"Phase-1 T={model_kwargs['T']} must equal the phase-2 T={phase2_model.T}."
            )

        phase2_obj = self.obj_model
        phase1_obj = ObjectTensorDecomp.from_model(
            CPTilted(**model_kwargs), shape=phase2_obj.shape, device=self.device
        )

        saved_pose_optimizer = self.dset.optimizer_params
        saved_pose_scheduler = self.dset.scheduler_params
        if "pose" not in phase1.optimizer_params:
            self.dset.remove_optimizer()

        shared_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ("num_samples_per_ray", "pose_warmup_epochs", "show_metrics")
        }
        start_steps, start_epochs = self._grad_steps, self.num_epochs
        self.obj_model = phase1_obj
        self._recon_phase = 1
        try:
            self.reconstruct(
                num_iter=phase1.num_iter,
                optimizer_params=phase1.optimizer_params,
                scheduler_params=phase1.scheduler_params,
                obj_constraints=phase1_constraints,
                num_samples_per_ray=phase1_samples,
                eval_callback=eval_callback,
                **shared_kwargs,
            )
            rotations = _unwrap(phase1_obj.model).extract_tau_state().to(self.device)
        finally:
            self.obj_model = phase2_obj
            self._recon_phase = None
            self.dset.remove_optimizer()
            self.dset.optimizer_params = saved_pose_optimizer
            self.dset.scheduler_params = saved_pose_scheduler

        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.broadcast(rotations, src=0)
        phase2_model.load_tau_state(rotations)
        phase2_obj._soft_constraint_losses.extend(phase1_obj._soft_constraint_losses)
        summary = {
            "phase1_epochs": self.num_epochs - start_epochs,
            "phase1_grad_steps": self._grad_steps - start_steps,
        }
        del phase1_obj

        phase1_epochs = summary["phase1_epochs"]
        phase2_eval = (
            None if eval_callback is None else (lambda epoch: eval_callback(epoch + phase1_epochs))
        )
        self._recon_phase = 2
        try:
            self.reconstruct(
                num_iter=num_iter,
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params,
                eval_callback=phase2_eval,
                **reconstruct_kwargs,
            )
        finally:
            self._recon_phase = None
        summary["phase2_epochs"] = self.num_epochs - start_epochs - phase1_epochs
        summary["phase2_grad_steps"] = self._grad_steps - start_steps - summary["phase1_grad_steps"]
        summary["total_grad_steps"] = self._grad_steps - start_steps
        self.two_phase_summary = summary
        return summary

    def grad_steps_per_epoch(
        self,
        batch_size: int,
        val_fraction: float = 0.0,
        holdout_fraction: float = 0.0,
        holdout_seed: int = 0,
    ) -> int:
        """Gradient steps per epoch on this rank for the given batch size and split."""
        self._setup_recon_dataloaders(
            batch_size, getattr(self, "num_workers", 0), val_fraction, holdout_fraction, holdout_seed
        )
        return len(self.dataloader)

    def _phase1_constraints(
        self, constraints: dict | ObjConstraintsType | None
    ) -> ObjConstraintsType:
        """Phase-1 constraints: explicit (validated) or phase-2's minus plane-only terms."""
        if constraints is None:
            derived = self.obj_model.constraints.copy()
            dropped = [k for k in PHASE1_UNSUPPORTED_CONSTRAINTS if getattr(derived, k, 0.0) > 0]
            for key in dropped:
                setattr(derived, key, 0.0)
            if dropped and self.global_rank == 0:
                print(f"Two-phase: phase 1 (CPTilted line grids) runs without {dropped}.")
            return derived
        if isinstance(constraints, dict):
            constraints = ObjConstraintParams.parse_dict(constraints)
        bad = [k for k in PHASE1_UNSUPPORTED_CONSTRAINTS if getattr(constraints, k, 0.0) > 0]
        if bad:
            raise ValueError(f"CPTilted (phase 1) does not support {bad}; set them to 0.")
        return constraints

    # --- Helper Functions ---

    def _evaluate_validation_loss(
        self,
        *,
        dataloader: DeviceBatchSampler,
        num_samples_per_ray: int,
        object_extent: int,
        loss_func: torch.nn.Module,
        use_bfloat16: bool = True,
    ) -> float | None:
        """Rank-averaged holdout loss, with the same precision handling as training."""
        val_loss = torch.tensor(0.0, device=self.device)
        val_batches = torch.tensor(0.0, device=self.device)
        model_was_training = self.obj_model.model.training
        dset_was_training = self.dset.training

        self.obj_model.model.eval()
        self.dset.eval()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    with _tomography_autocast(self.device, use_bfloat16):
                        all_coords = self.dset.get_coords(
                            batch, object_extent, num_samples_per_ray
                        )
                        all_densities = self.obj_model.forward(all_coords)
                        integrated_densities = self.dset.integrate_rays(
                            all_densities,
                            num_samples_per_ray,
                            len(batch["target_value"]),
                        )
                        target = batch["target_value"].to(self.device, non_blocking=True).float()
                        batch_val_loss = loss_func(integrated_densities.float(), target)
                        val_loss += batch_val_loss.detach()
                        val_batches += 1.0
        finally:
            self.obj_model.model.train(model_was_training)
            self.dset.train(dset_was_training)

        stats = torch.stack([val_loss, val_batches])
        if self.world_size > 1:
            dist.all_reduce(stats, dist.ReduceOp.SUM)
        if stats[1].item() == 0.0:
            return None
        return (stats[0] / stats[1]).item()

    def save_volume(self, path: str = "recon_volume.npz", overwrite: bool = False):
        """
        Saves volume to a numpy array file. Does not save the full Tomography object.
        """
        if self.global_rank == 0:
            if not overwrite and os.path.exists(path):
                raise FileExistsError(
                    f"File {path} already exists. Use overwrite=True to overwrite."
                )
            print(f"Saving volume to {path}")
            np.savez(path, volume=self.obj_model.obj_view)

        if torch.distributed.is_initialized():
            print("Barrier")
            torch.distributed.barrier()

    # Loading and Saving
    @classmethod
    def _recursive_load_from_path(cls, path: str):
        return autoserialize_load(path)

    @classmethod
    def from_file(
        cls,
        path: str,
        device: str = "cpu",
    ) -> Self:
        tomography = cls._recursive_load_from_path(path)
        tomography.to(device)
        tomography._rebuild_dataloader(
            batch_size=tomography.batch_size,
            num_workers=tomography.num_workers,
            val_fraction=tomography.val_fraction,
            holdout_fraction=getattr(tomography, "holdout_fraction", 0.0),
            holdout_seed=getattr(tomography, "holdout_seed", 0),
        )
        return tomography

    def _rebuild_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        val_fraction: float,
        holdout_fraction: float = 0.0,
        holdout_seed: int = 0,
    ):
        """
        Rebuilds the dataloader due to persistent workers error when reloading the object.
        """
        self._setup_recon_dataloaders(
            batch_size, num_workers, val_fraction, holdout_fraction, holdout_seed
        )

    def _setup_recon_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
        val_fraction: float,
        holdout_fraction: float = 0.0,
        holdout_seed: int = 0,
    ):
        """Build the train/val batch iterators.

        INR datasets use ``DeviceBatchSampler`` — batches are built with
        tensor ops on the compute device from a device-resident tilt stack,
        removing the per-pixel ``__getitem__`` / collate / H2D-copy
        dataloader bottleneck (``num_workers`` is ignored on this path).
        Distributed runs shard the same seeded epoch permutation across
        ranks (DistributedSampler semantics; the loop's ``set_epoch`` drives
        reshuffling). Non-INR datasets keep the DataLoader path.
        """
        self.val_fg_dataloader = None
        self.val_bg_dataloader = None
        if isinstance(self.dset, TomographyINRDataset):
            n = len(self.dset)
            if holdout_fraction > 0.0:
                split = build_pixel_holdout_split(
                    self.dset.tilt_stack,
                    holdout_fraction=holdout_fraction,
                    holdout_seed=holdout_seed,
                )
                train_indices = split.train_indices
                val_indices = split.val_indices
                val_fg_indices = split.val_fg_indices
                val_bg_indices = split.val_bg_indices
            else:
                n_val = int(n * val_fraction)
                # Fixed-seed split: identical across DDP ranks (no train/val
                # leakage between ranks) and stable across save/reload, so a
                # resumed run keeps validating on the same held-out pixels.
                split_gen = torch.Generator()
                split_gen.manual_seed(0)
                perm = torch.randperm(n, generator=split_gen)
                train_indices = perm[n_val:]
                val_indices = perm[:n_val]
                val_fg_indices = torch.empty(0, dtype=torch.long)
                val_bg_indices = torch.empty(0, dtype=torch.long)

            ddp = dict(rank=self.global_rank, world_size=self.world_size)
            self.dataloader = DeviceBatchSampler(
                self.dset, batch_size, self.device, indices=train_indices, **ddp
            )
            # The val sampler keeps its own device-resident copy of the tilt
            # stack; acceptable, since val_fraction > 0 is the rare case.
            val = (
                DeviceBatchSampler(
                    self.dset,
                    batch_size,
                    self.device,
                    indices=val_indices,
                    shuffle=False,
                    drop_last=False,
                    **ddp,
                )
                if len(val_indices) > 0
                else None
            )
            self.val_dataloader = val if val is not None and len(val) > 0 else None
            val_fg = (
                DeviceBatchSampler(
                    self.dset,
                    batch_size,
                    self.device,
                    indices=val_fg_indices,
                    shuffle=False,
                    drop_last=False,
                    **ddp,
                )
                if len(val_fg_indices) > 0
                else None
            )
            val_bg = (
                DeviceBatchSampler(
                    self.dset,
                    batch_size,
                    self.device,
                    indices=val_bg_indices,
                    shuffle=False,
                    drop_last=False,
                    **ddp,
                )
                if len(val_bg_indices) > 0
                else None
            )
            self.val_fg_dataloader = val_fg if val_fg is not None and len(val_fg) > 0 else None
            self.val_bg_dataloader = val_bg if val_bg is not None and len(val_bg) > 0 else None
            # The training loop calls set_epoch on self.sampler.
            self.sampler = self.dataloader
            self.val_sampler = None
            return

        self.dataloader, self.sampler, self.val_dataloader, self.val_sampler = (
            self.setup_dataloader(
                self.dset,
                batch_size,
                num_workers=num_workers,
                val_fraction=val_fraction,
            )
        )

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = ["dataloader"],
        compression_level: int | None = 4,
    ) -> None:
        super(Tomography, self).save(
            path=path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
        )

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(10, 4), ncols=2)

        ax[0].plot(self._epoch_losses, label="Total Training Loss")
        if len(self._val_losses) > 0:
            ax[0].plot(self._val_losses, label="Validation Loss")
        ax[0].legend()

        for key, value in self._lrs.items():
            ax[1].plot(value, label=key)

        ax[1].legend()
        ax[0].legend()
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].set_xlabel("Epoch")
        ax[1].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Learning Rate")


class TomographyConventional(TomographyBase):
    """
    Class for handling all conventional tomography reconstruction methods.
    Will also handle choosing the appropriate dataset model to use.
    """

    @classmethod
    def from_models(
        cls,
        dset: TomographyPixDataset,
        obj_model: ObjectPixelated,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
            verbose=verbose,
            _token=cls._token,
        )

    def reconstruct(
        self,
        num_iter: int = 10,
        obj_constraints: dict | ObjConstraintsType | None = None,
        mode: Literal["sirt", "fbp"] = "sirt",
        relaxation: float = 0.25,
        reset: bool = False,
        inline_alignment: bool = False,
        smoothing_sigma: float | None = None,
        show_metrics: bool = False,
    ):
        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)

            self.obj_model.constraints = obj_constraints

        pbar = tqdm(
            range(num_iter),
            desc=f"{mode} Reconstruction | Loss: {0:.4f}",
            disable=not self.verbose,
        )
        if mode == "sirt" or mode == "fbp":
            proj_forward = torch.zeros_like(self.dset.tilt_stack).permute(2, 0, 1)
        else:
            proj_forward = torch.zeros_like(self.dset.tilt_stack)

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        patience = self.dset.tilt_angles.max() // 10
        for iter in pbar:
            proj_forward, loss = self._reconstruction_epoch(
                inline_alignment=inline_alignment,
                mode=mode,
                proj_forward=proj_forward,
                gaussian_kernel=gaussian_kernel,
                relaxation=relaxation,
            )

            pbar.set_description(f"{mode} Reconstruction | Loss: {loss.item():.4f}")

            self._epoch_losses.append(loss.item())

            # Change relaxation parameter if loss greater than last epoch
            if len(self._epoch_losses) > 1 and self._epoch_losses[-1] > self._epoch_losses[-2]:
                if patience == 0:
                    relaxation *= 0.85
                    print(f"Relaxation parameter changed to: {relaxation}")
                    patience = 10
                else:
                    patience -= 1

            if mode == "fbp":
                break

        if show_metrics:
            self.plot_losses()

    # --- Conventional reconstruction method ---
    def _adaptive_relaxation(self, n_power_iter: int = 10) -> float:
        raise NotImplementedError(
            "Adaptive relaxation hasn't been implemented, please input a valid relaxation parameter."
        )

    def _reconstruction_epoch(
        self,
        inline_alignment: bool,
        mode: Literal["sirt", "fbp"],
        proj_forward: torch.Tensor,
        relaxation: float,
        gaussian_kernel: torch.Tensor | None = None,
    ):
        loss = 0

        if relaxation == 0.0:
            relaxation = self._adaptive_relaxation()
            print(f"Adaptive relaxation: {relaxation}")
        if inline_alignment:
            for ind in range(len(self.dset.tilt_angles)):
                im_proj = proj_forward[:, ind, :]
                # proj_forward rows are volume slices (the tilt axis), i.e. the
                # transpose of the stored tilt image -- the same orientation the
                # error term below compares against.
                im_meas = self.dset.forward(ind).target.T  # type: ignore
                shift = torch_phase_cross_correlation(im_proj, im_meas)
                if torch.linalg.norm(shift) <= 32:
                    shifted = torch.fft.ifft2(
                        torch.fft.fft2(im_meas)
                        * torch.exp(
                            -2j
                            * np.pi
                            * (
                                shift[0]
                                * torch.fft.fftfreq(
                                    im_meas.shape[0], device=im_meas.device
                                ).unsqueeze(1)
                                + shift[1]
                                * torch.fft.fftfreq(im_meas.shape[1], device=im_meas.device)
                            )
                        )
                    ).real

                    # Persist the aligned measurement in the tilt stack: the error
                    # term reads the stack, and proj_forward is overwritten by
                    # radon_torch below, so writing the aligned image there
                    # silently discarded the alignment.
                    self.dset.tilt_stack[ind] = shifted.T

        if mode == "sirt" or mode == "fbp":
            proj_forward = radon_torch(
                self.obj_model.obj,
                theta=self.dset.tilt_angles,
                device=self.device,
            )

            error = self.dset.tilt_stack.permute(2, 0, 1) - proj_forward

            correction = iradon_torch(
                error,
                theta=self.dset.tilt_angles,
                device=self.device,
                filter_name="ramp",
                circle=True,
            )

            normalization = iradon_torch(
                torch.ones_like(error),
                theta=self.dset.tilt_angles,
                device=self.device,
                circle=True,
                filter_name=None,
            )

            normalization[normalization == 0] = 1e-6

            correction /= normalization

            self.obj_model.obj += correction * relaxation

            if gaussian_kernel is not None:
                self.obj_model.obj = gaussian_filter_2d_stack(self.obj_model.obj, gaussian_kernel)

        loss = torch.mean(torch.abs(error))

        return proj_forward, loss

    # --- Helper Functions ---

    def plot_losses(self):
        fig, ax = plt.subplots()
        ax.plot(self._epoch_losses)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Reconstruction Loss")
        ax.set_yscale("log")
        plt.show()
