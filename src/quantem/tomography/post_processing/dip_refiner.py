"""Post-hoc 3D-U-Net deep-image-prior refinement of a reconstructed volume.

Freezes the reconstruction, takes its densified volume as the target, and fits an untrained
:class:`~quantem.core.ml.cnn.CNN3d` to it. The convolutional architecture is the prior: it
reaches structured, self-similar content (the atomic lattice) before incoherent shot noise,
so an early-stopped fit denoises. The denoising power comes from weight-sharing
self-similarity, not the input -- hence the input is an ablation axis (``input_mode``).

This is image-domain only: the target is the (noisy) reconstructed volume, so training to
convergence merely reproduces it. The useful result lives in the transient; the caller logs
the full metric history (via ``metric_callback``) and selects a stop post-hoc -- on a phantom
that stop is oracle (max F1 / valley-contrast), exposing how it differs from the PSNR-optimal
stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from quantem.core.ml.cnn import CNN3d
from quantem.core.ml.optimizer_mixin import (
    OptimizerMixin,
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)

MetricCallback = Callable[[np.ndarray], dict[str, float]]

Crop = tuple[int, int, int, int, int, int]


def support_bbox(vol: torch.Tensor, threshold: float, pad: int) -> Crop:
    """Padded bounding box of ``vol > threshold * max`` (full frame if empty)."""
    mask = vol > threshold * float(vol.max())
    nz = torch.nonzero(mask)
    if nz.numel() == 0:
        return (0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2])
    lo = (nz.min(0).values - pad).clamp_min(0)
    hi = nz.max(0).values + 1 + pad
    hi = torch.minimum(hi, torch.tensor(vol.shape))
    return (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))


def crop_and_pad(
    vol: torch.Tensor,
    spatial_mode: str,
    num_layers: int,
    bbox_threshold: float,
    bbox_pad: int,
) -> tuple[torch.Tensor, Crop, tuple[int, int, int], tuple[int, int, int]]:
    """Crop ``vol`` to its support (``bbox``) or keep it whole (``full``), then pad each axis
    up to a multiple of ``2**num_layers`` (the CNN3d pooling requirement).

    Returns ``(target_proc, crop, cropped_shape, full_shape)`` where ``target_proc`` is
    ``(1, Zp, Yp, Xp)`` and ``crop`` indexes the original frame for re-insertion.
    """
    full_shape = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
    crop = (
        support_bbox(vol, bbox_threshold, bbox_pad)
        if spatial_mode == "bbox"
        else (0, full_shape[0], 0, full_shape[1], 0, full_shape[2])
    )
    z0, z1, y0, y1, x0, x1 = crop
    cropped = vol[z0:z1, y0:y1, x0:x1]
    cropped_shape = (int(cropped.shape[0]), int(cropped.shape[1]), int(cropped.shape[2]))
    multiple = 2**num_layers
    pad = [(-s) % multiple for s in cropped_shape]
    # F.pad order is (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi).
    target_proc = F.pad(cropped[None], (0, pad[2], 0, pad[1], 0, pad[0]))
    return target_proc, crop, cropped_shape, full_shape


def to_full_frame(
    proc: torch.Tensor,
    crop: Crop,
    cropped_shape: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    """Un-pad ``proc`` and re-insert it into a zeroed full-frame volume (detached numpy)."""
    return to_full_frame_torch(proc, crop, cropped_shape, full_shape).detach().cpu().numpy()


def to_full_frame_torch(
    proc: torch.Tensor,
    crop: Crop,
    cropped_shape: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Un-pad ``proc`` (1, Zp, Yp, Xp) and re-insert into a zeroed full-frame tensor.

    Gradient-preserving: zero-padding the cropped output back to ``full_shape`` keeps the
    graph intact, so a differentiable forward model can backprop into ``proc``.
    """
    zc, yc, xc = cropped_shape
    z0, z1, y0, y1, x0, x1 = crop
    cropped = proc[0, :zc, :yc, :xc]
    # F.pad order is (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi).
    pad = (x0, full_shape[2] - x1, y0, full_shape[1] - y1, z0, full_shape[0] - z1)
    return F.pad(cropped, pad)


@dataclass
class RefineResult:
    """Outcome of a refinement run.

    Attributes
    ----------
    history : dict[str, list[float]]
        Per-eval scalars: ``iter``, ``loss``, ``lr``, plus every key returned by the
        metric callback.
    best : dict[str, dict]
        For each tracked ``(key, mode)`` selection, ``{"value", "iter", "volume"}`` at the
        best eval. ``volume`` is a full-frame ``(Z, Y, X)`` numpy array.
    final_volume : np.ndarray
        The refined volume at the last iteration, ``(Z, Y, X)``.
    """

    history: dict[str, list[float]]
    best: dict[str, dict]
    final_volume: np.ndarray
    metadata: dict = field(default_factory=dict)


class ObjectDIPRefiner(OptimizerMixin):
    """A 3D-U-Net deep-image-prior denoiser for a single reconstructed volume."""

    DEFAULT_LR = 1e-3

    def __init__(
        self,
        model: CNN3d,
        target: torch.Tensor,
        model_input: torch.Tensor,
        crop: tuple[int, int, int, int, int, int],
        cropped_shape: tuple[int, int, int],
        full_shape: tuple[int, int, int],
        input_noise_std: float,
        device: str,
    ):
        OptimizerMixin.__init__(self)
        self.device = device
        self.model = model.to(device)
        self._target = target.to(device)
        self._model_input = model_input.to(device)
        self._crop = crop
        self._cropped_shape = cropped_shape
        self._full_shape = full_shape
        self._input_noise_std = float(input_noise_std)

    # -- construction ------------------------------------------------------
    @staticmethod
    def default_unet(
        num_layers: int = 3,
        start_filters: int = 16,
        num_per_layer: int = 2,
        dropout: float = 0.0,
        use_skip_connections: bool = True,
        use_batchnorm: bool = True,
    ) -> CNN3d:
        """A real-valued 3D U-Net with a non-negative (ReLU) output."""
        return CNN3d(
            in_channels=1,
            out_channels=1,
            start_filters=start_filters,
            num_layers=num_layers,
            num_per_layer=num_per_layer,
            use_skip_connections=use_skip_connections,
            dtype=torch.float32,
            dropout=dropout,
            activation="relu",
            final_activation="relu",
            use_batchnorm=use_batchnorm,
            mode="real",
        )

    @classmethod
    def from_volume(
        cls,
        target: np.ndarray | torch.Tensor,
        model: CNN3d | None = None,
        input_mode: str = "warm",
        spatial_mode: str = "bbox",
        bbox_threshold: float = 0.05,
        bbox_pad: int = 8,
        input_noise_std: float = 0.0,
        input_scale: float = 0.1,
        num_layers: int = 3,
        start_filters: int = 16,
        device: str | None = None,
        seed: int = 0,
    ) -> "ObjectDIPRefiner":
        """Build a refiner for ``target`` (a densified reconstruction volume).

        Parameters
        ----------
        target : (Z, Y, X) or (1, Z, Y, X)
            The reconstructed volume to refine toward.
        input_mode : {"warm", "random"}
            ``"warm"`` feeds the target volume as the network input (anchored); ``"random"``
            feeds a frozen random code (classic deep image prior).
        spatial_mode : {"bbox", "full"}
            ``"bbox"`` refines only a padded crop around the support (cheap for a mostly
            vacuum nanoparticle); ``"full"`` refines the whole padded volume.
        bbox_threshold : float
            Fraction of the max used to define the support for ``bbox`` cropping.
        bbox_pad : int
            Voxels of padding around the support before size-rounding.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            model = cls.default_unet(num_layers=num_layers, start_filters=start_filters)

        vol = torch.as_tensor(np.asarray(target), dtype=torch.float32)
        if vol.dim() == 4:
            vol = vol[0]

        target_proc, crop, cropped_shape, full_shape = crop_and_pad(
            vol, spatial_mode, num_layers, bbox_threshold, bbox_pad
        )

        if input_mode == "warm":
            model_input = target_proc.clone()
        elif input_mode == "random":
            gen = torch.Generator().manual_seed(seed)
            model_input = torch.randn(target_proc.shape, generator=gen) * input_scale
        else:
            raise ValueError(f"Unknown input_mode: {input_mode!r}")

        return cls(
            model=model,
            target=target_proc,
            model_input=model_input,
            crop=crop,
            cropped_shape=cropped_shape,
            full_shape=full_shape,
            input_noise_std=input_noise_std,
            device=device,
        )

    # -- forward / volume extraction --------------------------------------
    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        # Single PPLR group: the whole U-Net trains at one learning rate.
        return {self.DEFAULT_OPTIMIZER_KEY: list(self.model.parameters())}

    def _forward(self, add_noise: bool) -> torch.Tensor:
        inp = self._model_input
        if add_noise and self._input_noise_std > 0.0:
            inp = inp + torch.randn_like(inp) * self._input_noise_std
        return self.model(inp)  # (1, Zp, Yp, Xp)

    def _to_full_frame(self, proc: torch.Tensor) -> np.ndarray:
        """Un-pad the cropped output and re-insert it into a zeroed full-frame volume."""
        return to_full_frame(proc, self._crop, self._cropped_shape, self._full_shape)

    def current_volume(self) -> np.ndarray:
        """The refined full-frame volume at the current weights (no input noise)."""
        with torch.no_grad():
            return self._to_full_frame(self._forward(add_noise=False))

    def reset(self) -> None:
        """Reinitialize the network weights and drop the optimizer."""
        self.model.reset_weights()
        self.remove_optimizer()

    # -- training ----------------------------------------------------------
    def refine(
        self,
        num_iters: int = 2000,
        optimizer_params: OptimizerParamsType | dict | None = None,
        scheduler_params: SchedulerParamsType | dict | None = None,
        loss_fn: str | Callable = "l2",
        metric_callback: MetricCallback | None = None,
        eval_every: int = 25,
        track_best: tuple[tuple[str, str], ...] = (),
        early_stop: tuple[str, str, int] | None = None,
        verbose: bool = True,
    ) -> RefineResult:
        """Fit the U-Net to the target volume, logging metrics every ``eval_every`` steps.

        Parameters
        ----------
        track_best : tuple of (key, mode)
            Metric keys (from ``metric_callback``) to checkpoint the best full-frame volume
            for; ``mode`` is ``"max"`` or ``"min"``. E.g. ``(("f1", "max"), ("psnr", "max"))``.
        early_stop : (key, mode, patience) or None
            Stop if the metric ``key`` has not improved for ``patience`` evals. ``None`` runs
            the full budget (the default for phantom experiments, which select post-hoc).
        """
        self.model.train()
        self.set_optimizer(optimizer_params or OptimizerParams.Adam(lr=self.DEFAULT_LR))
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iter=num_iters)

        history: dict[str, list[float]] = {"iter": [], "loss": [], "lr": []}
        best: dict[str, dict] = {}
        es_key, es_mode, es_patience = early_stop or (None, None, 0)
        es_best = -np.inf if es_mode == "max" else np.inf
        es_since = 0

        for it in range(num_iters):
            self.zero_optimizer_grad()
            out = self._forward(add_noise=True)
            loss = self._loss(loss_fn, out, self._target)
            loss.backward()
            self.step_optimizer()
            self.step_scheduler(loss.item())

            if it % eval_every == 0 or it == num_iters - 1:
                history["iter"].append(it)
                history["loss"].append(float(loss.item()))
                history["lr"].append(self.get_current_lr())
                vol = (
                    self.current_volume() if (metric_callback is not None or track_best) else None
                )
                if metric_callback is not None:
                    assert vol is not None
                    metrics = metric_callback(vol)
                    for k, v in metrics.items():
                        history.setdefault(k, []).append(float(v))
                    for key, mode in track_best:
                        if key in metrics:
                            self._update_best(best, key, mode, metrics[key], it, vol)
                    if es_key is not None and es_key in metrics:
                        improved = (
                            metrics[es_key] > es_best
                            if es_mode == "max"
                            else metrics[es_key] < es_best
                        )
                        if improved:
                            es_best, es_since = metrics[es_key], 0
                        else:
                            es_since += 1
                if verbose:
                    extra = "" if metric_callback is None else self._fmt_metrics(history)
                    print(f"  [dip] iter {it:5d}  loss {loss.item():.4e}{extra}")
                if es_key is not None and es_since >= es_patience > 0:
                    if verbose:
                        print(f"  [dip] early stop at iter {it} ({es_key} stalled)")
                    break

        return RefineResult(
            history=history,
            best=best,
            final_volume=self.current_volume(),
            metadata={"crop": self._crop, "cropped_shape": self._cropped_shape},
        )

    @staticmethod
    def _loss(loss_fn: str | Callable, out: torch.Tensor, target: torch.Tensor):
        if callable(loss_fn):
            return loss_fn(out, target)
        if loss_fn == "l2":
            return F.mse_loss(out, target)
        if loss_fn == "l1":
            return F.l1_loss(out, target)
        raise ValueError(f"Unknown loss_fn: {loss_fn!r}")

    @staticmethod
    def _update_best(best, key, mode, value, it, vol):
        rec = best.get(key)
        better = rec is None or (value > rec["value"] if mode == "max" else value < rec["value"])
        if better:
            best[key] = {"value": float(value), "iter": it, "volume": vol.copy()}

    @staticmethod
    def _fmt_metrics(history: dict[str, list[float]]) -> str:
        keys = [k for k in ("f1", "psnr", "valley_contrast") if k in history]
        return "".join(f"  {k} {history[k][-1]:.4f}" for k in keys)
