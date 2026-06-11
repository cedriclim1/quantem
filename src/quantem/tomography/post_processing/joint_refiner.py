"""Data-coupled CNN-head refinement of a reconstructed tomography volume.

Where :class:`~quantem.tomography.post_processing.dip_refiner.ObjectDIPRefiner` fits a CNN to
the (noisy) reconstructed volume in image space -- so its only honest stop is an oracle on
ground truth -- this refiner couples the CNN to the *measurements*. A frozen INR volume
``V0`` is passed through a 3D-U-Net, ``V = CNN(V0)``, and ``V`` is reprojected through the
reconstruction's own ray geometry and compared to the measured tilts. Training stops on the
same data loss evaluated over **held-out tilts** -- a ground-truth-free early-stop signal.

The reprojection reuses ``dset.get_coords`` + ``dset.integrate_rays`` (the INR forward),
sampling the dense ``V`` with ``grid_sample`` instead of querying the network. Geometry,
pose/shift parameters, the angle-sign convention and the target normalization are therefore
inherited from ``dset`` -- there is no second projection model to keep in sync.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from quantem.core.ml.optimizer_mixin import (
    OptimizerMixin,
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from quantem.tomography.post_processing.dip_refiner import (
    MetricCallback,
    ObjectDIPRefiner,
    RefineResult,
    crop_and_pad,
    to_full_frame,
    to_full_frame_torch,
)


def split_tilt_indices(
    n_angles: int,
    val_indices: np.ndarray | None = None,
    val_stride: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split tilt (projection) indices into train and held-out validation sets.

    Held-out tilts are taken every ``val_stride``-th angle so the validation data loss
    measures genuine angular interpolation (spread across the tilt range), not a random
    pixel subset. Pass ``val_indices`` to choose them explicitly.
    """
    idx = np.arange(int(n_angles))
    val = np.asarray(val_indices, dtype=int) if val_indices is not None else idx[::val_stride]
    train = np.setdiff1d(idx, val)
    return train, val


def _tilt_pixel_indices(dset, tilt_ids: np.ndarray) -> np.ndarray:
    """Flattened ray-pixel indices (into ``dset``) belonging to the given tilts.

    A projection's pixels occupy a contiguous block of size ``H*W``; projections must be
    square (``H == W``), which holds for these phantom tilt series.
    """
    h, w = int(dset.tilt_stack.shape[1]), int(dset.tilt_stack.shape[2])
    if h != w:
        raise ValueError(f"JointRefiner expects square projections, got ({h}, {w}).")
    block = h * w
    return np.concatenate([np.arange(p * block, (p + 1) * block) for p in tilt_ids])


def reproject_volume(
    volume: torch.Tensor,
    dset,
    batch: dict,
    N: int,
    num_samples_per_ray: int,
    align_corners: bool = True,
) -> torch.Tensor:
    """Reproject a dense ``(Z, Y, X)`` volume through the INR ray forward for one batch.

    Samples ``volume`` at the dataset's (pose-aware) ray coordinates with ``grid_sample``
    and integrates along each ray, returning predicted ray values aligned with
    ``batch["target_value"]``. Differentiable in ``volume``.
    """
    with torch.no_grad():
        coords = dset.get_coords(batch, N, num_samples_per_ray)  # (K, 3), [-1,1], (x,y,z)
    grid = coords.view(1, 1, 1, -1, 3)
    vol5 = volume[None, None]  # (1, 1, Z, Y, X) = (1, 1, D, H, W)
    samp = F.grid_sample(
        vol5, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )
    densities = samp.view(-1)
    n_rays = len(batch["target_value"])
    return dset.integrate_rays(densities, num_samples_per_ray, n_rays)


def _cycle(loader: DataLoader):
    """Infinite iterator over a DataLoader (re-shuffles each pass)."""
    while True:
        yield from loader


class JointRefiner(OptimizerMixin):
    """A 3D-U-Net head trained to make a reconstructed volume data-consistent."""

    DEFAULT_LR = 1e-3

    def __init__(
        self,
        model,
        model_input: torch.Tensor,
        crop,
        cropped_shape,
        full_shape,
        dset,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        N: int,
        num_samples_per_ray: int,
        batch_size: int,
        input_noise_std: float,
        device: str,
    ):
        OptimizerMixin.__init__(self)
        self.device = device
        self.model = model.to(device)
        self._model_input = model_input.to(device)
        self._crop = crop
        self._cropped_shape = cropped_shape
        self._full_shape = full_shape
        self._dset = dset
        self._train_indices = train_indices
        self._val_indices = val_indices
        self._N = N
        self._n_samples = num_samples_per_ray
        self._batch_size = batch_size
        self._input_noise_std = float(input_noise_std)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_volume_and_dset(
        cls,
        v0: np.ndarray | torch.Tensor,
        dset,
        *,
        val_indices: np.ndarray | None = None,
        val_stride: int = 5,
        input_mode: str = "warm",
        spatial_mode: str = "bbox",
        bbox_threshold: float = 0.05,
        bbox_pad: int = 8,
        input_scale: float = 0.1,
        input_noise_std: float = 0.0,
        num_layers: int = 3,
        start_filters: int = 16,
        batch_size: int = 4096,
        num_samples_per_ray: int | None = None,
        max_val_rays: int = 200_000,
        device: str | None = None,
        seed: int = 0,
    ) -> "JointRefiner":
        """Build a data-coupled refiner from a frozen INR volume and the recon's ``dset``.

        Parameters
        ----------
        v0 : (Z, Y, X) or (1, Z, Y, X)
            The reconstructed (frozen) volume to refine.
        dset : TomographyINRDataset
            The reconstruction's dataset (``tomo.dset``) -- carries tilts, angles, learned
            poses and the target normalization. Reused verbatim for reprojection.
        val_stride : int
            Hold out every ``val_stride``-th tilt for the early-stop signal.
        input_mode : {"warm", "random"}
            ``"warm"`` feeds ``v0`` as the CNN input; ``"random"`` a frozen random code.
        spatial_mode : {"bbox", "full"}
            Crop the CNN to the support (cheap) or run it on the whole padded volume.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = ObjectDIPRefiner.default_unet(num_layers=num_layers, start_filters=start_filters)

        vol = torch.as_tensor(np.asarray(v0), dtype=torch.float32)
        if vol.dim() == 4:
            vol = vol[0]

        proc, crop, cropped_shape, full_shape = crop_and_pad(
            vol, spatial_mode, num_layers, bbox_threshold, bbox_pad
        )
        if input_mode == "warm":
            model_input = proc.clone()
        elif input_mode == "random":
            gen = torch.Generator().manual_seed(seed)
            model_input = torch.randn(proc.shape, generator=gen) * input_scale
        else:
            raise ValueError(f"Unknown input_mode: {input_mode!r}")

        n_angles = int(dset.tilt_stack.shape[0])
        train_ids, val_ids = split_tilt_indices(n_angles, val_indices, val_stride)
        if train_ids.size == 0 or val_ids.size == 0:
            raise ValueError("Tilt split left an empty train or val set; lower val_stride.")
        train_px = _tilt_pixel_indices(dset, train_ids)
        val_px = _tilt_pixel_indices(dset, val_ids)
        if val_px.size > max_val_rays:
            rng = np.random.default_rng(seed)
            val_px = np.sort(rng.choice(val_px, size=max_val_rays, replace=False))

        N = int(max(dset.tilt_stack.shape))
        n_samples = num_samples_per_ray or N

        return cls(
            model=model,
            model_input=model_input,
            crop=crop,
            cropped_shape=cropped_shape,
            full_shape=full_shape,
            dset=dset,
            train_indices=train_px,
            val_indices=val_px,
            N=N,
            num_samples_per_ray=n_samples,
            batch_size=batch_size,
            input_noise_std=input_noise_std,
            device=device,
        )

    # -- forward / volume extraction --------------------------------------
    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        return {self.DEFAULT_OPTIMIZER_KEY: list(self.model.parameters())}

    def _forward(self, add_noise: bool) -> torch.Tensor:
        inp = self._model_input
        if add_noise and self._input_noise_std > 0.0:
            inp = inp + torch.randn_like(inp) * self._input_noise_std
        return self.model(inp)  # (1, Zp, Yp, Xp)

    def _volume(self, add_noise: bool) -> torch.Tensor:
        """Full-frame (Z, Y, X) refined volume tensor with the graph intact."""
        return to_full_frame_torch(
            self._forward(add_noise), self._crop, self._cropped_shape, self._full_shape
        )

    def current_volume(self) -> np.ndarray:
        """The refined full-frame volume at the current weights (numpy, detached)."""
        with torch.no_grad():
            return to_full_frame(
                self._forward(add_noise=False), self._crop, self._cropped_shape, self._full_shape
            )

    # -- training ----------------------------------------------------------
    def _val_data_loss(self) -> float:
        """Mean reprojection MSE over the held-out tilts (no grad)."""
        loader = DataLoader(
            Subset(self._dset, self._val_indices.tolist()),
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
        )
        self._dset.eval()
        with torch.no_grad():
            vol = self._volume(add_noise=False)
            total, count = 0.0, 0
            for batch in loader:
                pred = reproject_volume(vol, self._dset, batch, self._N, self._n_samples)
                target = batch["target_value"].to(self.device, non_blocking=True).float()
                n = len(target)
                total += F.mse_loss(pred, target).item() * n
                count += n
        return total / max(count, 1)

    def refine(
        self,
        num_iters: int = 2000,
        optimizer_params: OptimizerParamsType | dict | None = None,
        scheduler_params: SchedulerParamsType | dict | None = None,
        metric_callback: MetricCallback | None = None,
        eval_every: int = 25,
        track_best: tuple[tuple[str, str], ...] = (("val_data", "min"),),
        early_stop: tuple[str, str, int] | None = ("val_data", "min", 8),
        verbose: bool = True,
    ) -> RefineResult:
        """Train the CNN head against the measured tilts; early-stop on held-out tilts.

        Each iteration draws one minibatch of training rays, rebuilds ``V = CNN(V0)``,
        reprojects it for those rays and steps on the data MSE. Every ``eval_every`` steps
        the held-out-tilt data loss (``val_data``) is recorded and drives ``track_best`` /
        ``early_stop``; an optional ``metric_callback`` adds oracle metrics for analysis.
        """
        self.model.train()
        self.set_optimizer(optimizer_params or OptimizerParams.Adam(lr=self.DEFAULT_LR))
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iter=num_iters)

        train_loader = DataLoader(
            Subset(self._dset, self._train_indices.tolist()),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        train_iter = _cycle(train_loader)

        history: dict[str, list[float]] = {"iter": [], "loss": [], "lr": []}
        best: dict[str, dict] = {}
        es_key, es_mode, es_patience = early_stop or (None, None, 0)
        es_best = -np.inf if es_mode == "max" else np.inf
        es_since = 0

        for it in range(num_iters):
            self._dset.train()
            self.zero_optimizer_grad()
            batch = next(train_iter)
            vol = self._volume(add_noise=True)
            pred = reproject_volume(vol, self._dset, batch, self._N, self._n_samples)
            target = batch["target_value"].to(self.device, non_blocking=True).float()
            loss = F.mse_loss(pred, target)
            loss.backward()
            self.step_optimizer()
            self.step_scheduler(loss.item())

            if it % eval_every == 0 or it == num_iters - 1:
                history["iter"].append(it)
                history["loss"].append(float(loss.item()))
                history["lr"].append(self.get_current_lr())
                history.setdefault("val_data", []).append(self._val_data_loss())
                vol_np = (
                    self.current_volume() if metric_callback is not None or track_best else None
                )
                if metric_callback is not None:
                    assert vol_np is not None
                    for k, v in metric_callback(vol_np).items():
                        history.setdefault(k, []).append(float(v))
                for key, mode in track_best:
                    if key in history and history[key]:
                        ObjectDIPRefiner._update_best(
                            best, key, mode, history[key][-1], it, vol_np
                        )
                if es_key is not None and es_key in history and history[es_key]:
                    cur = history[es_key][-1]
                    improved = cur > es_best if es_mode == "max" else cur < es_best
                    if improved:
                        es_best, es_since = cur, 0
                    else:
                        es_since += 1
                if verbose:
                    extra = (
                        "" if metric_callback is None else ObjectDIPRefiner._fmt_metrics(history)
                    )
                    vd = history["val_data"][-1]
                    print(f"  [joint] iter {it:5d}  loss {loss.item():.4e}  val {vd:.4e}{extra}")
                if es_key is not None and es_since >= es_patience > 0:
                    if verbose:
                        print(f"  [joint] early stop at iter {it} ({es_key} stalled)")
                    break

        return RefineResult(
            history=history,
            best=best,
            final_volume=self.current_volume(),
            metadata={"crop": self._crop, "cropped_shape": self._cropped_shape},
        )
