"""Tests for the data-coupled JointRefiner and the volume-prior reconstruction hook.

The reprojection helper, the tilt split and the JointRefiner smoke run are CPU-fast (they
need only a phantom and a dataset, not a trained INR). The ``track_best_val`` checkpoint
exercises the full INR reconstruction path, so it is gated behind ``requires_gpu`` + ``slow``
like the other learned-reconstruction tests.
"""

import numpy as np
import pytest
import torch

from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.phantoms import make_au_nanoparticle, project_volume
from quantem.tomography.post_processing import JointRefiner, reproject_volume, split_tilt_indices
from quantem.tomography.post_processing.joint_refiner import _tilt_pixel_indices

from .conftest import requires_gpu


def _tiny_dset(n_vox=16, n_tilts=7, device="cpu"):
    ph = make_au_nanoparticle(n_vox=n_vox, voxel_size=0.5, diameter_ang=5.0)
    angles = np.linspace(-60, 60, n_tilts).astype(np.float32)
    series = project_volume(ph.volume, angles, device=device)
    dset = TomographyINRDataset.from_data(series, angles)
    dset.to(device)
    return ph, dset


def _full_tilt_batch(dset, tilt_idx):
    """Collate every pixel of one projection into a single batch dict."""
    h, w = int(dset.tilt_stack.shape[1]), int(dset.tilt_stack.shape[2])
    idx = range(tilt_idx * h * w, (tilt_idx + 1) * h * w)
    items = [dset[i] for i in idx]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}, h, w


def test_reproject_matches_dataset_targets():
    """Reprojecting the phantom reproduces the measured tilt pattern (axis convention)."""
    ph, dset = _tiny_dset()
    N = max(dset.tilt_stack.shape)
    batch, h, w = _full_tilt_batch(dset, 3)
    V = torch.as_tensor(ph.volume, dtype=torch.float32)
    pred = reproject_volume(V, dset, batch, N, N).reshape(h, w).detach().numpy()
    meas = batch["target_value"].numpy().reshape(h, w)
    # Geometry/axis correctness is scale-invariant: the spatial patterns must align.
    corr = np.corrcoef(pred.ravel(), meas.ravel())[0, 1]
    assert corr > 0.99


def test_reproject_is_differentiable():
    _, dset = _tiny_dset()
    N = max(dset.tilt_stack.shape)
    batch, _, _ = _full_tilt_batch(dset, 2)
    V = torch.zeros((16, 16, 16), requires_grad=True)
    reproject_volume(V + 0.5, dset, batch, N, N).sum().backward()
    assert V.grad is not None and torch.isfinite(V.grad).all() and V.grad.abs().sum() > 0


def test_split_tilt_indices():
    train, val = split_tilt_indices(10, val_stride=3)
    assert set(val.tolist()) == {0, 3, 6, 9}
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert sorted(train.tolist() + val.tolist()) == list(range(10))

    train2, val2 = split_tilt_indices(10, val_indices=np.array([1, 5]))
    assert set(val2.tolist()) == {1, 5}


def test_tilt_pixel_indices_are_contiguous_blocks():
    _, dset = _tiny_dset(n_vox=16, n_tilts=5)
    block = dset.tilt_stack.shape[1] * dset.tilt_stack.shape[2]
    px = _tilt_pixel_indices(dset, np.array([2]))
    assert px.tolist() == list(range(2 * block, 3 * block))


def test_joint_refiner_smoke_cpu():
    ph, dset = _tiny_dset(n_vox=16, n_tilts=7)
    refiner = JointRefiner.from_volume_and_dset(
        ph.volume,
        dset,
        val_stride=3,
        num_layers=2,
        start_filters=4,
        batch_size=128,
        num_samples_per_ray=16,
        max_val_rays=2000,
        device="cpu",
    )
    # bbox crop padded to a multiple of 2**num_layers.
    _, zp, yp, xp = refiner._model_input.shape
    assert zp % 4 == 0 and yp % 4 == 0 and xp % 4 == 0
    res = refiner.refine(num_iters=4, eval_every=2, verbose=False)
    assert len(res.history["iter"]) >= 2
    assert all(np.isfinite(res.history["loss"]))
    assert all(np.isfinite(res.history["val_data"]))
    assert res.final_volume.shape == ph.volume.shape
    assert "val_data" in res.best
    assert res.best["val_data"]["volume"].shape == ph.volume.shape


def test_volume_prior_loss(torch_device):
    from quantem.core.ml.models.kplanes import KPlanesTILTED
    from quantem.tomography.object_models import ObjectTensorDecomp

    n = 12
    model = KPlanesTILTED(M_features=2, resolution=(n, n, n), multiscale_res_multipliers=[1], T=2)
    obj = ObjectTensorDecomp.from_model(model, shape=(n, n, n), device=torch_device)
    coords = torch.rand(500, 3, device=torch_device) * 2 - 1

    ref_self = torch.as_tensor(obj.obj_view[0], dtype=torch.float32, device=torch_device)
    rng = torch.Generator(device="cpu").manual_seed(0)
    ref_rand = torch.randn(ref_self.shape, generator=rng).to(torch_device)

    assert float(obj.volume_prior_loss(coords, ref_self, 0.0)) == 0.0
    l_self = obj.volume_prior_loss(coords, ref_self, 1.0).item()
    l_rand = obj.volume_prior_loss(coords, ref_rand, 1.0).item()
    assert np.isfinite(l_self) and l_self < l_rand


@requires_gpu
@pytest.mark.slow
def test_track_best_val_checkpoint():
    from quantem.core.ml.models.kplanes import KPlanesTILTED
    from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams
    from quantem.tomography.object_models import ObjectTensorDecomp
    from quantem.tomography.tomography import Tomography

    ph, dset = _tiny_dset(n_vox=24, n_tilts=9, device="cuda:0")
    n = ph.volume.shape[1]
    model = KPlanesTILTED(M_features=2, resolution=(n, n, n), multiscale_res_multipliers=[1], T=2)
    obj = ObjectTensorDecomp.from_model(model, shape=(n, n, n), device="cuda:0")
    tomo = Tomography.from_models(dset=dset, obj_model=obj, device="cuda:0", verbose=False)
    tomo.reconstruct(
        optimizer_params={
            "object": {
                "grids": OptimizerParams.Adam(lr=1e-2),
                "sigma_net": OptimizerParams.Adam(lr=1e-3),
                "so3": OptimizerParams.Adam(lr=1e-2),
            },
            "pose": OptimizerParams.Adam(lr=1e-2),
        },
        scheduler_params={"object": SchedulerParams.CosineAnnealing(T_max=6)},
        num_iter=6,
        batch_size=256,
        num_samples_per_ray=20,
        num_workers=2,
        val_fraction=0.2,
        track_best_val=True,
    )
    assert tomo.val_losses.size == 6
    assert tomo.best_val_epoch is not None
    assert tomo.best_val_loss == pytest.approx(float(tomo.val_losses.min()))
    # Restoring the checkpoint must not raise and yields a finite volume.
    tomo.load_best_val_state()
    vol = tomo.obj_model.obj_view
    assert np.all(np.isfinite(vol))
