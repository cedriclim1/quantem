"""Tests for the post-hoc 3D-U-Net DIP refiner."""

import numpy as np
import pytest

from quantem.tomography.phantoms import make_au_nanoparticle
from quantem.tomography.post_processing import ObjectDIPRefiner, detection_f1, psnr_mse


def _metric_callback(ph):
    def cb(vol):
        det = detection_f1(vol, ph.atom_coords, voxel_size=ph.voxel_size, nn_dist_ang=ph.nn_dist)
        psnr, mse = psnr_mse(vol, ph.volume)
        return {"f1": det.f1, "psnr": psnr, "mse": mse}

    return cb


@pytest.mark.parametrize("input_mode", ["warm", "random"])
@pytest.mark.parametrize("spatial_mode", ["bbox", "full"])
def test_refine_smoke_cpu(input_mode, spatial_mode):
    ph = make_au_nanoparticle(n_vox=32, voxel_size=0.3, diameter_ang=8.0)
    refiner = ObjectDIPRefiner.from_volume(
        ph.volume,
        input_mode=input_mode,
        spatial_mode=spatial_mode,
        num_layers=2,
        start_filters=4,
        device="cpu",
    )
    res = refiner.refine(
        num_iters=4,
        eval_every=2,
        metric_callback=_metric_callback(ph),
        track_best=(("f1", "max"), ("psnr", "max")),
        verbose=False,
    )
    assert len(res.history["iter"]) >= 2
    assert all(np.isfinite(res.history["loss"]))
    assert res.final_volume.shape == ph.volume.shape
    assert "f1" in res.best
    assert res.best["f1"]["volume"].shape == ph.volume.shape


def test_bbox_padding_is_pool_divisible():
    ph = make_au_nanoparticle(n_vox=40, voxel_size=0.3, diameter_ang=8.0)
    refiner = ObjectDIPRefiner.from_volume(
        ph.volume, spatial_mode="bbox", num_layers=3, device="cpu"
    )
    _, zp, yp, xp = refiner._target.shape
    assert zp % 8 == 0 and yp % 8 == 0 and xp % 8 == 0  # multiples of 2**num_layers
    assert refiner.current_volume().shape == ph.volume.shape


def test_reset_drops_optimizer():
    ph = make_au_nanoparticle(n_vox=32, voxel_size=0.3, diameter_ang=8.0)
    refiner = ObjectDIPRefiner.from_volume(ph.volume, num_layers=2, start_filters=4, device="cpu")
    refiner.refine(num_iters=2, eval_every=1, verbose=False)
    assert refiner.has_optimizer()
    refiner.reset()
    assert not refiner.has_optimizer()
