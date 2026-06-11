"""Tests for the ground-truth-anchored atomicity metrics."""

import numpy as np

from quantem.tomography.phantoms import make_au_nanoparticle
from quantem.tomography.post_processing import (
    detection_f1,
    lattice_peak_snr,
    psnr_mse,
    score_volume,
    valley_contrast,
)
from quantem.tomography.post_processing.atomicity import match_atoms


def _phantom():
    return make_au_nanoparticle(n_vox=48, voxel_size=0.3, diameter_ang=12.0)


def test_detection_perfect_on_clean():
    ph = _phantom()
    m = detection_f1(ph.volume, ph.atom_coords, voxel_size=ph.voxel_size, nn_dist_ang=ph.nn_dist)
    assert m.f1 == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.rmse_ang < 0.1
    assert m.n_matched == ph.n_atoms


def test_missing_atom_lowers_recall():
    ph = _phantom()
    vol = ph.volume.copy()
    z, y, x = ph.atom_coords[ph.n_atoms // 2].astype(int)
    vol[z - 3 : z + 4, y - 3 : y + 4, x - 3 : x + 4] = 0.0  # erase one column
    m = detection_f1(vol, ph.atom_coords, voxel_size=ph.voxel_size, nn_dist_ang=ph.nn_dist)
    assert m.recall < 1.0


def test_spurious_peak_lowers_precision():
    ph = _phantom()
    vol = ph.volume.copy()
    vol[2, 2, 2] = 1.0  # bright voxel in vacuum, far from any real column
    m = detection_f1(vol, ph.atom_coords, voxel_size=ph.voxel_size, nn_dist_ang=ph.nn_dist)
    assert m.precision < 1.0


def test_match_respects_gate():
    detected = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    gt = np.array([[0.1, 0.0, 0.0], [50.0, 50.0, 50.0]])
    md, mg = match_atoms(detected, gt, gate_vox=1.0)
    assert len(mg) == 1
    assert md[0] == 0 and mg[0] == 0


def test_valley_contrast_high_on_clean():
    ph = _phantom()
    vc = valley_contrast(
        ph.volume, ph.atom_coords, voxel_size=ph.voxel_size, nn_dist_ang=ph.nn_dist
    )
    assert 0.5 < vc <= 1.0


def test_psnr_mse_identity():
    ph = _phantom()
    psnr, mse = psnr_mse(ph.volume, ph.volume)
    assert mse == 0.0
    assert np.isinf(psnr)


def test_lattice_peak_snr_positive():
    ph = _phantom()
    snr = lattice_peak_snr(ph.volume, voxel_size=ph.voxel_size)
    assert len(snr) == 2
    assert all(v > 1.0 for v in snr.values())


def test_score_volume_has_expected_keys():
    ph = _phantom()
    sc = score_volume(
        ph.volume, ph.atom_coords, ph.volume, voxel_size=ph.voxel_size, nn_dist_ang=ph.nn_dist
    )
    for key in ("f1", "precision", "recall", "rmse_ang", "valley_contrast", "psnr", "mse"):
        assert key in sc
