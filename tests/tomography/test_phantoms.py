"""Tests for the synthetic Au-nanoparticle phantom generator."""

import numpy as np
from scipy.spatial import cKDTree

from quantem.tomography.phantoms import (
    AuPhantom,
    add_poisson_noise,
    make_au_nanoparticle,
    project_au_phantom,
    project_volume,
)


def test_make_au_nanoparticle_geometry():
    ph = make_au_nanoparticle(n_vox=48, voxel_size=0.3, diameter_ang=12.0)
    assert isinstance(ph, AuPhantom)
    assert ph.volume.shape == (48, 48, 48)
    assert ph.volume.dtype == np.float32
    assert np.isfinite(ph.volume).all()
    assert ph.volume.min() >= 0.0
    assert ph.n_atoms > 0

    coords = ph.atom_coords
    assert coords.shape[1] == 3
    assert (coords >= 0).all()
    assert (coords < np.array(ph.volume.shape)).all()

    # FCC nearest-neighbor spacing should match nn_dist_vox.
    nn = cKDTree(coords).query(coords, k=2)[0][:, 1]
    assert abs(float(np.median(nn)) - ph.nn_dist_vox) < 0.5


def test_projection_shape_and_consistency():
    ph = make_au_nanoparticle(n_vox=32, voxel_size=0.5, diameter_ang=10.0)
    angles = np.linspace(-60, 60, 5).astype(np.float32)
    proj = project_au_phantom(ph, angles)
    assert proj.shape == (5, 32, 32)
    assert np.isfinite(proj).all()
    assert proj.min() >= 0.0
    # project_au_phantom is just project_volume on the phantom's array.
    assert np.allclose(proj, project_volume(ph.volume, angles))


def test_add_poisson_noise():
    ph = make_au_nanoparticle(n_vox=32, voxel_size=0.5, diameter_ang=10.0)
    proj = project_au_phantom(ph, np.array([0.0, 30.0], dtype=np.float32))

    # dose=None is a clean passthrough.
    assert np.array_equal(add_poisson_noise(proj, dose=None), proj.astype(np.float32))

    noisy = add_poisson_noise(proj, dose=20.0, rng=0)
    assert noisy.shape == proj.shape
    assert noisy.min() >= 0.0
    assert not np.allclose(noisy, proj)
    # Reproducible for a fixed seed.
    assert np.allclose(noisy, add_poisson_noise(proj, dose=20.0, rng=0))
