"""Synthetic FCC gold nanoparticle phantom for tomography experiments.

Builds a crystalline Au nanoparticle as a dense volume of Gaussian atoms with a known
ground-truth atom catalog, projects it into a tilt series with quantem's own ZXZ forward
model (the same geometry the reconstructors assume), and optionally adds Poisson (shot)
noise to the projections.

The ground-truth atom catalog (``AuPhantom.atom_coords``) is what downstream atomicity
metrics score against, so it is returned alongside the volume rather than re-detected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from quantem.tomography.utils import rot_ZXZ

# FCC fractional basis: 4 atoms per conventional cubic cell.
_FCC_BASIS = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    dtype=np.float64,
)


@dataclass
class AuPhantom:
    """A gold-nanoparticle phantom plus its ground-truth atom catalog.

    Attributes
    ----------
    volume : (Z, Y, X) float32
        Dense voxelized density.
    atom_coords : (n_atoms, 3) float32
        Ground-truth atom centers in voxel coordinates, ordered (z, y, x).
    voxel_size : float
        Angstrom per voxel.
    lattice_a : float
        FCC lattice constant (Angstrom).
    nn_dist : float
        Nearest-neighbor distance (Angstrom), ``lattice_a / sqrt(2)`` for FCC.
    """

    volume: np.ndarray
    atom_coords: np.ndarray
    voxel_size: float
    lattice_a: float
    nn_dist: float

    @property
    def n_atoms(self) -> int:
        return int(self.atom_coords.shape[0])

    @property
    def nn_dist_vox(self) -> float:
        """Nearest-neighbor distance expressed in voxels."""
        return self.nn_dist / self.voxel_size


def _fcc_atom_centers(diameter_ang: float, lattice_a: float, center_ang: np.ndarray) -> np.ndarray:
    """Atom centers (Angstrom, ordered z/y/x) of an FCC sphere about ``center_ang``."""
    radius = diameter_ang / 2.0
    n_cells = int(np.ceil(radius / lattice_a)) + 1
    idx = np.arange(-n_cells, n_cells + 1)
    gz, gy, gx = np.meshgrid(idx, idx, idx, indexing="ij")
    cells = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
    # Every lattice point = cell origin + basis offset (in fractional cell units).
    pts = (cells[:, None, :] + _FCC_BASIS[None, :, :]).reshape(-1, 3)
    coords = pts * lattice_a + np.asarray(center_ang, dtype=np.float64)
    dist = np.linalg.norm(coords - np.asarray(center_ang, dtype=np.float64), axis=1)
    return coords[dist <= radius]


def make_au_nanoparticle(
    n_vox: int = 300,
    voxel_size: float = 0.3,
    diameter_ang: float = 60.0,
    lattice_a: float = 4.078,
    atom_sigma_ang: float = 0.45,
    peak_intensity: float = 1.0,
) -> AuPhantom:
    """Build a roughly spherical FCC gold nanoparticle phantom.

    Atoms are placed on an FCC lattice (``lattice_a`` = 4.078 Angstrom for Au, giving a
    nearest-neighbor distance of 2.88 Angstrom) inside a centered sphere of diameter
    ``diameter_ang``, and splatted as isotropic Gaussians into an ``n_vox**3`` volume.
    Each atom is rendered only within a +/-3 sigma local window (never as a dense
    ``(n_vox**3, n_atoms)`` outer product).

    The box is ``n_vox * voxel_size`` Angstrom on a side; with the defaults that is a
    90 Angstrom box at 0.3 Angstrom/voxel holding a 60 Angstrom particle (so the
    nearest-neighbor spacing is ~9.6 voxels, comfortably resolvable in ground truth while
    still the regime where low-rank reconstructions tend to merge columns).
    """
    box_ang = n_vox * voxel_size
    center_ang = np.array([box_ang / 2.0] * 3, dtype=np.float64)
    centers_vox = _fcc_atom_centers(diameter_ang, lattice_a, center_ang) / voxel_size

    volume = np.zeros((n_vox, n_vox, n_vox), dtype=np.float32)
    sigma_vox = atom_sigma_ang / voxel_size
    half = int(np.ceil(3.0 * sigma_vox))
    off = np.arange(-half, half + 1)
    wz, wy, wx = np.meshgrid(off, off, off, indexing="ij")
    inv_two_sigma2 = 1.0 / (2.0 * sigma_vox**2)

    kept: list[tuple[float, float, float]] = []
    for cz, cy, cx in centers_vox:
        iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))
        # Skip atoms whose splat window would fall outside the box (the vacuum pad makes
        # this rare); keeps indexing simple and the catalog consistent with the volume.
        if not (
            half <= iz < n_vox - half and half <= iy < n_vox - half and half <= ix < n_vox - half
        ):
            continue
        dz = (iz + wz) - cz
        dy = (iy + wy) - cy
        dx = (ix + wx) - cx
        gauss = np.exp(-(dz**2 + dy**2 + dx**2) * inv_two_sigma2).astype(np.float32)
        volume[
            iz - half : iz + half + 1,
            iy - half : iy + half + 1,
            ix - half : ix + half + 1,
        ] += peak_intensity * gauss
        kept.append((cz, cy, cx))

    return AuPhantom(
        volume=volume,
        atom_coords=np.asarray(kept, dtype=np.float32),
        voxel_size=voxel_size,
        lattice_a=lattice_a,
        nn_dist=lattice_a / np.sqrt(2.0),
    )


def project_volume(
    volume: np.ndarray | torch.Tensor,
    angles: np.ndarray,
    device: str = "cpu",
    mode: str = "bilinear",
) -> np.ndarray:
    """Tilt-series projection via quantem's ZXZ forward model.

    Rotates the volume about the X axis by each angle (``rot_ZXZ`` with ``z1 = z3 = 0``)
    and sums along the beam (Z) axis -- the same recipe as the tomography test fixtures,
    so synthetic data stays consistent with the reconstruction geometry. Returns a
    ``(n_angles, Y, X)`` float32 array.
    """
    vol = torch.as_tensor(volume, dtype=torch.float32, device=device).unsqueeze(0)
    projections = []
    for angle in np.asarray(angles, dtype=np.float32):
        rotated = rot_ZXZ(vol, 0.0, float(angle), 0.0, device=device, mode=mode)
        projections.append(rotated[0].sum(0))
    return torch.stack(projections).cpu().numpy().astype(np.float32)


def project_au_phantom(
    phantom: AuPhantom, tilt_angles: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Project an :class:`AuPhantom` into a tilt series (see :func:`project_volume`)."""
    return project_volume(phantom.volume, tilt_angles, device=device)


def add_poisson_noise(
    tilt_series: np.ndarray,
    dose: float | None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Add Poisson (shot) noise to a tilt series.

    ``dose`` sets the peak expected electron count: the series is scaled so its maximum
    maps to ``dose`` counts, Poisson-sampled, then rescaled back to the original intensity
    range. Lower ``dose`` is noisier. ``dose=None`` returns the input unchanged (the clean
    gate). Noise is applied to the *projections* -- it lives in the measurements, not the
    volume.
    """
    if dose is None:
        return np.asarray(tilt_series, dtype=np.float32)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    ts = np.asarray(tilt_series, dtype=np.float64)
    vmax = float(ts.max())
    if vmax <= 0.0:
        return ts.astype(np.float32)
    scale = dose / vmax
    noisy = rng.poisson(np.clip(ts * scale, 0.0, None)) / scale
    return noisy.astype(np.float32)
