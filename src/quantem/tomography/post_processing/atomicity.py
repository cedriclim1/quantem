"""Ground-truth-anchored atomicity metrics for reconstructed volumes.

These metrics deliberately reward *atom separation / localization*, not smoothness, so they
will not be fooled by a denoiser that simply blurs adjacent columns together (which PSNR
rewards). They are pure numpy/scipy and depend only on a known ground-truth atom catalog,
not on the ``imaging.Lattice`` fitter.

Metrics
-------
detection_f1
    Atom-detection precision/recall/F1 + localization RMSE. Merging two columns into one
    blob drops recall; spurious noise peaks drop precision.
valley_contrast
    Mean valley-to-peak depth along ground-truth nearest-neighbor bonds. A pure separation
    measure: two merged atoms fill the valley between them and drive this toward zero.
lattice_peak_snr
    Amplitude at the crystal's reciprocal-lattice shells over background, from the 3D FFT.
psnr_mse
    Global fidelity to a reference volume (tracked to expose the denoise-vs-atomicity
    tension -- it can improve while atomicity does not).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.spatial import cKDTree


@dataclass
class DetectionMetrics:
    f1: float
    precision: float
    recall: float
    rmse_ang: float
    n_detected: int
    n_gt: int
    n_matched: int


def _otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold over a 1-D array of positive intensities (numpy-only)."""
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin
    hist, edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w_bg = np.cumsum(hist)
    total = w_bg[-1]
    w_fg = total - w_bg
    cum_mean = np.cumsum(hist * centers)
    valid = (w_bg > 0) & (w_fg > 0)
    mean_bg = np.zeros_like(w_bg)
    mean_fg = np.zeros_like(w_bg)
    mean_bg[valid] = cum_mean[valid] / w_bg[valid]
    mean_fg[valid] = (cum_mean[-1] - cum_mean[valid]) / w_fg[valid]
    between = np.where(valid, w_bg * w_fg * (mean_bg - mean_fg) ** 2, 0.0)
    return float(centers[int(np.argmax(between))])


def _resolve_threshold(volume: np.ndarray, spec: float | str) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    pos = volume[volume > 0]
    if pos.size == 0:
        return 0.0
    if spec == "otsu":
        return _otsu_threshold(pos)
    if spec == "mean":
        return float(pos.mean())
    raise ValueError(f"Unknown intensity threshold spec: {spec!r}")


def _subvoxel_refine(volume: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Parabolic sub-voxel peak refinement along each axis (skips border peaks)."""
    refined = coords.astype(np.float64).copy()
    shape = volume.shape
    for axis in range(3):
        for k, c in enumerate(coords):
            iz, iy, ix = int(c[0]), int(c[1]), int(c[2])
            idx = [iz, iy, ix]
            if idx[axis] <= 0 or idx[axis] >= shape[axis] - 1:
                continue
            lo, hi = idx.copy(), idx.copy()
            lo[axis] -= 1
            hi[axis] += 1
            v0 = volume[iz, iy, ix]
            vm = volume[lo[0], lo[1], lo[2]]
            vp = volume[hi[0], hi[1], hi[2]]
            denom = vm - 2.0 * v0 + vp
            if denom != 0.0:
                offset = 0.5 * (vm - vp) / denom
                refined[k, axis] += float(np.clip(offset, -0.5, 0.5))
    return refined


def detect_atoms(
    volume: np.ndarray,
    nn_dist_vox: float,
    nms_radius_frac: float = 0.6,
    intensity_thresh: float | str = "otsu",
) -> np.ndarray:
    """Detect atom centers as thresholded local maxima with sub-voxel refinement.

    Returns an ``(n, 3)`` float array of (z, y, x) voxel coordinates. The non-max
    suppression window is sized to ``nms_radius_frac * nn_dist_vox`` so two distinct
    columns at the nearest-neighbor spacing are not collapsed into one detection.
    """
    radius = max(1, int(round(nms_radius_frac * nn_dist_vox)))
    footprint = 2 * radius + 1
    local_max = maximum_filter(volume, size=footprint, mode="constant", cval=0.0)
    threshold = _resolve_threshold(volume, intensity_thresh)
    is_peak = (volume == local_max) & (volume > 0) & (volume >= threshold)
    coords = np.argwhere(is_peak)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return _subvoxel_refine(volume, coords)


def match_atoms(
    detected: np.ndarray, gt: np.ndarray, gate_vox: float
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one nearest-neighbor matching within ``gate_vox`` (KDTree).

    Returns ``(matched_detected_idx, matched_gt_idx)``. Greedy nearest-within-gate is
    effectively optimal for well-separated atoms and avoids the O(n^3) cost of a full
    Hungarian assignment over thousands of atoms.
    """
    empty = np.zeros(0, dtype=int)
    if len(detected) == 0 or len(gt) == 0:
        return empty, empty
    tree = cKDTree(detected)
    dist, idx = tree.query(gt, distance_upper_bound=gate_vox, k=1)
    cand = [(dist[i], i, int(idx[i])) for i in range(len(gt)) if np.isfinite(dist[i])]
    cand.sort()
    used_det: set[int] = set()
    used_gt: set[int] = set()
    md: list[int] = []
    mg: list[int] = []
    for _, gi, di in cand:
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi)
        used_det.add(di)
        mg.append(gi)
        md.append(di)
    return np.asarray(md, dtype=int), np.asarray(mg, dtype=int)


def detection_f1(
    volume: np.ndarray,
    gt_coords: np.ndarray,
    *,
    voxel_size: float,
    nn_dist_ang: float,
    nms_radius_frac: float = 0.6,
    intensity_thresh: float | str = "otsu",
    gate_frac: float = 0.5,
) -> DetectionMetrics:
    """Atom-detection precision/recall/F1 and localization RMSE against ``gt_coords``."""
    nn_vox = nn_dist_ang / voxel_size
    detected = detect_atoms(volume, nn_vox, nms_radius_frac, intensity_thresh)
    md, mg = match_atoms(detected, gt_coords, gate_frac * nn_vox)
    n_det, n_gt, n_match = len(detected), len(gt_coords), len(mg)
    precision = n_match / n_det if n_det else 0.0
    recall = n_match / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    if n_match > 0:
        diffs = detected[md] - np.asarray(gt_coords)[mg]
        rmse_ang = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1)))) * voxel_size
    else:
        rmse_ang = float("nan")
    return DetectionMetrics(f1, precision, recall, rmse_ang, n_det, n_gt, n_match)


def _sample(volume: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Trilinear sample ``volume`` at ``(n, 3)`` (z, y, x) fractional coordinates."""
    return map_coordinates(volume, pts.T, order=1, mode="constant", cval=0.0)


def valley_contrast(
    volume: np.ndarray,
    gt_coords: np.ndarray,
    *,
    voxel_size: float,
    nn_dist_ang: float,
    bond_tol_frac: float = 1.15,
    n_bonds_max: int = 2000,
    seed: int = 0,
) -> float:
    """Mean valley-to-peak depth along ground-truth nearest-neighbor bonds.

    For each bonded pair, samples the two atom intensities and the midpoint intensity;
    returns the mean of ``(peak - valley) / peak``. Near 1 when columns are well separated,
    toward 0 when they are merged. NaN if no bonds are found.
    """
    gt = np.asarray(gt_coords, dtype=np.float64)
    if len(gt) < 2:
        return float("nan")
    nn_vox = nn_dist_ang / voxel_size
    tree = cKDTree(gt)
    pairs = np.asarray(list(tree.query_pairs(r=bond_tol_frac * nn_vox)), dtype=int)
    if len(pairs) == 0:
        return float("nan")
    if len(pairs) > n_bonds_max:
        rng = np.random.default_rng(seed)
        pairs = pairs[rng.choice(len(pairs), n_bonds_max, replace=False)]
    p1, p2 = gt[pairs[:, 0]], gt[pairs[:, 1]]
    mid = 0.5 * (p1 + p2)
    peak = 0.5 * (_sample(volume, p1) + _sample(volume, p2))
    valley = _sample(volume, mid)
    good = peak > 1e-8
    if not np.any(good):
        return float("nan")
    return float(np.mean((peak[good] - valley[good]) / peak[good]))


def lattice_peak_snr(
    volume: np.ndarray,
    *,
    voxel_size: float,
    g_vectors_inv_ang: tuple[float, ...] = (0.4246, 0.4904),
    ring_tol_inv_ang: float = 0.03,
) -> dict[str, float]:
    """Reciprocal-lattice peak SNR from the 3D power spectrum.

    For each target ``|g|`` shell, returns ``(max power in shell) / (median background)``.
    A detection-free, global check that the periodic lattice is present and sharp. Assumes
    a cubic volume.
    """
    power = np.abs(np.fft.fftshift(np.fft.fftn(volume))) ** 2
    n = volume.shape[0]
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=voxel_size))
    fz, fy, fx = np.meshgrid(freq, freq, freq, indexing="ij")
    gmag = np.sqrt(fz**2 + fy**2 + fx**2)
    background = float(np.median(power[gmag > 0])) + 1e-12
    out: dict[str, float] = {}
    for g0 in g_vectors_inv_ang:
        shell = np.abs(gmag - g0) <= ring_tol_inv_ang
        peak = float(power[shell].max()) if np.any(shell) else 0.0
        out[f"snr_g{g0:.3f}"] = peak / background
    return out


def psnr_mse(
    volume: np.ndarray, reference: np.ndarray, data_range: float | None = None
) -> tuple[float, float]:
    """Return ``(psnr_db, mse)`` of ``volume`` against ``reference``."""
    vol = np.asarray(volume, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    mse = float(np.mean((vol - ref) ** 2))
    if data_range is None:
        data_range = float(ref.max() - ref.min())
    if mse <= 0.0 or data_range <= 0.0:
        return float("inf"), mse
    return 10.0 * np.log10(data_range**2 / mse), mse


def score_volume(
    volume: np.ndarray,
    gt_coords: np.ndarray,
    reference: np.ndarray,
    *,
    voxel_size: float,
    nn_dist_ang: float,
) -> dict[str, float]:
    """Bundle all metrics into a flat dict for tabulation / metric callbacks."""
    det = detection_f1(volume, gt_coords, voxel_size=voxel_size, nn_dist_ang=nn_dist_ang)
    psnr, mse = psnr_mse(volume, reference)
    out = {
        "f1": det.f1,
        "precision": det.precision,
        "recall": det.recall,
        "rmse_ang": det.rmse_ang,
        "n_detected": float(det.n_detected),
        "n_matched": float(det.n_matched),
        "valley_contrast": valley_contrast(
            volume, gt_coords, voxel_size=voxel_size, nn_dist_ang=nn_dist_ang
        ),
        "psnr": psnr,
        "mse": mse,
    }
    out.update(lattice_peak_snr(volume, voxel_size=voxel_size))
    return out
