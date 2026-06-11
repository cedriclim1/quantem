"""Cheap denoising baselines for the post-reconstruction comparison.

A DIP win is only meaningful if it beats simple smoothing, so these provide the reference
points (raw / gaussian / median) the experiment scores alongside the U-Net.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def identity_baseline(volume: np.ndarray) -> np.ndarray:
    """Return the volume unchanged (the raw-reconstruction reference)."""
    return np.asarray(volume, dtype=np.float32)


def gaussian_baseline(volume: np.ndarray, sigma_vox: float = 1.0) -> np.ndarray:
    """Isotropic 3D Gaussian smoothing."""
    return gaussian_filter(np.asarray(volume, dtype=np.float32), sigma=sigma_vox)


def median_baseline(volume: np.ndarray, size: int = 3) -> np.ndarray:
    """3D median filter (edge-preserving impulse-noise baseline)."""
    return median_filter(np.asarray(volume, dtype=np.float32), size=size)
