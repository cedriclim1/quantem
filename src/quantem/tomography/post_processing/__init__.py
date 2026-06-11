"""Post-reconstruction refinement and evaluation for tomography volumes.

- :mod:`dip_refiner` -- a 3D-U-Net deep-image-prior that refines a reconstructed volume.
- :mod:`atomicity` -- ground-truth-anchored metrics that reward atom separation.
- :mod:`baselines` -- cheap denoising baselines (gaussian / median) for comparison.
"""

from quantem.tomography.post_processing.atomicity import (
    DetectionMetrics,
    detection_f1,
    lattice_peak_snr,
    psnr_mse,
    score_volume,
    valley_contrast,
)
from quantem.tomography.post_processing.baselines import (
    gaussian_baseline,
    identity_baseline,
    median_baseline,
)
from quantem.tomography.post_processing.dip_refiner import ObjectDIPRefiner, RefineResult
from quantem.tomography.post_processing.joint_refiner import (
    JointRefiner,
    reproject_volume,
    split_tilt_indices,
)

__all__ = [
    "ObjectDIPRefiner",
    "RefineResult",
    "JointRefiner",
    "reproject_volume",
    "split_tilt_indices",
    "DetectionMetrics",
    "detection_f1",
    "valley_contrast",
    "lattice_peak_snr",
    "psnr_mse",
    "score_volume",
    "gaussian_baseline",
    "median_baseline",
    "identity_baseline",
]
