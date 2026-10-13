from dataclasses import dataclass
from typing import Optional

import torch

from quantem.core.ml.constraints import BaseContext


@dataclass
class ReconstructionContext(BaseContext):
    """
    Handles all reconstruction parameters to be passed into object models.

    Subclasses will pick whatever parameter they need
        - Pixelated reads ".volume"
        - INR reads ".coords" and recomputes via the model.
        - TensorDecomp reads ".coords" and ".pred" (and ".all densities")

    Variable descriptions:
    - volume: Reconstructed object (volume).
    - coords: Used for INR reconstructions to provide the coordinates to the model.
    - pred: Predicted values per coordinate position from the model.
    - all_densities: Per-sample densities from the model forward (box mask and hard
      constraints applied), flattened over every sample of every ray in the batch, BEFORE
      ray integration. ``pred`` is their integral per ray.
    - obj: Object model (INR, TensorDecomp, etc.).
    - target: Measured projection pixels for the batch, for data-coupled constraints (e.g. S3IM).
    - tv_tap_densities: Raw model outputs at the volume-TV finite-difference taps, when the
      training loop merged them into the main forward pass.
    - distortion: Unweighted batch-mean ray distortion loss computed by the dataset before
      ray integration (``TomographyINRDataset.ray_distortion``); ``None`` when the
      distortion weight is zero. Object models multiply it by ``constraints.distortion``.
    - soft_terms: Optional dict the caller passes to receive each weighted soft-constraint
      term (detached) keyed by name, e.g. ``tv_plane``, ``tv_vol``, ``sparsity``,
      ``l21_plane``, ``distortion``, ``s3im``. Object models only write into it when it is
      not ``None``; the returned total is unaffected. The training loop adds the dataset
      pose TV (``tv_zs`` + ``tv_shifts``) as ``pose_tv`` when active, so the epoch
      ``loss/soft`` equals the sum of the ``loss/soft/<name>`` terms.
    """

    volume: Optional[torch.Tensor] = None
    coords: Optional[torch.Tensor] = None
    pred: Optional[torch.Tensor] = None
    all_densities: Optional[torch.Tensor] = None
    obj: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    tv_tap_densities: Optional[torch.Tensor] = None
    distortion: Optional[torch.Tensor] = None
    soft_terms: Optional[dict[str, torch.Tensor]] = None


def record_soft_term(
    ctx: ReconstructionContext, name: str, value: torch.Tensor
) -> None:
    """Store a detached copy of a weighted soft-constraint term in ``ctx.soft_terms``.

    No-op when the caller did not request per-term values (``ctx.soft_terms is None``).
    """
    if ctx.soft_terms is not None:
        ctx.soft_terms[name] = value.detach()
