"""
Tensor Decomposition Methods for INR-based reconstructions
"""

import itertools
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# K-Planes Utilty Functions
# -----------------------------------------------------------------------------

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """
    Performs bilinear interpolation on a grid at given coordinates.
    
    Args:
        grid: Grid tensor of shape [B, C, H, W] or [C, H, W]
        coords: Coordinate tensor of shape [B, N, 2] or [N, 2]
        align_corners: Whether to align corners
        
    Returns:
        Interpolated values of shape [B, N, C] or [N, C]
    """
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp



def init_planes(
    in_dim: int,
    out_dim: int,
    resolution: Sequence[int],
    init_range: tuple = (0.1, 0.5),
) -> nn.ParameterList:
    """Create the set of 2D planes for a k-plane decomposition.
 
    For in_dim=3 (spatial), this creates 3 planes: XY, XZ, YZ.
    For in_dim=4 (spatial + time), this creates 6 planes: XY, XZ, XT, YZ, YT, ZT.
    Time planes (those involving axis 3) are initialized to 1 so they start
    as identity multipliers.
 
    Args:
        in_dim: Dimensionality of the input coordinates (3 or 4).
        out_dim: Number of feature channels per plane.
        resolution: Resolution along each axis, e.g. [128, 128, 128].
        init_range: (a, b) for uniform initialization of spatial planes.
 
    Returns:
        nn.ParameterList of plane parameters, each of shape [1, out_dim, res_j, res_i].
    """
    assert len(resolution) == in_dim
    # All pairs of axes
    axis_pairs = list(itertools.combinations(range(in_dim), 2))
    planes = nn.ParameterList()
    a, b = init_range
    for pair in axis_pairs:
        # grid_sample expects (N, C, H, W) — so resolution is reversed
        shape = [1, out_dim] + [resolution[ax] for ax in reversed(pair)]
        param = nn.Parameter(torch.empty(*shape))
        # Time planes init to 1; spatial planes init uniform
        if in_dim == 4 and 3 in pair:
            nn.init.ones_(param)
        else:
            nn.init.uniform_(param, a=a, b=b)
        planes.append(param)
    return planes
 
 
def query_planes(
    pts: torch.Tensor,
    planes: nn.ParameterList,
    in_dim: int,
) -> torch.Tensor:
    """Query the k-plane representation at a batch of points.
 
    Projects each point onto every axis-pair plane, bilinearly interpolates,
    and returns the element-wise product across all planes.
 
    Args:
        pts: (B, in_dim) coordinates in [-1, 1].
        planes: The ParameterList from init_planes.
        in_dim: 3 or 4.
 
    Returns:
        (B, out_dim) features.
    """
    axis_pairs = list(itertools.combinations(range(in_dim), 2))
    result = 1.0
    for plane_param, pair in zip(planes, axis_pairs):
        # Extract the 2D coords for this plane
        coords_2d = pts[..., list(pair)]                  # (B, 2)
        coords_2d = coords_2d.view(1, -1, 1, 2)          # (1, B, 1, 2) for grid_sample
        # grid_sample: input (N,C,H,W), grid (N, H_out, W_out, 2)
        sampled = F.grid_sample(
            plane_param,          # (1, C, H, W)
            coords_2d,            # (1, B, 1, 2)
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )  # -> (1, C, B, 1)
        sampled = sampled.squeeze(0).squeeze(-1).T        # (B, C)
        result = result * sampled
    return result
 
 
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
 
class KPlanesVolume(nn.Module):
    """K-Planes model for fitting a 3D scalar (grayscale) volume.

    Uses a multi-scale planar factorization where each scale maintains its own
    set of k-planes. Features are combined via Hadamard product within each
    scale, then concatenated across scales before decoding.

    Args:
        resolution: Base resolution per axis, e.g. [128, 128, 128].
        feature_dims: Feature channels per scale. If a single int is given,
                      the same dim is used for all scales. If a list, must
                      match the length of scale_resolutions.
        scale_resolutions: Explicit list of per-axis resolutions for each
                           scale, e.g. [[64,64,64], [128,128,128], [256,256,256]].
                           Mutually exclusive with resolution + multiscale.
        multiscale: List of resolution multipliers applied to `resolution`,
                    e.g. [1, 2, 4] -> 128, 256, 512. Ignored if
                    scale_resolutions is provided.
        use_decoder: If True, use a two-layer MLP decoder.
                     If False, use a single linear projection.
        decoder_hidden: Hidden dim for the MLP decoder.
        plane_lr: Learning rate for plane parameters.
        decoder_lr: Learning rate for decoder parameters.
        init_range: Uniform init range for spatial plane values.
    """

    def __init__(
        self,
        resolution: Sequence[int] = (128, 128, 128),
        feature_dims: Union[int, List[int]] = 32,
        scale_resolutions: Optional[List[Sequence[int]]] = None,
        multiscale: Optional[List[int]] = None,
        use_decoder: bool = True,
        decoder_hidden: int = 64,
        plane_lr: float = 1e-1,
        decoder_lr: float = 1e-3,
        init_range: tuple = (0.1, 0.5),
    ):
        super().__init__()
        self.in_dim = len(resolution)

        # --- Resolve per-scale resolutions ---
        if scale_resolutions is not None:
            self.scale_resolutions = scale_resolutions
        else:
            multipliers = multiscale or [1]
            self.scale_resolutions = [
                [r * m for r in resolution] for m in multipliers
            ]
        num_scales = len(self.scale_resolutions)

        # --- Resolve per-scale feature dims ---
        if isinstance(feature_dims, int):
            self.feature_dims = [feature_dims] * num_scales
        else:
            assert len(feature_dims) == num_scales, (
                f"feature_dims length {len(feature_dims)} must match "
                f"number of scales {num_scales}"
            )
            self.feature_dims = feature_dims

        # --- Build one plane set per scale ---
        self.plane_sets = nn.ModuleList([
            init_planes(self.in_dim, fdim, res, init_range=init_range)
            for fdim, res in zip(self.feature_dims, self.scale_resolutions)
        ])

        # --- Decoder ---
        total_feat = sum(self.feature_dims)
        if use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(total_feat, decoder_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(decoder_hidden, 1),
            )
        else:
            self.decoder = nn.Linear(total_feat, 1)

        # Store LRs for use in configure_optimizers
        self.plane_lr = plane_lr
        self.decoder_lr = decoder_lr

    def get_optimizer(self, **adam_kwargs) -> torch.optim.Optimizer:
        """Returns an Adam optimizer with separate LRs for planes and decoder."""
        return torch.optim.Adam([
            {"params": self.plane_sets.parameters(), "lr": self.plane_lr},
            {"params": self.decoder.parameters(),    "lr": self.decoder_lr},
        ], **adam_kwargs)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: (B, 3) points in [-1, 1].

        Returns:
            (B,) scalar values (e.g. density or grayscale intensity).
        """
        scale_features = [
            query_planes(pts, plane_set, self.in_dim)
            for plane_set in self.plane_sets
        ]                                        # list of (B, fdim_i)
        features = torch.cat(scale_features, dim=-1)   # (B, sum(feature_dims))
        return self.decoder(features).squeeze(-1)      # (B,)