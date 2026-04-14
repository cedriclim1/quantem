import itertools
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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
) -> float:
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
    return result  # pyright: ignore[reportReturnType]
 

def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: nn.ModuleList,
) -> torch.Tensor:
    coo_combs = list(itertools.combinations(range(3), 2))  # [(0,1), (0,2), (1,2)]
    multi_scale_interp = []

    for grid in ms_grids:
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            interp_space = interp_space * interp_out_plane
        multi_scale_interp.append(interp_space)

    return torch.cat(multi_scale_interp, dim=-1)