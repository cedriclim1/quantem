from typing import Callable

import numpy as np
import torch
from torch import nn

from .activation_functions import get_activation_function
from .blocks import SineLayer


class Siren(nn.Module):
    """Original SIREN implementation."""

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        alpha: float = 1.0,
        hsiren: bool = False,
        dtype: torch.dtype = torch.float32,
        final_activation: str | Callable = "identity",
        winner_initialization: bool | int = False,
    ) -> None:
        """Initialize Siren.

        Parameters
        ----------
        in_features : int, optional
            Dimensionality of input coordinates (3 for 3D: z, y, x), by default 3
        out_features : int, optional
            Dimensionality of output (1 for scalar field), by default 1
        hidden_layers : int, optional
            Number of hidden layers, by default 3
        hidden_features : int, optional
            Number of features in each hidden layer, by default 256
        first_omega_0 : float, optional
            Activation function scaling factor for the first layer, by default 30.0
        hidden_omega_0 : float, optional
            Activation function scaling factor for the hidden layers, by default 30.0
        alpha : float, optional
            Weight initialization scaling factor, by default 1.0
        hsiren : bool, optional
            Whether to use the H-Siren activation function, by default False
        dtype : torch.dtype, optional
            Data type for the network, by default torch.float32
        final_activation : str or Callable, optional
            Final activation function, by default "identity"
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.alpha = alpha
        self.hsiren = hsiren
        self.dtype = dtype
        self.winner_initialization = winner_initialization
        self.final_activation = final_activation

        self._build()

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @final_activation.setter
    def final_activation(self, act: str | Callable):
        self._final_activation = get_activation_function(act, dtype=self.dtype)

    def _build(self) -> None:
        net_list = []
        net_list.append(
            SineLayer(
                self.in_features,
                self.hidden_features,
                is_first=True,
                omega_0=self.first_omega_0,
                hsiren=self.hsiren,
                alpha=self.alpha,
                dtype=self.dtype,
            )
        )

        for i in range(self.hidden_layers):
            net_list.append(
                SineLayer(
                    self.hidden_features,
                    self.hidden_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                    alpha=self.alpha,
                    dtype=self.dtype,
                )
            )

        final_linear = nn.Linear(self.hidden_features, self.out_features, dtype=self.dtype)
        with torch.no_grad():
            # Final layer keeps original initialization (no alpha scaling)
            final_linear.weight.uniform_(
                -np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
            )
        net_list.append(final_linear)
        net_list.append(self._final_activation)
        self.net = nn.Sequential(*net_list)

        if self.winner_initialization:
            if type(self.winner_initialization) is int:
                rng = torch.Generator()
                rng.manual_seed(self.winner_initialization)
            else:
                rng = torch.Generator()
                rng.manual_seed(42)
            with torch.no_grad():
                self.net[0].linear.weight += (  # type: ignore[reportAttributeAccessIssue]
                    torch.randn_like(self.net[0].linear.weight) * 5 / self.first_omega_0  # type:ignore
                )
                self.net[1].linear.weight += (  # type: ignore[reportAttributeAccessIssue]
                    torch.randn_like(self.net[1].linear.weight) * 0.1 / self.hidden_omega_0  # type:ignore
                )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)
        return output

    def reset_weights(self) -> None:
        """Reset all weights in the network."""
        self._build()

    def make_equispaced_grid(
        self,
        bounds: tuple[tuple[float, float], ...],
        sampling: tuple[float, ...] | None = None,
        num_points: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Create an equispaced coordinate grid for the implicit neural representation.

        Parameters
        ----------
        bounds : tuple of tuples
            Bounds for each dimension as ((min_0, max_0), (min_1, max_1), ...).
            Length must match in_features.
        sampling : tuple of float
            Sampling interval for each dimension (spacing_0, spacing_1, ...).
            Length must match in_features.
        num_points : tuple of int, optional
            Number of points to sample in each dimension. If None, the number of points
            is calculated from the sampling interval. If both sampling and num_points are provided,
            num_points takes precedence.
        Returns
        -------
        torch.Tensor
            Flattened coordinate grid of shape (N, in_features), where N is the
            total number of grid points.

        Raises
        ------
        ValueError
            If bounds or sampling length does not match in_features.

        Examples
        --------
        For a model with in_features=2:
        >>> bounds = ((0, 1), (0, 1))
        >>> sampling = (0.1, 0.1)
        >>> coords = siren.make_equispaced_grid(bounds, sampling)
        """
        if len(bounds) != self.in_features:
            raise ValueError(
                f"Bounds length ({len(bounds)}) must match in_features ({self.in_features})"
            )
        if sampling is not None and num_points is not None:
            raise ValueError("Only one of sampling or num_points can be provided")
        if sampling is not None:
            if len(sampling) != self.in_features:
                raise ValueError(
                    f"Sampling length ({len(sampling)}) must match in_features ({self.in_features})"
                )
            num_points = tuple(
                int((bound_max - bound_min) / sample) + 1
                for (bound_min, bound_max), sample in zip(bounds, sampling)
            )
        elif num_points is not None:
            if len(num_points) != self.in_features:
                raise ValueError(
                    f"Num points length ({len(num_points)}) must match in_features ({self.in_features})"
                )
        else:
            raise ValueError("Either sampling or num_points must be provided")
        grids = []
        for i, (bound_min, bound_max) in enumerate(bounds):
            n = num_points[i]
            grids.append(torch.linspace(bound_min, bound_max, n))

        coords = torch.meshgrid(*grids, indexing="ij")
        coords = torch.stack(coords, dim=-1).to(self.dtype)
        return coords.reshape(-1, self.in_features)


class HSiren(Siren):
    """H-Siren implementation, the first layer uses sinh instead of sine activation function."""

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
        final_activation: str | Callable = "identity",
        winner_initialization: bool | int = False,
    ) -> None:
        """Initialize HSiren.

        Parameters
        ----------
        in_features : int, optional
            Dimensionality of input coordinates (3 for 3D: z, y, x), by default 3
        out_features : int, optional
            Dimensionality of output (1 for scalar field), by default 1
        hidden_layers : int, optional
            Number of hidden layers, by default 3
        hidden_features : int, optional
            Number of features in each hidden layer, by default 256
        first_omega_0 : float, optional
            Activation function scaling factor for the first layer, by default 30
        hidden_omega_0 : float, optional
            Activation function scaling factor for the hidden layers, by default 30
        alpha : float, optional
            Weight initialization scaling factor, by default 1.0
        dtype : torch.dtype, optional
            Data type for the network, by default torch.float32
        final_activation : str or Callable, optional
            Final activation function, by default "identity"
        """
        super().__init__(
            in_features,
            out_features,
            hidden_layers,
            hidden_features,
            first_omega_0,
            hidden_omega_0,
            alpha,
            hsiren=True,
            dtype=dtype,
            final_activation=final_activation,
            winner_initialization=winner_initialization,
        )


# Tensor Decomposition Methods

import itertools
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Core K-Planes building blocks
# ---------------------------------------------------------------------------
 
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
 
    Supports multi-scale planes and an optional tiny MLP decoder.
 
    Args:
        resolution: Base resolution per axis, e.g. [128, 128, 128].
        feature_dim: Feature channels per plane.
        multiscale: List of resolution multipliers, e.g. [1, 2, 4].
        use_decoder: If True, pass features through a small MLP.
                     If False, directly sum features to a scalar (simpler).
        decoder_hidden: Hidden dim for the decoder MLP.
    """
 
    def __init__(
        self,
        resolution: Sequence[int] = (128, 128, 128),
        feature_dim: int = 32,
        multiscale: Optional[List[int]] = None,
        use_decoder: bool = True,
        decoder_hidden: int = 64,
    ):
        super().__init__()
        self.in_dim = len(resolution)
        self.feature_dim = feature_dim
        self.multiscale = multiscale or [1]
        self.use_decoder = use_decoder
 
        # Create multi-scale planes
        self.plane_sets = nn.ModuleList()
        for mult in self.multiscale:
            res = [r * mult for r in resolution]
            self.plane_sets.append(init_planes(self.in_dim, feature_dim, res))
 
        total_feat = feature_dim * len(self.multiscale)  # concat across scales
 
        # Decoder: features -> scalar density/intensity
        if use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(total_feat, decoder_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(decoder_hidden, 1),
            )
        else:
            # Simple linear projection
            self.decoder = nn.Linear(total_feat, 1)
 
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: (B, 3) points in [-1, 1].
 
        Returns:
            (B,) scalar values (e.g., density or grayscale intensity).
        """
        multi_scale_features = []
        for plane_set in self.plane_sets:
            feat = query_planes(pts, plane_set, self.in_dim)
            multi_scale_features.append(feat)
        features = torch.cat(multi_scale_features, dim=-1)
        out = self.decoder(features).squeeze(-1)
        return out