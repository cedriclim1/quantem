"""
Tensor Decomposition Methods for INR-based reconstructions
"""

from typing import Any, Callable, Optional, Sequence

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from torch import nn

from .inr_td.utils import init_planes, interpolate_ms_features, query_planes

# -----------------------------------------------------------------------------
# K-Planes Utilty Functions
# -----------------------------------------------------------------------------


class KPlanes(nn.Module):

    def __init__(
        self,
        # Grid parameters
        grid_dimensions: int = 2,
        input_coords_dims: int = 3,
        M_features: int = 32,
        resolution: Sequence[int] = (200, 200, 200),
        multiscale_res_multipliers: Optional[Sequence[int]] = None,
        concat_features: bool = True,
        density_activation: Callable = lambda x: F.softplus(x - 1),
    ):
        """
        Assume coords are [-1, 1] in each dimension.
        """
        super().__init__()
        
        self.grid_dimensions = grid_dimensions
        self.input_coords_dims = input_coords_dims
        self.M_features = M_features
        self.resolution = resolution
        self.multiscale_res_multipliers = multiscale_res_multipliers or [1]
        self.concat_features = concat_features
        self.density_activation = density_activation

        
        # Initialize planes
        self.grids = nn.ParameterList()
        self.feature_dim = 0

        # Resolution pyramid
        for res_mult in self.multiscale_res_multipliers:
            scaled_res = [r * res_mult for r in self.resolution]
            gp = init_planes(
                in_dim=self.input_coords_dims,
                out_dim=self.M_features,
                resolution=scaled_res,
            )
            
            self.feature_dim += gp[-1].shape[1]
            self.grids.append(gp)


        # Linear net
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "None",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 0,
            },
        )



    def get_densities(self, coords: torch.Tensor):
        """Computes and returns densities"""

        pts = coords.reshape(-1, 3)
        features = interpolate_ms_features(
            pts=pts,
            ms_grids=self.grids,
        )
        density_before_activation = self.sigma_net(features)
        density = self.density_activation(density_before_activation)
        return density

    def forward(
        self,
        pts: torch.Tensor,
    ):
        return self.get_densities(pts)

    def get_params(self) -> dict[str, list[torch.nn.Parameter]]:
        return {
            "grids": [p for grid in self.grids for p in grid],  # flatten ParameterLists
            "sigma_net": list(self.sigma_net.parameters()),
        }

 
    def set_optimizer(self, optimizer_params: dict[str, Any]):
        
        self._grids.set_optimizer(optimizer_params["grids"])
        self._sigmanet.set_optimizer(optimizer_params["sigmanet"])


    
    def get_params(self) -> dict[str, list[torch.nn.Parameter]]:
        return {
            "grids": self._grids.params  # flatten ParameterLists
            "sigma_net": self._sigma_net.params
        }

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
 
# class KPlanesVolume(nn.Module):
#     """K-Planes model for fitting a 3D scalar (grayscale) volume.

#     Uses a multi-scale planar factorization where each scale maintains its own
#     set of k-planes. Features are combined via Hadamard product within each
#     scale, then concatenated across scales before decoding.

#     Args:
#         resolution: Base resolution per axis, e.g. [128, 128, 128].
#         feature_dims: Feature channels per scale. If a single int is given,
#                       the same dim is used for all scales. If a list, must
#                       match the length of scale_resolutions.
#         scale_resolutions: Explicit list of per-axis resolutions for each
#                            scale, e.g. [[64,64,64], [128,128,128], [256,256,256]].
#                            Mutually exclusive with resolution + multiscale.
#         multiscale: List of resolution multipliers applied to `resolution`,
#                     e.g. [1, 2, 4] -> 128, 256, 512. Ignored if
#                     scale_resolutions is provided.
#         use_decoder: If True, use a two-layer MLP decoder.
#                      If False, use a single linear projection.
#         decoder_hidden: Hidden dim for the MLP decoder.
#         plane_lr: Learning rate for plane parameters.
#         decoder_lr: Learning rate for decoder parameters.
#         init_range: Uniform init range for spatial plane values.
#     """

#     def __init__(
#         self,
#         resolution: Sequence[int] = (128, 128, 128),
#         feature_dims: Union[int, List[int]] = 32,
#         scale_resolutions: Optional[List[Sequence[int]]] = None,
#         multiscale: Optional[List[int]] = None,
#         use_decoder: bool = True,
#         decoder_hidden: int = 64,
#         plane_lr: float = 1e-1,
#         decoder_lr: float = 1e-3,
#         init_range: tuple = (0.1, 0.5),
#     ):
#         super().__init__()
#         self.in_dim = len(resolution)

#         # --- Resolve per-scale resolutions ---
#         if scale_resolutions is not None:
#             self.scale_resolutions = scale_resolutions
#         else:
#             multipliers = multiscale or [1]
#             self.scale_resolutions = [
#                 [r * m for r in resolution] for m in multipliers
#             ]
#         num_scales = len(self.scale_resolutions)

#         # --- Resolve per-scale feature dims ---
#         if isinstance(feature_dims, int):
#             self.feature_dims = [feature_dims] * num_scales
#         else:
#             assert len(feature_dims) == num_scales, (
#                 f"feature_dims length {len(feature_dims)} must match "
#                 f"number of scales {num_scales}"
#             )
#             self.feature_dims = feature_dims

#         # --- Build one plane set per scale ---
#         self.plane_sets = nn.ModuleList([
#             init_planes(self.in_dim, fdim, res, init_range=init_range)
#             for fdim, res in zip(self.feature_dims, self.scale_resolutions)
#         ])

#         # --- Decoder ---
#         total_feat = sum(self.feature_dims)
#         if use_decoder:
#             self.decoder = nn.Sequential(
#                 nn.Linear(total_feat, decoder_hidden),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(decoder_hidden, 1),
#             )
#         else:
#             self.decoder = nn.Linear(total_feat, 1)

#         # Store LRs for use in configure_optimizers
#         self.plane_lr = plane_lr
#         self.decoder_lr = decoder_lr

#     def get_optimizer(self, **adam_kwargs) -> torch.optim.Optimizer:
#         """Returns an Adam optimizer with separate LRs for planes and decoder."""
#         return torch.optim.Adam([
#             {"params": self.plane_sets.parameters(), "lr": self.plane_lr},
#             {"params": self.decoder.parameters(),    "lr": self.decoder_lr},
#         ], **adam_kwargs)

#     def forward(self, pts: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pts: (B, 3) points in [-1, 1].

#         Returns:
#             (B,) scalar values (e.g. density or grayscale intensity).
#         """
#         scale_features = [
#             query_planes(pts, plane_set, self.in_dim)
#             for plane_set in self.plane_sets
#         ]                                        # list of (B, fdim_i)
#         features = torch.cat(scale_features, dim=-1)   # (B, sum(feature_dims))
#         return self.decoder(features).squeeze(-1)      # (B,)