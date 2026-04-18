"""
Tensor Decomposition Methods for INR-based reconstructions
"""

import itertools
import math
from typing import Callable, Optional, Sequence

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from torch import nn
from quantem.core.ml.profiling import nvtx_range
from .model_base import PPLR

"""
K-planes utility functions
"""
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
    ms_grids: nn.ParameterList,
) -> torch.Tensor:
    mat_mode = [[0, 1], [0, 2], [1, 2]]
    coord_plane = torch.stack([
        pts[:, mat_mode[0]],
        pts[:, mat_mode[1]],
        pts[:, mat_mode[2]],
    ]).view(3, -1, 1, 2)

    per_scale = []
    for plane_coef in ms_grids:
        C = plane_coef.shape[1]
        feats = F.grid_sample(
            plane_coef, coord_plane, align_corners=True, mode="bilinear", padding_mode="border"
        ).reshape(3, C, -1)
        fused = feats[0] * feats[1] * feats[2]
        per_scale.append(fused.T)

    return torch.cat(per_scale, dim=-1)


class KPlanes(nn.Module, PPLR):

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
        # Hybrid MLP parameters
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
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

        self.grids = nn.ParameterList()
        self.feature_dim = 0
        for res_mult in self.multiscale_res_multipliers:
            scaled_res = [int(r * res_mult) for r in self.resolution]
            plane = nn.Parameter(torch.empty(3, self.M_features, scaled_res[1], scaled_res[0]))
            nn.init.uniform_(plane, 0.1, 0.5)
            self.grids.append(plane)
            self.feature_dim += self.M_features

        # Network head
        if use_hybrid_mlp:
            hybrid_hidden_dim = int(hybrid_hidden_dim)
            hybrid_num_layers = int(hybrid_num_layers)
            if hybrid_hidden_dim <= 0:
                raise ValueError(f"hybrid_hidden_dim must be >= 1, got {hybrid_hidden_dim}")
            if hybrid_num_layers <= 0:
                raise ValueError(f"hybrid_num_layers must be >= 1, got {hybrid_num_layers}")

            factory = {}  # add dtype/device kwargs here if needed
            layers = []
            in_dim = self.feature_dim
            for _ in range(hybrid_num_layers):
                lin = nn.Linear(in_dim, hybrid_hidden_dim, **factory)
                nn.init.kaiming_uniform_(lin.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.append(lin)
                layers.append(nn.ReLU(inplace=True))
                in_dim = hybrid_hidden_dim

            out = nn.Linear(in_dim, 1, bias=True, **factory)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            layers.append(out)
            self.sigma_net = nn.Sequential(*layers)
        else:
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

    def forward(self, pts: torch.Tensor):
        return self.get_densities(pts)

    def get_params(self) -> dict[str, list[torch.nn.Parameter]]:
        return {
            "grids": list(self.grids.parameters()),
            "sigma_net": list(self.sigma_net.parameters()),
        }

    @property
    def param_keys(self) -> list[str]:
        return ["grids", "sigma_net"]


# --- Tilted KPlanes ---

# ---------------------------------------------------------------------------
# SO(3) quaternion parameter module
# ---------------------------------------------------------------------------
 
# class SO3Param(nn.Module):
#     """
#     Stores T unit quaternions as learnable parameters in R^4 and normalises
#     them on every call to `as_matrix()`.
 
#     Quaternion convention: [x, y, z, w]  (scalar-last, same as scipy).
 
#     Initialisation
#     --------------
#     "random"  – uniform sampling over SO(3) via Shoemake's method.
#     "identity" – all rotations start as the identity (good for fine-tuning).
#     """
 
#     def __init__(self, T: int, init: str = "random"):
#         super().__init__()
#         if T < 1:
#             raise ValueError(f"T must be >= 1, got {T}")
#         quats = self._init_quaternions(T, init)   # (T, 4)
#         self.quats = nn.Parameter(quats)
 
#     # ------------------------------------------------------------------
#     # Initialisers
#     # ------------------------------------------------------------------
 
#     @staticmethod
#     def _shoemake_sample(T: int) -> torch.Tensor:
#         """Uniform SO(3) sampling via Shoemake (1992). Returns (T, 4) [x,y,z,w]."""
#         u = torch.rand(T, 3)
#         sqrt1_u0 = torch.sqrt(1.0 - u[:, 0])
#         sqrt_u0  = torch.sqrt(u[:, 0])
#         two_pi   = 2.0 * math.pi
#         x = sqrt1_u0 * torch.sin(two_pi * u[:, 1])
#         y = sqrt1_u0 * torch.cos(two_pi * u[:, 1])
#         z = sqrt_u0  * torch.sin(two_pi * u[:, 2])
#         w = sqrt_u0  * torch.cos(two_pi * u[:, 2])
#         return torch.stack([x, y, z, w], dim=-1)   # (T, 4)
 
#     @staticmethod
#     def _identity(T: int) -> torch.Tensor:
#         """All-identity rotations: [0,0,0,1] * T."""
#         q = torch.zeros(T, 4)
#         q[:, 3] = 1.0
#         return q
 
#     @classmethod
#     def _init_quaternions(cls, T: int, init: str) -> torch.Tensor:
#         if init == "random":
#             return cls._shoemake_sample(T)
#         elif init == "identity":
#             return cls._identity(T)
#         else:
#             raise ValueError(f"Unknown init '{init}'; choose 'random' or 'identity'.")
 
#     # ------------------------------------------------------------------
#     # Forward helpers
#     # ------------------------------------------------------------------
 
#     def normalized(self) -> torch.Tensor:
#         """Returns (T, 4) unit quaternions."""
#         return F.normalize(self.quats, p=2, dim=-1)
 
#     def as_matrix(self) -> torch.Tensor:
#         """
#         Converts the T stored quaternions to (T, 3, 3) rotation matrices.
 
#         Uses the standard formula; no trig, just multiplications.
#         """
#         q = self.normalized()          # (T, 4)  [x, y, z, w]
#         x, y, z, w = q.unbind(dim=-1)  # each (T,)
 
#         # Precompute products
#         xx, yy, zz = x*x, y*y, z*z
#         xy, xz, yz = x*y, x*z, y*z
#         wx, wy, wz = w*x, w*y, w*z
 
#         # Row-major: R[i,j]
#         R = torch.stack([
#             1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy),
#               2*(xy + wz),    1 - 2*(xx + zz),     2*(yz - wx),
#               2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy),
#         ], dim=-1).reshape(-1, 3, 3)   # (T, 3, 3)
 
#         return R
 
#     def extra_repr(self) -> str:
#         return f"T={self.quats.shape[0]}"


class SO3Param(nn.Module):
    """
    SO(3) rotation bank using R9+SVD parameterization.
    Each rotation is stored as an unconstrained 3x3 matrix M,
    projected to SO(3) via SVD+(M) = U diag(1,1,det(UVt)) Vt.
    """

    def __init__(self, T: int, init: str = "random"):
        super().__init__()
        if init == "random":
            # Initialize near identity with small noise
            M = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
            M = M + 0.1 * torch.randn(T, 3, 3)
        elif init == "identity":
            M = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        else:
            raise ValueError(f"Unknown init '{init}'")
        self.M = nn.Parameter(M)   # (T, 3, 3)

    def as_matrix(self) -> torch.Tensor:
        """Projects each M to SO(3) via SVD. Returns (T, 3, 3)."""
        U, _, Vh = torch.linalg.svd(self.M)   # U: (T,3,3), Vh: (T,3,3)
        # Fix reflections: det(U Vh) must be +1
        d = torch.det(U @ Vh)                  # (T,)
        diag = torch.ones(self.M.shape[0], 3, device=self.M.device)
        diag[:, 2] = d                          # multiply last singular vector by sign
        return U @ (diag.unsqueeze(-1) * Vh)   # (T, 3, 3)

@torch.compile(mode="reduce-overhead")
def interpolate_ms_features_tilted(pts, ms_grids, rotation_matrices):

    with nvtx_range(enabled=True, name="setup"):
        T = rotation_matrices.shape[0]
        B = pts.shape[0]

    with nvtx_range(enabled=True, name="rotate"):
        rotated = torch.einsum("tij,bj->tbi", rotation_matrices, pts)

    with nvtx_range(enabled=True, name="build_coords"):
        idx = torch.tensor([[0, 1], [2, 0], [1, 2]], device=pts.device)
        coords = rotated.unsqueeze(1).expand(T, 3, B, 3).gather(
            -1, idx.view(1, 3, 1, 2).expand(T, 3, B, 2)
        )
        coord_tensor = coords.reshape(3 * T, B, 1, 2).contiguous()

    with nvtx_range(enabled=True, name="per_scale_features"):
        per_scale_features = []
        for i, plane_coef in enumerate(ms_grids):
            C = plane_coef.shape[1]
            with nvtx_range(enabled=True, name=f"grid_sample_s{i}"):
                sampled = F.grid_sample(
                    plane_coef, coord_tensor,
                    align_corners=True, mode="bilinear", padding_mode="border",
                )
            with nvtx_range(enabled=True, name=f"hadamard_s{i}"):
                sampled = sampled.squeeze(-1).view(T, 3, C, B).prod(dim=1)
                per_scale_features.append(sampled.permute(2, 0, 1).reshape(B, T * C))

    with nvtx_range(enabled=True, name="cat_scales"):
        return torch.cat(per_scale_features, dim=-1)

# ---------------------------------------------------------------------------
# KPlanesTILTED
# ---------------------------------------------------------------------------
 
class KPlanesTILTED(KPlanes):
    """
    K-Planes with T learned SO(3) rotations (TILTED).
 
    Inherits KPlanes for the sigma_net, density_activation, and get_params
    interface.  Overrides:
      * __init__  – replaces the axis-aligned grids with (3*T)-plane grids
                    and adds SO3Param.
      * get_densities  – calls the TILTED interpolation instead.
      * get_params     – adds "so3" key so callers can set a separate lr.
      * param_keys     – updated list.
 
    Parameters
    ----------
    M_features : int
        Feature channels *per transform per scale*.  Total feature_dim will
        be M_features * T * len(multiscale_res_multipliers).
    T : int
        Number of learned rotations (TILTED-T in the paper; 4 or 8 recommended).
    tau_init : str
        "random" (paper default) or "identity".
    tau_warmup_steps : int
        If > 0, grids and sigma_net are frozen for this many steps so the
        rotations can find good basins first (two-phase warm-up).
        Call model.training_step() once per optimiser step.
    All other args are forwarded to KPlanes.
    """
 
    def __init__(
        self,
        # Grid parameters
        input_coords_dims: int = 3,
        M_features: int = 32,
        resolution: Sequence[int] = (200, 200, 200),
        multiscale_res_multipliers: Optional[Sequence[int]] = None,
        density_activation: Callable = lambda x: F.softplus(x - 1),
        # TILTED parameters
        T: int = 4,
        tau_init: str = "random",
        tau_warmup_steps: int = 0,
        # Hybrid MLP parameters
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
    ):
        if input_coords_dims != 3:
            raise NotImplementedError("KPlanesTILTED is implemented for 3D only.")
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
 
        multiscale_res_multipliers = list(multiscale_res_multipliers or [1])
        num_scales = len(multiscale_res_multipliers)
 
        # Total feature dim seen by the MLP head.
        # Each scale contributes M_features * T channels.
        feature_dim = M_features * T * num_scales
 
        # Call KPlanes.__init__ with grid_dimensions=2 so it builds sigma_net
        # correctly; we immediately replace self.grids below.
        super().__init__(
            grid_dimensions=2,
            input_coords_dims=3,
            M_features=M_features,         # base class stores this
            resolution=resolution,
            multiscale_res_multipliers=multiscale_res_multipliers,
            concat_features=True,
            density_activation=density_activation,
            use_hybrid_mlp=use_hybrid_mlp,
            hybrid_hidden_dim=hybrid_hidden_dim,
            hybrid_num_layers=hybrid_num_layers,
        )
        # KPlanes.__init__ built grids with shape (3, M, H, W) and feature_dim
        # = M * num_scales.  We rebuild them for the TILTED shape.
 
        self.T = T
        self.tau_warmup_steps = tau_warmup_steps
        self._global_step: int = 0
 
        # ---- Rebuild grids: (3*T, M_features, H, W) per scale ----
        self.grids = nn.ParameterList()
        for res_mult in multiscale_res_multipliers:
            scaled_res = [int(r * res_mult) for r in resolution]
            plane = nn.Parameter(
                torch.empty(3 * T, M_features, scaled_res[1], scaled_res[0])
            )
            nn.init.uniform_(plane, 0.1, 0.5)
            self.grids.append(plane)
 
        # ---- Rebuild sigma_net with the correct feature_dim ----
        # KPlanes built sigma_net with self.feature_dim (= M * num_scales),
        # which is wrong for T > 1.  Rebuild here.
        self.feature_dim = feature_dim
        self._build_sigma_net(use_hybrid_mlp, hybrid_hidden_dim, hybrid_num_layers)
 
        # ---- Learnable rotations ----
        self.so3 = SO3Param(T, init=tau_init)
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _build_sigma_net(
        self,
        use_hybrid_mlp: bool,
        hybrid_hidden_dim: int,
        hybrid_num_layers: int,
    ) -> None:
        """Rebuild sigma_net for self.feature_dim (called after grids are set)."""
        if use_hybrid_mlp:
            layers = []
            in_dim = self.feature_dim
            for _ in range(hybrid_num_layers):
                lin = nn.Linear(in_dim, hybrid_hidden_dim)
                nn.init.kaiming_uniform_(lin.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.append(lin)
                layers.append(nn.ReLU(inplace=True))
                in_dim = hybrid_hidden_dim
            out = nn.Linear(in_dim, 1, bias=True)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            layers.append(out)
            self.sigma_net = nn.Sequential(*layers)
        else:
            # Match mentor's "explicit" decoder: a single linear layer.
            # Small init so density stays near 0 initially.
            self.sigma_net = nn.Linear(self.feature_dim, 1, bias=True)
            nn.init.normal_(self.sigma_net.weight, std=0.01)
            nn.init.zeros_(self.sigma_net.bias)
 
    # ------------------------------------------------------------------
    # Warm-up bookkeeping
    # ------------------------------------------------------------------
 
    def training_step(self) -> None:
        """
        Call once per optimiser step to advance the internal counter.
 
        During the first `tau_warmup_steps` iterations, grids and sigma_net
        have their gradients zeroed after the backward pass so only the SO(3)
        parameters update.  This is the lightweight version of two-phase
        optimisation from the paper.
        """
        print("Global Stepped")
        self._global_step += 1
 
    def _in_warmup(self) -> bool:
        return self.tau_warmup_steps > 0 and self._global_step < self.tau_warmup_steps
 
    def zero_non_tau_grads(self) -> None:
        """
        Call after loss.backward() and before optimizer.step() when you want
        to implement the rotation warm-up manually.  Alternatively just check
        model.in_warmup and configure your optimizer accordingly.
        """
        if self._in_warmup():
            for p in self.grids.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            for p in self.sigma_net.parameters():
                if p.grad is not None:
                    p.grad.zero_()
 
    @property
    def in_warmup(self) -> bool:
        return self._in_warmup()
 
    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
 
    def get_densities(self, coords):
        pts = coords.reshape(-1, 3)
        with nvtx_range(enabled=True, name="tilted_interp"):
            with nvtx_range(enabled=True, name="so3_as_matrix"):
                R = self.so3.as_matrix()
            with nvtx_range(enabled=True, name="interpolate_ms_features_tilted"):
                features = interpolate_ms_features_tilted(pts, self.grids, R)
        with nvtx_range(enabled=True, name="sigma_net"):
            density_pre = self.sigma_net(features)
        with nvtx_range(enabled=True, name="density_act"):
            return self.density_activation(density_pre)
 
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return self.get_densities(pts)
 
    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------
 
    def get_params(self) -> dict[str, list[nn.Parameter]]:
        return {
            "grids":     list(self.grids.parameters()),
            "sigma_net": list(self.sigma_net.parameters()),
            "so3":       list(self.so3.parameters()),
        }
 
    @property
    def param_keys(self) -> list[str]:
        return ["grids", "sigma_net", "so3"]
 

    # ------------------------------------------------------------------
    # Two-phase helper: extract tau for phase-2 initialisation
    # ------------------------------------------------------------------
 
    def extract_tau_state(self) -> torch.Tensor:
        """
        Returns the current quaternion tensor (detached copy) so it can be
        used to initialise a larger phase-2 model via `load_tau_state`.
        """
        return self.so3.quats.detach().clone()
 
    def load_tau_state(self, quats: torch.Tensor) -> None:
        """
        Load pre-trained quaternions (e.g. from a bottleneck phase-1 model).
 
        quats : (T, 4) tensor, will be normalised internally.
        """
        if quats.shape != self.so3.quats.shape:
            raise ValueError(
                f"Shape mismatch: got {quats.shape}, "
                f"expected {self.so3.quats.shape}"
            )
        with torch.no_grad():
            self.so3.quats.copy_(F.normalize(quats, p=2, dim=-1))
 
    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
 
    def extra_repr(self) -> str:
        return (
            f"T={self.T}, "
            f"M_features={self.M_features}, "
            f"feature_dim={self.feature_dim}, "
            f"num_scales={len(self.multiscale_res_multipliers)}, "
            f"tau_warmup_steps={self.tau_warmup_steps}"
        )