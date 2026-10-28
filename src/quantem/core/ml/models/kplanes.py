"""
Tensor Decomposition Methods for INR-based reconstructions
"""

import copy
import itertools
import math
from typing import Callable, Literal, Sequence

# import tinycudann as tcnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from quantem.core import config
from quantem.core.ml.dist_utils import sum_across_ranks, sum_grad_across_ranks

from .model_base import PPLR, TensorDecompositionModel
from .so3params import SO3ParamQuat, SO3ParamR9SVD

"""
K-planes utility functions
"""


def grid_sample_wrapper(
    grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True
) -> torch.Tensor:
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
        raise NotImplementedError(
            f"Grid-sample was called with {grid_dim}D data but is only "
            f"implemented for 2 and 3D data."
        )

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode="bilinear",
        padding_mode="border",
    )
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
        coords_2d = pts[..., list(pair)]  # (B, 2)
        coords_2d = coords_2d.view(1, -1, 1, 2)  # (1, B, 1, 2) for grid_sample
        # grid_sample: input (N,C,H,W), grid (N, H_out, W_out, 2)
        sampled = F.grid_sample(
            plane_param,  # (1, C, H, W)
            coords_2d,  # (1, B, 1, 2)
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )  # -> (1, C, B, 1)
        sampled = sampled.squeeze(0).squeeze(-1).T  # (B, C)
        result = result * sampled
    return result  # pyright: ignore[reportReturnType]


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: nn.ParameterList,
) -> torch.Tensor:
    # Plane axis layout: XY=(0,1), XZ=(0,2), YZ=(1,2). Stacking views avoids the
    # per-call index-tensor allocation (a host-to-device copy) that list-based
    # advanced indexing does three times per forward.
    x, y, z = pts.unbind(-1)
    coord_plane = torch.stack(
        [
            torch.stack((x, y), dim=-1),
            torch.stack((x, z), dim=-1),
            torch.stack((y, z), dim=-1),
        ]
    ).view(3, -1, 1, 2)

    per_scale = []
    for plane_coef in ms_grids:
        C = plane_coef.shape[1]
        feats = F.grid_sample(
            plane_coef, coord_plane, align_corners=True, mode="bilinear", padding_mode="border"
        ).reshape(3, C, -1)
        fused = feats[0] * feats[1] * feats[2]
        per_scale.append(fused.T)

    return torch.cat(per_scale, dim=-1)


class KPlanes(PPLR, TensorDecompositionModel):
    """
    K-Planes model adapted from Fridovich-Keil et al., https://arxiv.org/abs/2301.10241
    """

    def __init__(
        self,
        # Grid parameters
        grid_dimensions: int = 2,
        input_coords_dims: int = 3,
        M_features: int = 32,
        resolution: Sequence[int] = (200, 200, 200),
        multiscale_res_multipliers: Sequence[float] | None = None,
        concat_features: bool = True,
        density_activation: Callable = lambda x: F.softplus(
            x - 1
        ),  # Keep playing around with this and trunc_exp
        # Hybrid MLP parameters
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
    ):
        """
        Assume coords are [-1, 1] in each dimension.
        """
        super().__init__()
        self.td_type = "kplanes"
        self.grid_dimensions = grid_dimensions
        self.input_coords_dims = input_coords_dims
        self.M_features = M_features
        self.resolution = resolution
        self.multiscale_res_multipliers = multiscale_res_multipliers or [1]
        self.concat_features = concat_features
        self.density_activation = density_activation

        # All three planes share one (3, C, res[1], res[0]) tensor, which ignores
        # res[2]: an anisotropic resolution would silently give the XZ/YZ planes the
        # wrong grid along z. Refuse it rather than misallocate.
        if len(set(self.resolution)) != 1:
            raise ValueError(
                f"KPlanes currently requires an isotropic resolution, got {self.resolution}; "
                "the plane grids are allocated as (res[1], res[0]) for all three axis pairs."
            )

        self.grids = nn.ParameterList()
        self.feature_dim = 0
        for res_mult in self.multiscale_res_multipliers:
            scaled_res = [int(r * res_mult) for r in self.resolution]
            plane = nn.Parameter(torch.empty(3, self.M_features, scaled_res[1], scaled_res[0]))
            nn.init.uniform_(plane, 0.1, 0.5)
            self.grids.append(plane)
            self.feature_dim += self.M_features

        # Network head (single linear when not hybrid; small ReLU MLP when hybrid)
        self._build_sigma_net(use_hybrid_mlp, hybrid_hidden_dim, hybrid_num_layers)

    def _build_sigma_net(
        self,
        use_hybrid_mlp: bool,
        hybrid_hidden_dim: int,
        hybrid_num_layers: int,
    ) -> None:
        """Build the decoder head mapping concatenated grid features -> density.

        ``use_hybrid_mlp=True`` builds a small ReLU MLP; otherwise a single linear
        "explicit" decoder. Both init the final layer small (``weight ~ N(0, 0.01**2)``,
        ``bias=0``) so the density starts near zero. Called after ``self.feature_dim``
        is finalized.
        """
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
            # Single-linear "explicit" decoder. Small init -> density ~ 0 initially.
            self.sigma_net = nn.Linear(self.feature_dim, 1, bias=True)
            nn.init.normal_(self.sigma_net.weight, std=0.01)
            nn.init.zeros_(self.sigma_net.bias)

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

    @property
    def tilted(self) -> bool:
        return False

    @property
    def grids(self) -> torch.nn.ParameterList:
        return self._grids

    @grids.setter
    def grids(self, grids: torch.nn.ParameterList):
        if not isinstance(grids, torch.nn.ParameterList):
            raise TypeError("Grids must be a ParameterList")
        self._grids = grids

    @property
    def resolution(self) -> list[int]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: Sequence[int]):
        if not isinstance(resolution, Sequence):
            raise TypeError("Resolution must be a sequence")
        self._resolution = list(resolution)


# ---------------------------------------------------------------------------
# KPlanesTILTED
# ---------------------------------------------------------------------------


_KERNEL_CASTABLE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _rotate_tilted_points_fp32(
    pts: torch.Tensor, rotation_matrices: torch.Tensor
) -> torch.Tensor:
    """Rotate points by every TILTED rotation with autocast off, in at least FP32."""
    dtype = torch.promote_types(
        torch.promote_types(pts.dtype, rotation_matrices.dtype), torch.float32
    )
    with torch.autocast(device_type=pts.device.type, enabled=False):
        return torch.einsum("tij,bj->tbi", rotation_matrices.to(dtype), pts.to(dtype))


def interpolate_ms_features_tilted(
    pts: torch.Tensor,  # (B, 3)
    ms_grids: nn.ParameterList,  # each grid: (3*T, C, H, W)
    rotation_matrices: torch.Tensor,  # (T, 3, 3)
    scale_gates: Sequence[float] | None = None,  # per-scale multiplier (coarse-to-fine)
) -> torch.Tensor:
    """
    Fully-vectorized multi-scale, multi-rotation K-Planes feature interpolation.
    Returns features of shape (B, C * T * num_scales).

    scale_gates (optional): one scalar per scale in [0,1] applied to that scale's feature
    block before concatenation. Used for coarse-to-fine band-limiting: ramp the finer
    scales on over training so the model commits to view-consistent low-frequency structure
    first (a classical limited-angle prior). None => all scales fully on (default behaviour).

    When the optional ``quantem-cuda`` package is installed, the tensors live
    on a CUDA device, and the ``use_cuda_kernels`` config option is true
    (default), each scale dispatches to the fused CUDA kernel (rotate +
    grid_sample + Hadamard product in one launch, analytic backward) —
    identical semantics to the torch path below, including scale_gates. The
    kernel is FP32-only: bfloat16/float16 inputs are cast to FP32 (autocast
    disabled, differentiably) and the features come back in FP32; float64
    inputs take the torch path so their precision is kept.
    """
    T = rotation_matrices.shape[0]
    B = pts.shape[0]

    if (
        pts.is_cuda
        and all(
            t.dtype in _KERNEL_CASTABLE_DTYPES for t in (pts, rotation_matrices, *ms_grids)
        )
        and config.get("has_quantem_cuda")
        and config.get("use_cuda_kernels", default=True)
    ):
        from quantem.cuda.core.ml import kplanes_tilted_fuse

        with torch.autocast(device_type=pts.device.type, enabled=False):
            pts32 = pts.float()
            rot32 = rotation_matrices.float()
            feats = [kplanes_tilted_fuse(pts32, rot32, g.float()) for g in ms_grids]
            if scale_gates is not None:
                feats = [f * scale_gates[si] for si, f in enumerate(feats)]
            return torch.cat(feats, dim=-1)

    # (T, B, 3)  — rotate all points by all rotations at once
    rotated = _rotate_tilted_points_fp32(pts, rotation_matrices)

    # Build (T, 3, B, 2) coords for planes XY, ZX, YZ in one shot.
    # Plane axis layout: XY=(0,1), ZX=(2,0), YZ=(1,2). Stacking views avoids the
    # per-call index-tensor allocation (a host-to-device copy) and the
    # (T, 3, B, 3) expand+gather this used to do.
    x, y, z = rotated.unbind(-1)  # each (T, B)
    coords = torch.stack(
        (
            torch.stack((x, y), dim=-1),
            torch.stack((z, x), dim=-1),
            torch.stack((y, z), dim=-1),
        ),
        dim=1,
    )  # (T, 3, B, 2)

    # Flatten (T, 3) -> 3*T so it matches grid's first dim, and add the H_out=1 axis
    coord_tensor = coords.reshape(3 * T, B, 1, 2)  # (3T, B, 1, 2)

    per_scale_features = []
    for si, plane_coef in enumerate(ms_grids):
        # plane_coef: (3T, C, H, W)
        C = plane_coef.shape[1]

        sampled = F.grid_sample(
            plane_coef,
            coord_tensor,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )  # (3T, C, B, 1)

        # (3T, C, B) -> (T, 3, C, B) -> Hadamard across the "3" dim -> (T, C, B)
        sampled = sampled.squeeze(-1).view(T, 3, C, B).prod(dim=1)

        # (T, C, B) -> (B, T, C) -> (B, T*C) to concatenate rotations along feature dim
        feat = sampled.permute(2, 0, 1).reshape(B, T * C)
        if scale_gates is not None:
            feat = feat * scale_gates[si]
        per_scale_features.append(feat)

    # Concatenate across scales -> (B, T * C * num_scales)
    return torch.cat(per_scale_features, dim=-1)


def c2f_scale_gates(progress: float, warmup_frac: float, num_scales: int) -> list[float]:
    """Cosine (BARF/Nerfies) per-scale gates; level 0 always on, all on at warmup end."""
    if warmup_frac <= 0 or num_scales <= 1:
        return [1.0] * num_scales
    eta = (num_scales - 1) * min(1.0, max(0.0, progress / warmup_frac))
    gates = []
    for i in range(num_scales):
        alpha = min(1.0, max(0.0, eta - i + 1))
        gates.append(0.5 * (1.0 - math.cos(math.pi * alpha)))
    return gates


def c2f_blur_sigma(progress: float, sigma0: float, blur_frac: float) -> float:
    """Plane blur sigma, cosine-annealed from sigma0 to 0 over the first blur_frac of training."""
    if sigma0 <= 0 or blur_frac <= 0:
        return 0.0
    t = min(1.0, max(0.0, progress / blur_frac))
    return 0.0 if t >= 1.0 else sigma0 * 0.5 * (1.0 + math.cos(math.pi * t))


def gaussian_blur_planes(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable, differentiable Gaussian low-pass over the last two axes (replicate padding).

    Runs with autocast disabled, so the blurred planes keep the grid's dtype (FP32)
    under a bf16 autocast forward.
    """
    if sigma <= 0:
        return grid
    radius = max(1, math.ceil(3.0 * sigma))
    with torch.autocast(device_type=grid.device.type, enabled=False):
        support = torch.arange(-radius, radius + 1, device=grid.device, dtype=grid.dtype)
        kernel = torch.exp(-0.5 * (support / sigma) ** 2)
        kernel = kernel / kernel.sum()
        K = kernel.numel()
        shape = grid.shape
        x = grid.reshape(-1, 1, shape[-2], shape[-1])
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, K))
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, K, 1))
        return x.reshape(shape)


def extract_tau_state(so3: SO3ParamQuat | SO3ParamR9SVD) -> torch.Tensor:
    """Detached CPU float32 (T, 3, 3) rotations of an SO(3) bank, computed in float64."""
    with torch.no_grad():
        return copy.deepcopy(so3).to("cpu", torch.float64).as_matrix().float()


class KPlanesTILTED(KPlanes):
    """
    K-Planes with T learned SO(3) rotations (TILTED). Adapted from Yi et al., https://arxiv.org/abs/2308.15461

    Inherits KPlanes for the sigma_net, density_activation, and get_params
    interface; grids are (3*T, M_features, H, W) per scale and "so3" is an extra
    parameter group.

    Two-phase optimisation: a CPTilted bottleneck is trained first and its
    rotations are transferred with ``extract_tau_state()`` / ``load_tau_state()``
    (proper (T, 3, 3) rotation matrices, so the quat and r9svd parameterisations
    can be mixed); see ``Tomography.reconstruct_two_phase``.

    Coarse-to-fine schedules, both driven by ``set_progress(frac)`` (the training
    loop calls it every batch with completed / planned gradient steps):

    * ``c2f_warmup_frac``: multiscale level gating. Level 0 (coarsest) is always on;
      finer levels fade in with a cosine ramp and all are on at ``frac = c2f_warmup_frac``.
      No-op for a single level.
    * ``c2f_blur_sigma0`` / ``c2f_blur_frac``: every plane is Gaussian low-passed
      before interpolation, sigma cosine-annealed from ``c2f_blur_sigma0`` (pixels of
      the finest level; level l uses ``sigma * mult_l / max(mult)``, the same physical
      width) to 0 at ``frac = c2f_blur_frac``. At sigma 0 no extra ops run.

    Parameters
    ----------
    M_features : int
        Feature channels *per transform per scale*.  Total feature_dim will
        be M_features * T * len(multiscale_res_multipliers).
    T : int
        Number of learned rotations (TILTED-T in the paper; 4 or 8 recommended).
        Must match between phase 1 and phase 2 when doing two-phase transfer.
    tau_init : str
        "random" (paper default) or "identity".
        Irrelevant if you're calling load_tau_state() right after __init__.
    All other args are forwarded to KPlanes.
    """

    def __init__(
        self,
        # Grid parameters
        input_coords_dims: int = 3,
        M_features: int = 32,
        resolution: Sequence[int] = (200, 200, 200),
        multiscale_res_multipliers: Sequence[float] | None = None,
        density_activation: Callable = lambda x: F.softplus(x - 1),
        # TILTED parameters
        T: int = 4,
        tau_init: Literal["random", "identity"] = "random",
        # Hybrid MLP parameters
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
        so3_param_type: str = "r9svd",
        c2f_warmup_frac: float = 0.0,
        c2f_blur_sigma0: float = 0.0,
        c2f_blur_frac: float = 0.0,
    ):
        self.td_type = "tilted"
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
            M_features=M_features,
            resolution=resolution,
            multiscale_res_multipliers=multiscale_res_multipliers,
            concat_features=True,
            density_activation=density_activation,
            use_hybrid_mlp=use_hybrid_mlp,
            hybrid_hidden_dim=hybrid_hidden_dim,
            hybrid_num_layers=hybrid_num_layers,
        )

        self.T = T

        # ---- Rebuild grids: (3*T, M_features, H, W) per scale ----
        self.grids = nn.ParameterList()
        for res_mult in multiscale_res_multipliers:
            scaled_res = [int(r * res_mult) for r in resolution]
            plane = nn.Parameter(torch.empty(3 * T, M_features, scaled_res[1], scaled_res[0]))
            nn.init.uniform_(plane, 0.1, 0.5)
            self.grids.append(plane)

        # ---- Rebuild sigma_net with the correct feature_dim ----
        # KPlanes built sigma_net with self.feature_dim (= M * num_scales),
        # which is wrong for T > 1.  Rebuild here.
        self.feature_dim = feature_dim
        self._build_sigma_net(use_hybrid_mlp, hybrid_hidden_dim, hybrid_num_layers)

        # ---- Learnable rotations ----
        self.set_so3_param_type(so3_param_type, init=tau_init)

        self.num_scales = num_scales
        self.c2f_warmup_frac = float(c2f_warmup_frac)
        self.c2f_blur_sigma0 = float(c2f_blur_sigma0)
        self.c2f_blur_frac = float(c2f_blur_frac)
        if self.c2f_warmup_frac < 0 or self.c2f_blur_sigma0 < 0 or self.c2f_blur_frac < 0:
            raise ValueError("c2f_warmup_frac, c2f_blur_sigma0 and c2f_blur_frac must be >= 0.")
        if (self.c2f_blur_sigma0 > 0) != (self.c2f_blur_frac > 0):
            raise ValueError(
                "c2f_blur_sigma0 and c2f_blur_frac must both be > 0 to enable plane blur, "
                f"got sigma0={self.c2f_blur_sigma0}, frac={self.c2f_blur_frac}."
            )
        self._scale_gates = [1.0] * num_scales
        self._blur_sigma = 0.0
        self._progress = 0.0
        self.set_progress(0.0)

    def set_progress(self, frac: float) -> None:
        """Set training progress in [0, 1] and update the coarse-to-fine schedules."""
        self._progress = min(1.0, max(0.0, float(frac)))
        self._scale_gates = c2f_scale_gates(self._progress, self.c2f_warmup_frac, self.num_scales)
        self._blur_sigma = c2f_blur_sigma(self._progress, self.c2f_blur_sigma0, self.c2f_blur_frac)

    @property
    def scale_gates(self) -> list[float]:
        return list(self._scale_gates)

    @property
    def blur_sigma(self) -> float:
        return self._blur_sigma

    def _active_scale_gates(self) -> list[float] | None:
        """Per-scale gates, or None when every gate is fully open."""
        if all(g == 1.0 for g in self._scale_gates):
            return None
        return self._scale_gates

    def _active_grids(self) -> Sequence[torch.Tensor]:
        """Plane grids, low-passed when the c2f blur sigma is positive."""
        sigma = self._blur_sigma
        if sigma <= 0:
            return self.grids
        finest = max(self.multiscale_res_multipliers)
        return [
            gaussian_blur_planes(g, sigma * m / finest)
            for g, m in zip(self.grids, self.multiscale_res_multipliers)
        ]

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
            # Single-linear "explicit" decoder. Small init -> density ~ 0 initially.
            self.sigma_net = nn.Linear(self.feature_dim, 1, bias=True)
            nn.init.normal_(self.sigma_net.weight, std=0.01)
            nn.init.zeros_(self.sigma_net.bias)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def get_densities(self, coords: torch.Tensor) -> torch.Tensor:
        pts = coords.reshape(-1, 3)
        R = self.so3.as_matrix()  # (T, 3, 3)
        if hasattr(self, "_blur_sigma"):
            grids, gates = self._active_grids(), self._active_scale_gates()
        else:
            grids, gates = self.grids, None
        features = interpolate_ms_features_tilted(
            pts=pts,
            ms_grids=grids,
            rotation_matrices=R,
            scale_gates=gates,
        )
        density_before_activation = self.sigma_net(features)
        return self.density_activation(density_before_activation)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return self.get_densities(pts)

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, list[nn.Parameter]]:
        return {
            "grids": list(self.grids.parameters()),
            "sigma_net": list(self.sigma_net.parameters()),
            "so3": list(self.so3.parameters()),
        }

    @property
    def param_keys(self) -> list[str]:
        return ["grids", "sigma_net", "so3"]

    # ------------------------------------------------------------------
    # Two-phase transfer: extract / load learned rotations
    # ------------------------------------------------------------------

    def extract_tau_state(self) -> torch.Tensor:
        """Detached CPU copy of the current rotations as (T, 3, 3) SO(3) matrices."""
        return extract_tau_state(self.so3)

    def load_tau_state(self, R: torch.Tensor) -> None:
        """Seed the rotations from (T, 3, 3) matrices, for either SO(3) parameterisation."""
        self.so3.load_matrix(R)

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, "
            f"M_features={self.M_features}, "
            f"feature_dim={self.feature_dim}, "
            f"num_scales={len(self.multiscale_res_multipliers)}"
        )

    def set_so3_param_type(
        self, so3_param_type: str, init: Literal["random", "identity"] = "random"
    ) -> None:
        """
        Set the SO3 parameterization type.

        Parameters
        ----------
        so3_param_type : str
            SO3 parameterization type ("quat" or "r9svd").
        """
        if so3_param_type == "r9svd":
            self.so3 = SO3ParamR9SVD(self.T, init=init)
        elif so3_param_type == "quat":
            self.so3 = SO3ParamQuat(self.T, init=init)
        else:
            raise ValueError(f"Invalid SO3 parameterization type: {so3_param_type}")

    @property
    def tilted(self) -> bool:
        return True


def plane_split_layout(world_size: int, T: int, M: int, t_shards: int | None = None) -> tuple[int, int]:
    """(rotation shards, channel shards) for ``world_size`` ranks; rotations are split first by default."""
    t = math.gcd(world_size, T) if t_shards is None else int(t_shards)
    if t < 1 or world_size % t or T % t:
        raise ValueError(f"t_shards={t} must divide both world_size={world_size} and T={T}")
    c = world_size // t
    if M % c:
        raise ValueError(
            f"{world_size} ranks with {t} rotation shards need {c} channel shards, "
            f"which must divide M_features={M}"
        )
    return t, c


class KPlanesTILTEDShard(nn.Module):
    """One rank's slice of a ``KPlanesTILTED`` for plane-split (model-parallel) training.

    The ``T`` rotations and ``M`` channels of every scale are dealt out in blocks: this rank
    keeps rotations ``t0 : t0 + T_local`` and channels ``c0 : c0 + M_local`` of each plane
    level plus the matching decoder columns. The decoder must be the single ``nn.Linear``,
    whose output is a sum over features, so each rank computes a partial pre-activation for
    every point and one ``sum_across_ranks`` gives the full model's value. Every rank must
    see the same points. The bias and the rotations are replicated; their gradients are
    summed by ``sync_replicated_grads``. When the coordinates carry a gradient (pose
    learning) it is summed over ranks at the model input, so pose gradients come out
    complete and identical on every rank. Coarse-to-fine gates and plane blur act per
    channel and are applied to the local slice.
    """

    model_parallel = True

    def __init__(self, model: KPlanesTILTED, rank: int, world_size: int, t_shards: int | None = None):
        super().__init__()
        if not isinstance(model.sigma_net, nn.Linear):
            raise ValueError("plane split needs the single-Linear decoder (use_hybrid_mlp=False)")
        T, M = model.T, model.M_features
        t, c = plane_split_layout(world_size, T, M, t_shards)
        self.T, self.M_features, self.t_shards, self.c_shards = T, M, t, c
        self.T_local, self.M_local = T // t, M // c
        self.t0 = (rank // c) * self.T_local
        self.c0 = (rank % c) * self.M_local
        self.rank, self.world_size = rank, world_size
        t_sl = slice(3 * self.t0, 3 * (self.t0 + self.T_local))
        c_sl = slice(self.c0, self.c0 + self.M_local)
        self.grids = nn.ParameterList(
            [nn.Parameter(g.detach()[t_sl, c_sl].clone().contiguous()) for g in model.grids]
        )
        num_scales = len(model.grids)
        cols = [
            si * T * M + ti * M + ci
            for si in range(num_scales)
            for ti in range(self.t0, self.t0 + self.T_local)
            for ci in range(self.c0, self.c0 + self.M_local)
        ]
        self.weight = nn.Parameter(model.sigma_net.weight.detach()[:, cols].clone())
        self.bias = nn.Parameter(model.sigma_net.bias.detach().clone())
        self.so3 = model.so3
        self.density_activation = model.density_activation
        self.resolution = list(model.resolution)
        self.multiscale_res_multipliers = list(model.multiscale_res_multipliers)
        self.num_scales = num_scales
        self.td_type = model.td_type
        self.c2f_warmup_frac = model.c2f_warmup_frac
        self.c2f_blur_sigma0 = model.c2f_blur_sigma0
        self.c2f_blur_frac = model.c2f_blur_frac
        self.set_progress(0.0)

    @property
    def tilted(self) -> bool:
        return True

    @property
    def layout(self) -> dict[str, int]:
        return {
            "world_size": self.world_size,
            "t_shards": self.t_shards,
            "c_shards": self.c_shards,
            "rotations_per_rank": self.T_local,
            "channels_per_rank": self.M_local,
        }

    def set_progress(self, frac: float) -> None:
        """Set training progress in [0, 1] and update the coarse-to-fine schedules."""
        self._progress = min(1.0, max(0.0, float(frac)))
        self._scale_gates = c2f_scale_gates(self._progress, self.c2f_warmup_frac, self.num_scales)
        self._blur_sigma = c2f_blur_sigma(self._progress, self.c2f_blur_sigma0, self.c2f_blur_frac)

    @property
    def scale_gates(self) -> list[float]:
        return list(self._scale_gates)

    @property
    def blur_sigma(self) -> float:
        return self._blur_sigma

    def _active_grids(self) -> Sequence[torch.Tensor]:
        if self._blur_sigma <= 0:
            return self.grids
        finest = max(self.multiscale_res_multipliers)
        return [
            gaussian_blur_planes(g, self._blur_sigma * m / finest)
            for g, m in zip(self.grids, self.multiscale_res_multipliers)
        ]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.requires_grad:
            coords = sum_grad_across_ranks(coords)
        pts = coords.reshape(-1, 3)
        R = self.so3.as_matrix()[self.t0 : self.t0 + self.T_local]
        gates = None if all(g == 1.0 for g in self._scale_gates) else self._scale_gates
        features = interpolate_ms_features_tilted(pts, self._active_grids(), R, scale_gates=gates)
        pre = features @ self.weight.t()
        if self.rank == 0:
            pre = pre + self.bias
        return self.density_activation(sum_across_ranks(pre))

    def get_params(self) -> dict[str, list[nn.Parameter]]:
        return {
            "grids": list(self.grids),
            "sigma_net": [self.weight, self.bias],
            "so3": list(self.so3.parameters()),
        }

    @property
    def param_keys(self) -> list[str]:
        return ["grids", "sigma_net", "so3"]

    def owned_parameters(self) -> list[nn.Parameter]:
        return [*self.grids, self.weight]

    def replicated_parameters(self) -> list[nn.Parameter]:
        return [self.bias, *self.so3.parameters()]

    def sync_replicated_grads(self) -> None:
        """Sum the partial gradients of the replicated bias and rotations over ranks."""
        for p in self.replicated_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(p.grad)

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """``torch.nn.utils.clip_grad_norm_`` on the full model's gradient; replicated params count once."""
        owned = [p.grad for p in self.owned_parameters() if p.grad is not None]
        replicated = [p.grad for p in self.replicated_parameters() if p.grad is not None]
        sq = torch.zeros((), device=self.weight.device)
        if owned:
            sq = sq + torch.stack([g.pow(2).sum() for g in owned]).sum()
        if self.rank == 0 and replicated:
            sq = sq + torch.stack([g.pow(2).sum() for g in replicated]).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sq)
        total = sq.sqrt()
        coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(owned + replicated, coef)
        return total

    def _full_groups(self, local: torch.Tensor) -> torch.Tensor:
        """Place this rank's per-group values (3 * T_local,) into all 3 * T groups, summed over ranks."""
        full = local.new_zeros(3 * self.T)
        full = full.index_add(0, torch.arange(3 * self.t0, 3 * (self.t0 + self.T_local), device=local.device), local)
        return sum_across_ranks(full)

    def plane_tv(self) -> torch.Tensor:
        """The unsliced model's plane TV (``ObjectTensorDecomp._get_plane_tv_loss`` without the weight)."""
        per_level = []
        for p in self.grids:
            H, W = p.shape[-2:]
            dh = self._full_groups((p[:, :, 1:, :] - p[:, :, :-1, :]).pow(2).sum(dim=(1, 2, 3)))
            dw = self._full_groups((p[:, :, :, 1:] - p[:, :, :, :-1]).pow(2).sum(dim=(1, 2, 3)))
            per_group = dh / (self.M_features * (H - 1) * W) + dw / (self.M_features * H * (W - 1))
            per_level.append(per_group.view(self.T, 3).sum(dim=1).mean())
        return torch.stack(per_level).sum()

    def plane_l21(self) -> torch.Tensor:
        """The unsliced model's group l2,1 norm (``ObjectTensorDecomp.get_plane_l21_loss`` without the weight)."""
        norms = []
        for p in self.grids:
            H, W = p.shape[-2:]
            sq = self._full_groups(p.reshape(p.shape[0], -1).pow(2).sum(dim=1))
            norms.append(torch.sqrt(sq / (self.M_features * H * W) + 1e-12))
        return torch.cat(norms).mean()

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}/{self.world_size}, rotations={self.t0}:{self.t0 + self.T_local} of {self.T}, "
            f"channels={self.c0}:{self.c0 + self.M_local} of {self.M_features}"
        )


# CP Decomp for Warmup SO3 rotations


def interpolate_ms_features_cp_tilted(
    pts: torch.Tensor,  # (B, 3)
    ms_grids: nn.ParameterList,  # each grid: (3*T, C, L) — 1D lines
    rotation_matrices: torch.Tensor,  # (T, 3, 3)
) -> torch.Tensor:
    """
    CP (vector outer product) version of TILTED interpolation.
    Returns features of shape (B, C * T * num_scales).
    """
    T = rotation_matrices.shape[0]
    B = pts.shape[0]

    # Rotate all points by all rotations: (T, B, 3)
    rotated = _rotate_tilted_points_fp32(pts, rotation_matrices)

    # For each transform t, we need three 1D samples: at x_t, y_t, z_t.
    # Lay them out as (3T, B) coords, matching each line grid's first dim.
    # Axis order per transform: x, y, z.
    coords_1d = rotated.reshape(T, B, 3).permute(0, 2, 1).reshape(3 * T, B)

    # grid_sample wants 4D input for 2D sampling: sample the (3T, C, 1, L) lines
    # with 2D coords whose y is fixed at 0. The grid only depends on the points,
    # so it is built once here rather than per scale inside the loop.
    grid = torch.stack(
        [
            coords_1d,  # x
            torch.zeros_like(coords_1d),  # y
        ],
        dim=-1,
    ).unsqueeze(1)  # (3T, 1, B, 2)

    per_scale_features = []
    for line_coef in ms_grids:
        # line_coef: (3T, C, L)  — three 1D feature lines per transform (x, y, z)
        C = line_coef.shape[1]

        sampled = F.grid_sample(
            line_coef.unsqueeze(2),  # (3T, C, 1, L)
            grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        ).squeeze(2)  # (3T, C, B)

        # Hadamard across the 3 axes per transform: (T, 3, C, B) -> (T, C, B)
        sampled = sampled.view(T, 3, C, B).prod(dim=1)

        # (T, C, B) -> (B, T*C)
        per_scale_features.append(sampled.permute(2, 0, 1).reshape(B, T * C))

    return torch.cat(per_scale_features, dim=-1)


class CPTilted(PPLR, TensorDecompositionModel):
    """
    CP decomposition with TILTED rotations — the true bottleneck model for
    phase 1. Rank-1-per-channel feature representation. Adapted from Yi et al., https://arxiv.org/abs/2308.15461

    Shares the SO3Param and sigma_net design with KPlanesTILTED so you can
    lift τ directly across: cp_model.extract_tau_state() ->
    kplanes_model.load_tau_state().
    """

    def __init__(
        self,
        C: int = 4,  # channels per transform per scale
        resolution: Sequence[int] = (128, 128, 128),
        multiscale_res_multipliers: Sequence[float] | None = None,
        T: int = 4,
        tau_init: Literal["random", "identity"] = "random",
        density_activation: Callable = lambda x: F.softplus(x - 1),
        so3_param_type: str = "r9svd",
    ):
        super().__init__()
        self.td_type = "cp_tilted"
        self.T = T
        self.C = C
        self.resolution = list(resolution)
        self.multiscale_res_multipliers = list(multiscale_res_multipliers or [1])
        self.density_activation = density_activation

        # 1D feature lines, one per axis per transform per scale.
        # Shape per scale: (3*T, C, L).  We use max(resolution) for L; if your
        # scene is strongly anisotropic use 3 separate lines per axis.
        self.grids = nn.ParameterList()
        for mult in self.multiscale_res_multipliers:
            L = int(max(resolution) * mult)
            line = nn.Parameter(torch.empty(3 * T, C, L))
            nn.init.uniform_(line, 0.1, 0.5)
            self.grids.append(line)

        self.feature_dim = C * T * len(self.multiscale_res_multipliers)

        # Same minimal single-linear decoder as your KPlanesTILTED default.
        self.sigma_net = nn.Linear(self.feature_dim, 1, bias=True)
        nn.init.normal_(self.sigma_net.weight, std=0.01)
        nn.init.zeros_(self.sigma_net.bias)

        if so3_param_type == "r9svd":
            self.so3 = SO3ParamR9SVD(T, init=tau_init)
        elif so3_param_type == "quat":
            self.so3 = SO3ParamQuat(T, init=tau_init)
        else:
            raise ValueError(f"Unknown SO3 param type: {so3_param_type}")

    def get_densities(self, coords: torch.Tensor) -> torch.Tensor:
        pts = coords.reshape(-1, 3)
        R = self.so3.as_matrix()
        features = interpolate_ms_features_cp_tilted(pts, self.grids, R)
        return self.density_activation(self.sigma_net(features))

    def forward(self, pts):
        return self.get_densities(pts)

    def get_params(self):
        return {
            "grids": list(self.grids.parameters()),
            "sigma_net": list(self.sigma_net.parameters()),
            "so3": list(self.so3.parameters()),
        }

    @property
    def param_keys(self):
        return ["grids", "sigma_net", "so3"]

    def extract_tau_state(self) -> torch.Tensor:
        """Detached CPU copy of the current rotations as (T, 3, 3) SO(3) matrices."""
        return extract_tau_state(self.so3)

    def load_tau_state(self, R: torch.Tensor) -> None:
        """Seed the rotations from (T, 3, 3) matrices, for either SO(3) parameterisation."""
        self.so3.load_matrix(R)

    @property
    def tilted(self) -> bool:
        return True


KPlanesType = KPlanes | KPlanesTILTED | CPTilted
