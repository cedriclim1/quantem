import math
from typing import Callable, Literal, Optional, Tuple

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
                self.net[0].linear.weight += (  # pyright: ignore[reportAttributeAccessIssue]
                    torch.randn_like(self.net[0].linear.weight) * 5 / self.first_omega_0  # type:ignore
                )
                self.net[1].linear.weight += (  # pyright: ignore[reportAttributeAccessIssue]
                    torch.randn_like(self.net[1].linear.weight) * 0.1 / self.hidden_omega_0  # type:ignore
                )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)
        return output

    def reset_weights(self) -> None:
        """Reset all weights in the network."""
        self._build()

    def make_equispaced_grid(
        self, bounds: tuple[tuple[float, float], ...], sampling: tuple[float, ...]
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
        if len(sampling) != self.in_features:
            raise ValueError(
                f"Sampling length ({len(sampling)}) must match in_features ({self.in_features})"
            )

        grids = []
        for (bound_min, bound_max), sample in zip(bounds, sampling):
            num_points = int((bound_max - bound_min) / sample) + 1
            grids.append(torch.linspace(bound_min, bound_max, num_points))

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


class HSiren_TT(nn.Module):
    def __init__(
        self,
        r: int = 128,
        depth: int = 3,
        hidden_features: int = 128,
        out_features: int = 1,
        combine_bias: bool = True,
        use_combine: bool = True,  # toggles CP-like Linear head vs pure scalar
    ):
        super().__init__()
        self.r = int(r)
        self.out_features = int(out_features)
        self.use_combine = bool(use_combine)

        self.fx = HSiren(
            in_features=1,
            out_features=self.r,
            hidden_layers=depth,
            hidden_features=hidden_features,
            final_activation="softplus",
        )
        self.fy = HSiren(
            in_features=1,
            out_features=self.r * self.r,
            hidden_layers=depth,
            hidden_features=hidden_features,
            final_activation="softplus",
        )
        self.fz = HSiren(
            in_features=1,
            out_features=self.r,
            hidden_layers=depth,
            hidden_features=hidden_features,
            final_activation="softplus",
        )

        # CP-style head: maps an r-dim feature vector -> out_features
        self.combine = nn.Linear(self.r, self.out_features, bias=combine_bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [..., 3]
        returns: [..., out_features]
        """
        if coords.shape[-1] != 3:
            raise ValueError(f"Expected coords[..., 3], got {coords.shape}")

        orig_shape = coords.shape[:-1]
        c = coords.reshape(-1, 3)  # [N, 3]

        xs, ys, zs = c[:, 0:1], c[:, 1:2], c[:, 2:3]

        fx = self.fx(xs)  # [N, r]
        fy = self.fy(ys).view(-1, self.r, self.r)  # [N, r, r]
        fz = self.fz(zs)  # [N, r]

        # v = fx^T fy  -> [N, r]
        v = torch.einsum("nr,nrs->ns", fx, fy)  # [N, r]

        # CP-like rank features that use all 3 coords:
        h = v * fz  # [N, r]

        if self.use_combine:
            # Closest to CPFactorizedHSiren:
            out = self.combine(h)  # [N, out_features]
        else:
            # Pure TT scalar intensity:
            scalar = h.sum(dim=-1, keepdim=True)  # [N, 1]
            out = scalar.expand(-1, self.out_features)  # [N, out_features]

        return out.view(*orig_shape, self.out_features)

    def forward_xyz(self, xs: torch.Tensor, ys: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        """
        Optional convenience: xs,ys,zs each shaped [..., 1] or [...].
        Returns [..., out_features].
        """
        xs = xs.unsqueeze(-1) if xs.ndim == 0 or xs.shape[-1] != 1 else xs
        ys = ys.unsqueeze(-1) if ys.ndim == 0 or ys.shape[-1] != 1 else ys
        zs = zs.unsqueeze(-1) if zs.ndim == 0 or zs.shape[-1] != 1 else zs
        coords = torch.cat([xs, ys, zs], dim=-1)
        return self.forward(coords)

    def forward_volume(self, xs, ys, zs):
        fx = self.fx(xs)
        fy = self.fy(ys).view(-1, self.r, self.r)
        fz = self.fz(zs)

        full_tensor = torch.einsum("ia,jab,kb->ijk", fx, fy, fz)

        return full_tensor


class CPFactorizedHSiren(nn.Module):
    def __init__(
        self,
        in_dims: int = 3,
        rank: int = 32,
        out_features: int = 1,
        normalize_rank: bool = True,
        hidden_layers: int = 3,
        hidden_features: int = 128,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        alpha: float = 1.0,
        combine_bias: bool = True,
    ):
        super().__init__()
        self.in_dims = int(in_dims)
        self.rank = int(rank)
        self.out_features = int(out_features)
        self.normalize_rank = bool(normalize_rank)

        self.axis_nets = nn.ModuleList(
            [
                HSiren(
                    in_features=1,
                    out_features=self.rank,
                    hidden_layers=hidden_layers,
                    hidden_features=hidden_features,
                    first_omega_0=first_omega_0,
                    hidden_omega_0=hidden_omega_0,
                    alpha=alpha,
                )
                for _ in range(self.in_dims)
            ]
        )

        self.combine = nn.Linear(self.rank, self.out_features, bias=combine_bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        orig_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, self.in_dims)  # [N, D]

        h = None
        for d, net in enumerate(self.axis_nets):
            vd = net(coords_flat[:, d : d + 1])
            h = vd if h is None else (h * vd)

        out = self.combine(h)  # [N, out_features]
        out = out.reshape(*orig_shape, self.out_features)
        return out


# Dynamical SIREN

"""
Implementation of Dynamical SIREN
"""


class SIRENLayer(nn.Module):
    """SIREN layer with sinusoidal activation function.

    Attributes:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        omega_0 (float): Angular frequency factor
        is_first (bool): Whether this is the first layer
        linear (nn.Linear): Linear transformation layer
    """

    def __init__(
        self, input_dim: int, output_dim: int, omega_0: float = 30.0, is_first: bool = False
    ) -> None:
        """Initialize SIREN layer.

        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            omega_0: Angular frequency factor
            is_first: Whether this is the first layer
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(input_dim, output_dim)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights using uniform distribution."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.input_dim, 1 / self.input_dim)
            else:
                limit = math.sqrt(6 / self.input_dim) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sine activation.

        Args:
            x: Input tensor

        Returns:
            Output tensor after sinusoidal activation
        """
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENBlock(nn.Module):
    """Basic SIREN block for ODE dynamics."""

    def __init__(self, dim: int, omega_0: float, dropout_rate: float = 0.0) -> None:
        """Initialize SIREN block.

        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            dropout_rate: Dropout rate (applied after activation)
        """
        super().__init__()

        self.siren_layer = SIRENLayer(
            input_dim=dim, output_dim=dim, omega_0=omega_0, is_first=False
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SIREN block.

        Args:
            x: Input features

        Returns:
            Output features
        """
        out = self.siren_layer(x)
        return self.dropout(out)


class SIRENResidualBlock(nn.Module):
    """SIREN residual block for ODE dynamics."""

    def __init__(self, dim: int, omega_0: float, dropout_rate: float = 0.0) -> None:
        """Initialize SIREN residual block.

        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.siren1 = SIRENLayer(input_dim=dim, output_dim=dim, omega_0=omega_0, is_first=False)

        self.siren2 = SIRENLayer(input_dim=dim, output_dim=dim, omega_0=omega_0, is_first=False)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SIREN residual block.

        Args:
            x: Input features

        Returns:
            Output features with residual connection
        """
        identity = x

        out = self.siren1(x)
        out = self.dropout(out)

        out = self.siren2(out)
        out = self.dropout(out)

        return identity + out


class ODEFuncSIREN(nn.Module):
    """ODE dynamics function f(z, t) using SIREN blocks with concatenation-based time conditioning."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        omega_0_hidden: float,
        dropout_rate: float,
        block_type: Literal["mlp", "residual"] = "residual",
    ) -> None:
        """Initialize SIREN-based ODE function.

        Args:
            dim: Feature dimension
            num_layers: Number of SIREN layers
            omega_0_hidden: Hidden layer frequency factor
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
        """
        super().__init__()

        # Input dimension includes time (dim + 1)
        block_dim = dim + 1

        # Choose block type
        Block = SIRENResidualBlock if block_type == "residual" else SIRENBlock

        # Build SIREN layers
        self.layers = nn.ModuleList(
            [
                Block(dim=block_dim, omega_0=omega_0_hidden, dropout_rate=dropout_rate)
                for _ in range(num_layers)
            ]
        )

        # Output projection to remove time dimension
        self.output_proj = nn.Linear(block_dim, dim)

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward pass through SIREN ODE function.

        Args:
            x: State tensor of shape (B, D)
            t: Time scalar

        Returns:
            Time derivative dz/dt of shape (B, D)
        """
        # Concatenate time to features
        t_vec = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        x = torch.cat([x, t_vec], dim=1)

        # Pass through SIREN layers
        for layer in self.layers:
            x = layer(x)

        # Project back to original dimension
        return self.output_proj(x)


class DynamicalSIREN(nn.Module):
    """Dynamical SIREN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0,
        dropout_rate: float = 0.0,
        block_type: Literal["mlp", "residual"] = "residual",
        num_steps: int = 10,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        final_activation: Optional[str] = None,
    ) -> None:
        """Initialize the Dynamical SIREN model.

        Args:
            input_dim: Number of input dimensions
            hidden_dim: Width of hidden layers
            output_dim: Number of output dimensions
            num_layers: Number of hidden layers in ODE function
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
            num_steps: Number of discretization steps for the ODE
            total_time: Total integration time T for the ODE
            ot_lambda: Weight for the optimal transport regularization
            final_activation: Optional activation for the output layer

        Raises:
            ValueError: If an unsupported activation name is provided
        """
        super().__init__()

        # Validate final activation
        VALID_ACTIVATIONS = {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "ELU": nn.ELU(),
            "SELU": nn.SELU(),
            "Mish": nn.Mish(),
            "Identity": nn.Identity(),
        }

        if final_activation and final_activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported final activation: {final_activation}. "
                f"Choose from {list(VALID_ACTIVATIONS.keys())}"
            )

        self.total_time = total_time
        self.num_steps = num_steps
        self.ot_lambda = ot_lambda

        # Initial embedding: z(0) = SIREN_first_layer(x)
        # Use SIREN's first layer initialization for input embedding
        self.input_embedding = SIRENLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            omega_0=omega_0,
            is_first=True,  # Use first-layer initialization
        )

        # ODE function with SIREN dynamics and concatenation-only time conditioning
        self.ode_func = ODEFuncSIREN(
            dim=hidden_dim,
            num_layers=num_layers,
            omega_0_hidden=omega_0_hidden,
            dropout_rate=dropout_rate,
            block_type=block_type,
        )

        # Output projection matching SIREN's final layer initialization
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            limit = math.sqrt(6 / hidden_dim) / omega_0_hidden
            self.output_proj.weight.uniform_(-limit, limit)

        # Apply final activation if specified
        if final_activation:
            self.final_activation = VALID_ACTIVATIONS[final_activation]
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Dynamical SIREN.

        Args:
            x: Input coordinates of shape (batch_size, input_dim)

        Returns:
            Tuple of (network_output, ot_regularization_term)
            - network_output: shape (batch_size, output_dim)
            - ot_regularization_term: scalar tensor
        """
        # Set initial state z(0) = SIREN_embedding(x)
        z = self.input_embedding(x)

        # Solve ODE using Euler's method
        ot_accum = 0.0
        dt = self.total_time / self.num_steps
        for i in range(self.num_steps):
            t = i * dt
            v = self.ode_func(z, t)
            # Accumulate optimal transport regularization
            ot_accum = ot_accum + v.pow(2).mean()
            z = z + dt * v

        ot_reg = 0.5 * self.ot_lambda * dt * ot_accum

        # Output projection from final state z(T)
        output = self.output_proj(z)
        output = self.final_activation(output)

        return output, ot_reg

    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params


# ---- ODE building blocks (reuse your SineLayer init/behavior) ----

class HSIrenBlock(nn.Module):
    """Single H-SIREN block (no residual) for ODE dynamics, with time concatenation already included in dim."""
    def __init__(
        self,
        dim: int,
        omega_0: float,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.layer = SineLayer(
            dim,
            dim,
            is_first=False,
            omega_0=omega_0,
            hsiren=False,   # only first layer of the *whole net* is sinh
            dtype=dtype,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return self.dropout(x)


class HSIrenResidualBlock(nn.Module):
    """Residual H-SIREN block for ODE dynamics, with time concatenation already included in dim."""
    def __init__(
        self,
        dim: int,
        omega_0: float,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.layer1 = SineLayer(
            dim,
            dim,
            is_first=False,
            omega_0=omega_0,
            hsiren=False,
            dtype=dtype,
        )
        self.layer2 = SineLayer(
            dim,
            dim,
            is_first=False,
            omega_0=omega_0,
            hsiren=False,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        return identity + out


class ODEFuncHSIREN(nn.Module):
    """
    ODE dynamics function v = f(z, t) using your SineLayer-based blocks
    with concatenation-only time conditioning (exactly like your DynamicalSIREN).
    """
    def __init__(
        self,
        dim: int,
        num_layers: int,
        omega_0_hidden: float,
        dropout_rate: float = 0.0,
        block_type: Literal["mlp", "residual"] = "residual",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        block_dim = dim + 1  # time concatenated

        Block = HSIrenResidualBlock if block_type == "residual" else HSIrenBlock

        self.layers = nn.ModuleList(
            [
                Block(
                    dim=block_dim,
                    omega_0=omega_0_hidden,
                    dropout_rate=dropout_rate,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Project back to dim (remove time channel)
        self.output_proj = nn.Linear(block_dim, dim, dtype=dtype)

        # Optional: you can match Siren-style final init if you want for this proj.
        # Keeping it default is also fine; uncomment to mimic SIREN-ish scaling:
        with torch.no_grad():
            limit = math.sqrt(6 / block_dim) / omega_0_hidden
            self.output_proj.weight.uniform_(-limit, limit)

    def forward(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        # t can be python float or tensor scalar; we broadcast to (B, 1)
        if not torch.is_tensor(t):
            t_val = float(t)
            t_vec = torch.full((z.shape[0], 1), t_val, device=z.device, dtype=z.dtype)
        else:
            # tensor scalar or shape (B,) supported
            if t.ndim == 0:
                t_vec = t.expand(z.shape[0]).to(device=z.device, dtype=z.dtype).view(-1, 1)
            elif t.ndim == 1 and t.shape[0] == z.shape[0]:
                t_vec = t.to(device=z.device, dtype=z.dtype).view(-1, 1)
            else:
                raise ValueError(f"t must be a scalar or shape (B,), got shape {tuple(t.shape)}")

        x = torch.cat([z, t_vec], dim=1)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


# ---- Main model ----

class DynamicalHSiren(nn.Module):
    """
    Dynamical H-SIREN:
    - Initial embedding uses your H-Siren *first* layer behavior (sinh) via SineLayer(hsiren=True, is_first=True).
    - ODE dynamics uses standard sine layers (hsiren=False) with time concatenation.
    - Euler integration + OT regularization identical to your DynamicalSIREN.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0,
        dropout_rate: float = 0.0,
        block_type: Literal["mlp", "residual"] = "residual",
        num_steps: int = 10,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        dtype: torch.dtype = torch.float32,
        final_activation: str | Callable = "identity",
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.total_time = float(total_time)
        self.num_steps = int(num_steps)
        self.ot_lambda = float(ot_lambda)
        self.dtype = dtype

        # Initial embedding z(0) = HSiren first layer (sinh) w/ first-layer init
        # Mirrors your HSiren/Siren._build() first layer construction.
        self.input_embedding = SineLayer(
            input_dim,
            hidden_dim,
            is_first=True,
            omega_0=omega_0,
            hsiren=True,     # THIS is the H-SIREN behavior (sinh in first layer)
            alpha=alpha,
            dtype=dtype,
        )

        # ODE function
        self.ode_func = ODEFuncHSIREN(
            dim=hidden_dim,
            num_layers=num_layers,
            omega_0_hidden=omega_0_hidden,
            dropout_rate=dropout_rate,
            block_type=block_type,
            dtype=dtype,
        )

        # Output projection: match your Siren final-layer initialization
        self.output_proj = nn.Linear(hidden_dim, output_dim, dtype=dtype)
        with torch.no_grad():
            limit = math.sqrt(6 / hidden_dim) / omega_0_hidden
            self.output_proj.weight.uniform_(-limit, limit)

        # Final activation: reuse your existing activation factory for consistency
        self.final_activation = get_activation_function(final_activation, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim)

        Returns:
            output: (B, output_dim)
            ot_reg: scalar tensor
        """
        x = x.to(dtype=self.dtype)

        # z(0)
        z = self.input_embedding(x)

        dt = self.total_time / self.num_steps
        ot_accum = z.new_zeros(())  # scalar

        for i in range(self.num_steps):
            t = i * dt
            v = self.ode_func(z, t)
            ot_accum = ot_accum + v.pow(2).mean()
            z = z + dt * v

        ot_reg = 0.5 * self.ot_lambda * dt * ot_accum

        y = self.output_proj(z)
        y = self.final_activation(y)
        return y, ot_reg

    def get_param_count(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

"""
Dynamical FFNet
"""


class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping module for input coordinate lifting.

    Maps input coordinates to a higher dimensional space using random Fourier features,
    enabling better learning of high-frequency functions.

    Attributes:
        input_dim (int): Dimensionality of input coordinates
        mapping_size (int): Output dimension of the Fourier mapping
        sigma (float): Standard deviation for feature sampling
        B (nn.Parameter): Random Fourier feature matrix
    """

    def __init__(self, input_dim: int, mapping_size: int, sigma: float = 1.0) -> None:
        """Initialize Fourier feature mapping.

        Args:
            input_dim: Number of input dimensions
            mapping_size: Size of the feature mapping (must be even)
            sigma: Standard deviation for sampling feature matrix

        Raises:
            ValueError: If mapping_size is not even
        """
        super().__init__()

        if mapping_size % 2 != 0:
            raise ValueError(f"mapping_size must be even, got {mapping_size}")

        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.sigma = sigma

        # Initialize random Fourier features
        self.B = nn.Parameter(
            torch.randn(input_dim, mapping_size // 2) * sigma, requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping to input coordinates.

        Args:
            x: Input coordinates of shape (batch_size, input_dim)

        Returns:
            Fourier features of shape (batch_size, mapping_size)

        Raises:
            ValueError: If input dimensions don't match expected shape
        """
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(-1)}")

        # Project and apply sinusoidal activation
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPBlock(nn.Module):
    """Basic MLP block for ODE dynamics."""

    def __init__(self, dim: int, dropout_rate: float, activation: nn.Module) -> None:
        """Initialize MLP block.

        Args:
            dim: Feature dimension
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP block.

        Args:
            x: Input features

        Returns:
            Output features
        """
        out = self.norm(x)
        out = self.linear(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block for ODE dynamics."""

    def __init__(self, dim: int, dropout_rate: float, activation: nn.Module) -> None:
        """Initialize residual block.

        Args:
            dim: Feature dimension
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input features

        Returns:
            Output features with residual connection
        """
        identity = x

        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.dropout(out)

        return identity + out


class ODEFunc(nn.Module):
    """ODE dynamics function f(z, t) with concatenation-based time conditioning."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        dropout_rate: float,
        activation: nn.Module,
        block_type: Literal["mlp", "residual"] = "residual",
    ) -> None:
        """Initialize ODE function.

        Args:
            dim: Feature dimension
            num_layers: Number of layers
            dropout_rate: Dropout rate
            activation: Activation function
            block_type: Type of block ("mlp" or "residual")
        """
        super().__init__()

        # Input dimension includes time (dim + 1)
        block_dim = dim + 1

        # Choose block type
        Block = MLPBlock if block_type == "mlp" else ResidualBlock

        # Build layers
        self.layers = nn.ModuleList(
            [
                Block(dim=block_dim, dropout_rate=dropout_rate, activation=activation)
                for _ in range(num_layers)
            ]
        )

        # Output projection to remove time dimension
        self.output_proj = nn.Linear(block_dim, dim)

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward pass through ODE function.

        Args:
            x: State tensor of shape (B, D)
            t: Time scalar

        Returns:
            Time derivative dz/dt of shape (B, D)
        """
        # Concatenate time to features
        t_vec = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        x = torch.cat([x, t_vec], dim=1)

        # Pass through layers
        for layer in self.layers:
            x = layer(x)

        # Project back to original dimension
        return self.output_proj(x)


class DynamicalFourierFeatureNetwork(nn.Module):
    """Dynamical Fourier Feature Network."""

    VALID_ACTIVATIONS = {
        "ReLU": nn.ReLU(),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "ELU": nn.ELU(),
        "SELU": nn.SELU(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
    }

    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout_rate: float,
        activation: str,
        block_type: Literal["mlp", "residual"] = "residual",
        num_steps: int = 10,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        sigma: float = 1.0,
        final_activation: Optional[str] = None,
    ) -> None:
        """Initialize the Dynamical Fourier Feature Network model.

        Args:
            input_dim: Number of input dimensions
            mapping_size: Size of Fourier feature mapping
            hidden_dim: Width of hidden layers
            output_dim: Number of output dimensions
            num_layers: Number of layers in ODE function
            dropout_rate: Dropout rate
            activation: Activation function name
            block_type: Type of block ("mlp" or "residual")
            num_steps: Number of discretization steps for the ODE
            total_time: Total integration time T for the ODE
            ot_lambda: Weight for the optimal transport regularization
            sigma: Standard deviation for Fourier features
            final_activation: Optional activation for the output layer

        Raises:
            ValueError: If an unsupported activation name is provided
        """
        super().__init__()

        if final_activation and final_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported final activation: {final_activation}. "
                f"Choose from {list(self.VALID_ACTIVATIONS.keys())}"
            )

        self.total_time = total_time
        self.num_steps = num_steps
        self.ot_lambda = ot_lambda

        # Initial embedding: z(0) = phi(x)
        self.fourier_features = torch.jit.script(
            FourierFeatureMapping(input_dim=input_dim, mapping_size=mapping_size, sigma=sigma)
        )

        # Projection from mapping_size to hidden_dim if needed
        if mapping_size != hidden_dim:
            self.input_proj = nn.Linear(mapping_size, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # ODE function with concatenation-only time conditioning
        self.ode_func = ODEFunc(
            dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            activation=self._get_activation(activation),
            block_type=block_type,
        )

        # Output projection
        output_layers = [nn.Linear(hidden_dim, output_dim)]
        if final_activation:
            output_layers.append(self._get_activation(final_activation))
        self.output_proj = nn.Sequential(*output_layers)

    @classmethod
    def _get_activation(cls, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        if activation_name not in cls.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation_name}. "
                f"Choose from {list(cls.VALID_ACTIVATIONS.keys())}"
            )
        return cls.VALID_ACTIVATIONS[activation_name]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Dynamical Fourier Feature Network.

        Args:
            x: Input coordinates of shape (batch_size, input_dim)

        Returns:
            Tuple of (network_output)
            - network_output: shape (batch_size, output_dim)
        """
        # Set initial state z(0) = phi(x)
        z = self.fourier_features(x)
        z = self.input_proj(z)

        # Solve ODE using Euler's method
        ot_accum = 0.0
        dt = self.total_time / self.num_steps
        for i in range(self.num_steps):
            t = i * dt
            v = self.ode_func(z, t)
            # Accumulate optimal transport regularization
            ot_accum = ot_accum + v.pow(2).mean()
            z = z + dt * v

        ot_reg = 0.5 * self.ot_lambda * dt * ot_accum

        # Output projection from final state z(T)
        return self.output_proj(z), ot_reg

    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params
