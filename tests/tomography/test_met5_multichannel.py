"""MET-5 regressions for multi-channel KPlanes heads and volume materialization."""

import torch
from torch import nn

from quantem.core.ml.models.kplanes import CPTilted, KPlanes, KPlanesTILTED
from quantem.tomography.object_models import ObjectINR


class CoordinatePatternModel(nn.Module):
    def __init__(self, n: int, channels: int):
        super().__init__()
        self.n = n
        self.channels = channels

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        voxel_idx = torch.round((coords + 1.0) * 0.5 * (self.n - 1)).to(torch.long)
        flat_idx = (
            voxel_idx[:, 0] * self.n * self.n + voxel_idx[:, 1] * self.n + voxel_idx[:, 2]
        ).to(coords.dtype)
        channel_idx = torch.arange(self.channels, device=coords.device, dtype=coords.dtype)
        return channel_idx.unsqueeze(0) + flat_idx.unsqueeze(1) / 1000.0


def test_create_volume_preserves_channel_identity_for_multichannel_output():
    n = 8
    channels = 5
    model = CoordinatePatternModel(n=n, channels=channels)
    obj = ObjectINR.from_model(model, shape=(n, n, n), device="cpu")

    volume = obj.create_volume(return_vol=True)

    flat_idx = torch.arange(n**3, dtype=torch.float32).reshape(n, n, n)
    expected = torch.stack([channel + flat_idx / 1000.0 for channel in range(channels)])
    assert volume.shape == (channels, n, n, n)
    assert torch.allclose(volume, expected)


def test_kplanes_tilted_multichannel_outputs_are_not_pairwise_identical():
    torch.manual_seed(7)
    model = KPlanesTILTED(
        M_features=3,
        resolution=(8, 8, 8),
        multiscale_res_multipliers=[1],
        T=2,
        out_features=5,
    )
    coords = torch.rand(32, 3) * 2 - 1

    densities = model.get_densities(coords)

    assert densities.shape == (32, 5)
    for first in range(densities.shape[1]):
        for second in range(first + 1, densities.shape[1]):
            assert not torch.allclose(densities[:, first], densities[:, second])


def test_kplanes_tilted_forward_shape_respects_out_features_and_default():
    coords = torch.rand(11, 3) * 2 - 1
    multichannel = KPlanesTILTED(
        M_features=2,
        resolution=(8, 8, 8),
        multiscale_res_multipliers=[1],
        T=2,
        out_features=5,
    )
    default = KPlanesTILTED(
        M_features=2,
        resolution=(8, 8, 8),
        multiscale_res_multipliers=[1],
        T=2,
    )

    assert multichannel(coords).shape == (11, 5)
    assert default(coords).shape == (11, 1)


def test_tensor_decomposition_heads_accept_multichannel_out_features():
    coords = torch.rand(7, 3) * 2 - 1
    kplanes = KPlanes(
        M_features=2,
        resolution=(8, 8, 8),
        multiscale_res_multipliers=[1],
        use_hybrid_mlp=True,
        out_features=5,
    )
    cp_tilted = CPTilted(C=2, resolution=(8, 8, 8), T=2, out_features=5)

    assert kplanes(coords).shape == (7, 5)
    assert cp_tilted(coords).shape == (7, 5)


def test_kplanes_tilted_out_features_one_keeps_state_dict_shapes():
    kwargs = {
        "M_features": 2,
        "resolution": (8, 8, 8),
        "multiscale_res_multipliers": [1],
        "T": 2,
    }
    default = KPlanesTILTED(**kwargs)
    explicit_single_channel = KPlanesTILTED(**kwargs, out_features=1)

    default_shapes = {key: tuple(value.shape) for key, value in default.state_dict().items()}
    explicit_shapes = {
        key: tuple(value.shape) for key, value in explicit_single_channel.state_dict().items()
    }

    assert explicit_shapes == default_shapes
    assert explicit_shapes["sigma_net.weight"] == (1, explicit_single_channel.feature_dim)
    assert explicit_shapes["sigma_net.bias"] == (1,)
