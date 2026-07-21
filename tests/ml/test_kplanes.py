"""Tests for ``quantem.core.ml.models.kplanes`` construction and training controls."""

import pytest
import torch

from quantem.core.ml.models.kplanes import KPlanes, KPlanesTILTED


class TestResolutionValidation:
    def test_isotropic_builds(self):
        model = KPlanes(M_features=2, resolution=(16, 16, 16))
        assert model.grids[0].shape == (3, 2, 16, 16)

    def test_anisotropic_raises(self):
        """Regression: the plane grids are allocated as (res[1], res[0]) for all three
        axis pairs, ignoring res[2] -- anisotropic resolutions silently gave the
        XZ/YZ planes the wrong grid along z instead of erroring."""
        with pytest.raises(ValueError, match="isotropic"):
            KPlanes(M_features=2, resolution=(16, 16, 8))
        with pytest.raises(ValueError, match="isotropic"):
            KPlanesTILTED(M_features=2, T=2, resolution=(16, 8, 16))


class TestDefaultHeadConstruction:
    """The default non-hybrid KPlanes decoder is a usable linear head."""

    @pytest.mark.parametrize("out_features", [1, 4])
    def test_default_get_params(self, out_features):
        model = KPlanes(M_features=2, resolution=(8, 8, 8), out_features=out_features)
        params = model.get_params()
        assert set(params) == set(model.param_keys)
        assert all(len(values) > 0 for values in params.values())

    @pytest.mark.parametrize("out_features", [1, 4])
    def test_default_forward(self, out_features):
        model = KPlanes(M_features=2, resolution=(8, 8, 8), out_features=out_features)
        output = model(torch.rand(5, 3) * 2 - 1)
        assert output.shape == (5, out_features)
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("so3_param_type", ["r9svd", "quat"])
@pytest.mark.parametrize("out_features", [1, 4])
def test_tilted_c2f_gating_and_so3_parameterizations(so3_param_type, out_features):
    """Every supported rotation representation works with scalar and EDS heads."""
    torch.manual_seed(0)
    model = KPlanesTILTED(
        M_features=2,
        resolution=(4, 4, 4),
        multiscale_res_multipliers=[1, 2, 3],
        T=2,
        tau_init="random",
        so3_param_type=so3_param_type,
        c2f_warmup_frac=0.5,
        out_features=out_features,
        density_activation=lambda x: x,
    )

    assert model._scale_gates == [0.0, 0.0, 0.0]
    model.set_progress(0.25)
    assert model._scale_gates == [1.0, 0.5, 0.0]

    model.set_progress(0.5)
    assert model._scale_gates == [1.0, 1.0, 1.0]
    rotations = model.so3.as_matrix()
    identity = torch.eye(3).expand_as(rotations)
    torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), identity)
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(model.T))

    coords = torch.rand(5, 3) * 2 - 1
    output = model(coords)
    assert output.shape == (5, out_features)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.so3.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.so3.parameters())
