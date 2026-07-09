"""Tests for ``quantem.core.ml.models.kplanes`` construction guards."""

import pytest

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
    """Regression: KPlanes(use_hybrid_mlp=False) -- the constructor default --
    built no sigma_net at all, so forward / get_params / ObjectTensorDecomp
    .from_model crashed with AttributeError. KPlanesTILTED and CPTilted both
    fall back to a linear head; plain KPlanes must do the same."""

    def test_default_get_params(self):
        model = KPlanes(M_features=2, resolution=(8, 8, 8))
        params = model.get_params()
        assert set(params) == set(model.param_keys)
        assert all(len(v) > 0 for v in params.values())

    def test_default_forward(self):
        import torch

        model = KPlanes(M_features=2, resolution=(8, 8, 8))
        out = model(torch.rand(5, 3) * 2 - 1)
        assert out.shape == (5, 1)
        assert torch.isfinite(out).all()
