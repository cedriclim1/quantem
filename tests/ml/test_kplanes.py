"""Tests for ``quantem.core.ml.models.kplanes`` construction guards."""

import io

import pytest
import torch

from quantem.core.ml.models.kplanes import KPlanes, KPlanesTILTED
from quantem.core.ml.models.so3params import SO3ParamR9SVD


def test_r9svd_as_matrix_stays_fp32_under_autocast():
    so3 = SO3ParamR9SVD(T=2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        rotation_matrices = so3.as_matrix()

    assert rotation_matrices.dtype == torch.float32


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


@pytest.mark.parametrize(
    ("model_type", "kwargs"),
    [
        (KPlanes, {}),
        (KPlanesTILTED, {"T": 2}),
    ],
)
def test_grid_parameters_are_channels_last(model_type, kwargs):
    model = model_type(M_features=2, resolution=(8, 8, 8), **kwargs)

    for plane in model.grids:
        assert plane.is_contiguous(memory_format=torch.channels_last)
        assert plane.permute(0, 2, 3, 1).is_contiguous()


def test_whole_module_load_reformats_legacy_grids():
    model = KPlanesTILTED(
        M_features=2,
        T=2,
        resolution=(8, 8, 8),
        density_activation=torch.relu,
    )
    for plane in model.grids:
        plane.data = plane.data.contiguous()

    checkpoint = io.BytesIO()
    torch.save(model, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)

    assert all(plane.is_contiguous(memory_format=torch.channels_last) for plane in restored.grids)


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


def test_tilted_forward_is_dynamo_traceable():
    """The public ``grids`` child must not be hidden behind a broken property.

    ``nn.Module.__setattr__`` registers the ParameterList as ``grids``.  Eager
    lookup tolerates a same-named property whose getter reads missing ``_grids``,
    but Dynamo traces that getter and turns its AttributeError into Unsupported.
    """
    model = KPlanesTILTED(M_features=2, T=2, resolution=(4, 4, 4))
    coords = torch.rand(8, 3) * 2 - 1

    eager = model(coords)
    compiled = torch.compile(model, backend="eager", fullgraph=True)(coords)

    torch.testing.assert_close(compiled, eager)
