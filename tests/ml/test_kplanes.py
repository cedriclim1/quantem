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


def _tilted_fused_available():
    import torch

    from quantem.core.ml.models import kplanes as kplanes_mod

    return torch.cuda.is_available() and kplanes_mod._kplanes_tilted_fuse_cuda is not None


class TestTiltedFusedDispatchParity:
    """interpolate_ms_features_tilted must give the same result through the
    fused quantem-cuda kernel and the torch fallback path."""

    def _inputs(self, device="cuda", dtype=None, T=3):
        import torch
        from torch import nn

        dtype = dtype or torch.float32
        gen = torch.Generator(device=device).manual_seed(0)
        pts = (
            torch.empty(513, 3, device=device, dtype=torch.float32).uniform_(
                -1.1, 1.1, generator=gen
            )
        ).to(dtype)
        rotations = (
            torch.empty(T, 3, 3, device=device, dtype=torch.float32).uniform_(
                -1.0, 1.0, generator=gen
            )
        ).to(dtype)
        grids = nn.ParameterList(
            nn.Parameter(
                torch.empty(3 * T, C, R, R, device=device, dtype=torch.float32)
                .uniform_(0.1, 0.5, generator=gen)
                .to(dtype)
            )
            for C, R in ((4, 16), (4, 32))
        )
        return pts, grids, rotations

    @pytest.mark.skipif(
        not _tilted_fused_available(), reason="requires a CUDA device and quantem-cuda"
    )
    def test_forward_matches_torch_path(self, monkeypatch):
        import torch

        from quantem.core.ml.models import kplanes as kplanes_mod
        from quantem.core.ml.models.kplanes import interpolate_ms_features_tilted

        pts, grids, rotations = self._inputs()
        out_fused = interpolate_ms_features_tilted(pts, grids, rotations)
        monkeypatch.setattr(kplanes_mod, "_kplanes_tilted_fuse_cuda", None)
        out_torch = interpolate_ms_features_tilted(pts, grids, rotations)
        assert out_fused.shape == out_torch.shape
        torch.testing.assert_close(out_fused, out_torch, rtol=1e-4, atol=5e-6)

    @pytest.mark.skipif(
        not _tilted_fused_available(), reason="requires a CUDA device and quantem-cuda"
    )
    def test_gradients_match_torch_path(self, monkeypatch):
        import torch

        from quantem.core.ml.models import kplanes as kplanes_mod
        from quantem.core.ml.models.kplanes import interpolate_ms_features_tilted

        pts, grids, rotations = self._inputs()
        pts.requires_grad_(True)
        rotations.requires_grad_(True)
        T = rotations.shape[0]
        upstream = torch.randn(513, T * sum(g.shape[1] for g in grids), device="cuda")

        def run():
            for t in (pts, rotations, *grids):
                t.grad = None
            (interpolate_ms_features_tilted(pts, grids, rotations) * upstream).sum().backward()
            return [pts.grad.clone(), rotations.grad.clone()] + [g.grad.clone() for g in grids]

        g_fused = run()
        monkeypatch.setattr(kplanes_mod, "_kplanes_tilted_fuse_cuda", None)
        g_torch = run()
        for gf, gt in zip(g_fused, g_torch):
            torch.testing.assert_close(gf, gt, rtol=1e-3, atol=1e-5)

    @pytest.mark.skipif(
        not _tilted_fused_available(), reason="requires a CUDA device and quantem-cuda"
    )
    def test_non_fp32_falls_back(self):
        import torch

        from quantem.core.ml.models.kplanes import interpolate_ms_features_tilted

        pts, grids, rotations = self._inputs(dtype=torch.float64)
        out = interpolate_ms_features_tilted(pts, grids, rotations)
        assert out.dtype == torch.float64

    def test_cpu_path_unaffected(self):
        import torch

        from quantem.core.ml.models.kplanes import interpolate_ms_features_tilted

        pts, grids, rotations = self._inputs(device="cpu")
        out = interpolate_ms_features_tilted(pts, grids, rotations)
        assert out.shape == (513, 3 * 8) and out.dtype == torch.float32
