"""End-to-end optimizer parity for combined multiscale interpolation and plane TV."""

import copy

import pytest
import torch

from quantem.core import config
from quantem.core.ml.models.kplanes import KPlanesTILTED

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@requires_cuda
def test_combined_plane_tv_matches_separate_path_over_optimizer_steps(monkeypatch):
    try:
        import quantem.cuda.core.ml as cuda_ml
    except (ImportError, RuntimeError):
        pytest.skip("quantem-cuda is unavailable")
    if not hasattr(cuda_ml, "kplanes_tilted_fuse_ms_tv"):
        pytest.skip("combined multiscale/plane-TV capability is unavailable")

    original_get = config.get

    def enabled_cuda_config(key, default=None):
        if key in {"has_quantem_cuda", "use_cuda_kernels"}:
            return True
        return original_get(key, default=default)

    monkeypatch.setattr(config, "get", enabled_cuda_config)
    monkeypatch.setenv("QUANTEM_KPLANES_MS_TV_FUSED", "1")
    torch.manual_seed(41)
    separate = KPlanesTILTED(
        T=1,
        M_features=2,
        resolution=(5, 5, 5),
        multiscale_res_multipliers=(1, 2, 3),
    ).cuda()
    combined = copy.deepcopy(separate)
    coords = torch.tensor([[0.125, -0.25, 0.375]], device="cuda")
    tv_weight = 0.0375
    separate_optim = torch.optim.SGD(separate.parameters(), lr=2e-4)
    combined_optim = torch.optim.SGD(combined.parameters(), lr=2e-4)

    for _ in range(3):
        separate_optim.zero_grad(set_to_none=True)
        combined_optim.zero_grad(set_to_none=True)
        separate._plane_tv_fusion_requested = False
        combined._plane_tv_fusion_requested = True

        separate_density = separate(coords)
        combined_density, combined_tv = combined(coords)
        separate_tv = cuda_ml.plane_tv_loss(*separate.grids)
        separate_loss = separate_density.square().mean() + tv_weight * separate_tv
        combined_loss = combined_density.square().mean() + tv_weight * combined_tv
        torch.testing.assert_close(combined_loss, separate_loss, rtol=2e-6, atol=2e-7)

        separate_loss.backward()
        combined_loss.backward()
        for separate_grid, combined_grid in zip(separate.grids, combined.grids):
            torch.testing.assert_close(
                combined_grid.grad, separate_grid.grad, rtol=1e-6, atol=1e-7
            )
        for separate_param, combined_param in zip(separate.parameters(), combined.parameters()):
            torch.testing.assert_close(
                combined_param.grad, separate_param.grad, rtol=1e-6, atol=1e-7
            )

        separate_optim.step()
        combined_optim.step()
        for separate_param, combined_param in zip(separate.parameters(), combined.parameters()):
            torch.testing.assert_close(combined_param, separate_param, rtol=1e-6, atol=1e-7)
