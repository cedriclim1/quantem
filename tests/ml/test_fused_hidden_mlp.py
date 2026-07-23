"""Dispatch and model parity for the opt-in cuBLASLt sigma head."""

import copy
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from quantem.core import config
from quantem.core.ml.models.kplanes import FusedHiddenMLP, KPlanesTILTED


def _head():
    return FusedHiddenMLP(
        nn.Linear(12, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, 1),
    )


def test_kill_switch_is_default_off(monkeypatch):
    monkeypatch.delenv("QUANTEM_FUSED_MLP", raising=False)
    head = _head()
    inputs = torch.randn(7, 12)
    expected = nn.Sequential.forward(head, inputs)
    torch.testing.assert_close(head(inputs), expected)


def test_unsupported_device_and_autocast_off_fall_back(monkeypatch):
    monkeypatch.setenv("QUANTEM_FUSED_MLP", "1")
    head = _head()
    inputs = torch.randn(7, 12)
    expected = nn.Sequential.forward(head, inputs)
    torch.testing.assert_close(head(inputs), expected)


def test_missing_extension_capability_falls_back(monkeypatch):
    monkeypatch.setenv("QUANTEM_FUSED_MLP", "1")
    head = _head()
    inputs = torch.randn(7, 12)
    expected = nn.Sequential.forward(head, inputs)
    monkeypatch.setitem(sys.modules, "quantem.cuda.core.ml", ModuleType("quantem.cuda.core.ml"))
    monkeypatch.setattr(FusedHiddenMLP, "_can_fuse", lambda self, tensor: True)
    torch.testing.assert_close(head(inputs), expected)


def test_available_extension_is_selected_without_changing_parameter_keys(monkeypatch):
    monkeypatch.setenv("QUANTEM_FUSED_MLP", "1")
    head = _head()
    inputs = torch.randn(7, 12)
    calls = []
    cuda_ml = ModuleType("quantem.cuda.core.ml")

    def fake_fused(x, w1, b1, w2, b2, w3, b3):
        calls.append((x, w1, b1, w2, b2, w3, b3))
        return torch.full((x.shape[0], w3.shape[0]), 4.0)

    cuda_ml.fused_hidden_mlp = fake_fused
    monkeypatch.setitem(sys.modules, "quantem.cuda.core.ml", cuda_ml)
    monkeypatch.setattr(FusedHiddenMLP, "_can_fuse", lambda self, tensor: True)

    output = head(inputs)
    assert torch.equal(output, torch.full((7, 1), 4.0))
    assert len(calls) == 1
    assert set(head.state_dict()) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    }


def test_torch_compile_uses_noninterfering_fallback(monkeypatch):
    monkeypatch.setenv("QUANTEM_FUSED_MLP", "1")
    head = _head()
    inputs = torch.randn(7, 12, requires_grad=True)
    expected = head(inputs)
    compiled = torch.compile(head, backend="eager")
    actual = compiled(inputs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert inputs.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fp32_autocast_off_falls_back(monkeypatch):
    monkeypatch.setenv("QUANTEM_FUSED_MLP", "1")
    head = _head().cuda()
    inputs = torch.randn(7, 12, device="cuda")
    expected = nn.Sequential.forward(head, inputs)
    torch.testing.assert_close(head(inputs), expected)


requires_fused_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not config.get("has_quantem_cuda"),
    reason="requires CUDA and quantem-cuda",
)


@requires_fused_cuda
def test_kplanes_tilted_forward_backward_parity(monkeypatch):
    import quantem.cuda.core.ml as cuda_ml

    torch.manual_seed(723)
    eager_model = KPlanesTILTED(
        M_features=8,
        T=1,
        resolution=(8, 8, 8),
        multiscale_res_multipliers=(1, 1, 1),
        density_activation=lambda value: value,
        use_hybrid_mlp=True,
        hybrid_hidden_dim=16,
        hybrid_num_layers=2,
    ).cuda()
    fused_model = copy.deepcopy(eager_model)
    eager_pts = (torch.rand((37, 3), device="cuda") * 2 - 1).requires_grad_()
    fused_pts = eager_pts.detach().clone().requires_grad_()
    upstream = torch.randn((37, 1), device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("QUANTEM_FUSED_MLP", "0")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_out = eager_model(eager_pts)
    eager_out.backward(upstream)

    real_fused = cuda_ml.fused_hidden_mlp
    calls = []

    def spy(*args):
        calls.append(args[0].shape)
        return real_fused(*args)

    monkeypatch.setattr(cuda_ml, "fused_hidden_mlp", spy)
    monkeypatch.setenv("QUANTEM_FUSED_MLP", "1")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused_out = fused_model(fused_pts)
    fused_out.backward(upstream)

    assert calls == [torch.Size((37, 24))]
    torch.testing.assert_close(fused_out, eager_out, rtol=8e-3, atol=1e-3)
    torch.testing.assert_close(fused_pts.grad, eager_pts.grad, rtol=3e-3, atol=1e-4)
    for (_, eager_parameter), (_, fused_parameter) in zip(
        eager_model.named_parameters(), fused_model.named_parameters()
    ):
        torch.testing.assert_close(
            fused_parameter.grad, eager_parameter.grad, rtol=3e-3, atol=1e-4
        )
