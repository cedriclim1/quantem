"""Dispatch and model parity for the opt-in cuBLASLt sigma head."""

import copy
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from quantem.core import config
from quantem.core.ml.models import kplanes as kplanes_module
from quantem.core.ml.models.kplanes import FusedHiddenMLP, KPlanesTILTED


def _head():
    return FusedHiddenMLP(
        nn.Linear(12, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, 1),
    )


@pytest.fixture(autouse=True)
def reset_fused_mlp_fallback_state():
    kplanes_module._unsupported_fused_mlp_shapes.clear()
    kplanes_module._warned_fused_mlp_reasons.clear()
    yield
    kplanes_module._unsupported_fused_mlp_shapes.clear()
    kplanes_module._warned_fused_mlp_reasons.clear()


@pytest.mark.parametrize("value", [None, "", "1"])
def test_fused_mlp_is_default_on_and_only_zero_disables(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("QUANTEM_FUSED_MLP", raising=False)
    else:
        monkeypatch.setenv("QUANTEM_FUSED_MLP", value)
    assert kplanes_module._fused_mlp_enabled()

    monkeypatch.setenv("QUANTEM_FUSED_MLP", "0")
    assert not kplanes_module._fused_mlp_enabled()


def test_default_on_still_falls_back_for_unsupported_inputs(monkeypatch):
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
    with pytest.warns(RuntimeWarning, match="memoizing the eager fallback") as records:
        torch.testing.assert_close(head(inputs), expected)
        torch.testing.assert_close(head(inputs), expected)
    assert len(records) == 1


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
    truth_model = copy.deepcopy(eager_model)
    eager_pts = (torch.rand((37, 3), device="cuda") * 2 - 1).requires_grad_()
    fused_pts = eager_pts.detach().clone().requires_grad_()
    truth_pts = eager_pts.detach().clone().requires_grad_()
    upstream = torch.randn((37, 1), device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("QUANTEM_FUSED_MLP", "0")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_out = eager_model(eager_pts)
    eager_out.backward(upstream)

    truth_out = truth_model(truth_pts)
    truth_out.backward(upstream.float())

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
    eager_grads = (eager_pts.grad, *(parameter.grad for parameter in eager_model.parameters()))
    fused_grads = (fused_pts.grad, *(parameter.grad for parameter in fused_model.parameters()))
    truth_grads = (truth_pts.grad, *(parameter.grad for parameter in truth_model.parameters()))
    # At M=37 the measured fused/eager mean-error ratios against fp32 truth
    # were 1.11 for dx and 1.34 for dW1; allow reduction-order variation while
    # requiring the fused path to remain close to eager's fp32 error envelope.
    for fused_grad, eager_grad, truth_grad in zip(fused_grads, eager_grads, truth_grads):
        eager_error = (eager_grad.float() - truth_grad).abs()
        fused_error = (fused_grad.float() - truth_grad).abs()
        assert torch.mean(fused_error) <= torch.mean(eager_error) * 1.5 + 1e-7
        truth_scale = truth_grad.abs().max()
        bf16_eps_at_truth_scale = truth_scale * 2**-8
        assert torch.max(fused_error) <= (
            torch.max(eager_error) * 1.5 + 2 * bf16_eps_at_truth_scale
        )
