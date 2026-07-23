"""Reference and capability coverage for tomography's trunc-exp activation."""

import torch

from quantem.core.ml.activation_functions import TruncExpActivation, trunc_exp


def test_trunc_exp_cpu_forward_and_clamped_backward():
    values = torch.tensor([-2.0, 0.5, 20.0], requires_grad=True)
    upstream = torch.tensor([0.25, -0.5, 0.75])
    output = trunc_exp(values, offset=1.25)
    output.backward(upstream)

    shifted = values.detach() - 1.25
    torch.testing.assert_close(output, torch.exp(shifted))
    torch.testing.assert_close(values.grad, upstream * torch.exp(shifted.clamp(max=15)))


def test_trunc_exp_capability_is_explicit_on_function_and_module():
    activation = TruncExpActivation(offset=1.5)
    assert trunc_exp.quantem_guarantees_nonnegative is True
    assert activation.quantem_guarantees_nonnegative is True
    assert activation.offset == 1.5
