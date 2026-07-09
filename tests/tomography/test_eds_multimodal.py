import numpy as np
import pytest
import torch
import torch.nn as nn

from quantem.core.ml.inr import HSiren
from quantem.core.ml.optimizer_mixin import OptimizerParams
from quantem.tomography.dataset_models import DeviceBatchSampler, TomographyEDSINRDataset
from quantem.tomography.object_models import ObjectINR
from quantem.tomography.tomography import Tomography, _multimodal_consistency_loss


def _eds_dset(
    n_proj: int = 41,
    n: int = 64,
    n_chem: int = 4,
    sparse_step: int = 5,
) -> TomographyEDSINRDataset:
    haadf = np.ones((n_proj, n, n), dtype=np.float32)
    eds = np.ones((n_chem, len(range(0, n_proj, sparse_step)), n, n), dtype=np.float32)
    angles = np.linspace(-60.003, 59.997, n_proj, dtype=np.float32)
    sparse_angles = angles[::sparse_step]
    return TomographyEDSINRDataset.from_data(
        haadf_tilt_stack=haadf,
        eds_signals_tilt_stack=eds,
        sparse_view_tilt_angles=sparse_angles,
        tilt_angles=angles,
    )


def test_eds_index_map_matches_real_sparse_angle_pattern():
    dset = _eds_dset()
    expected = torch.full((41,), -1, dtype=torch.long)
    expected[::5] = torch.arange(9)
    torch.testing.assert_close(dset.chem_idx_of_proj, expected, rtol=0, atol=0)


def test_eds_index_map_rejects_unmatched_sparse_angle():
    haadf = np.ones((41, 64, 64), dtype=np.float32)
    eds = np.ones((2, 9, 64, 64), dtype=np.float32)
    angles = np.linspace(-60.003, 59.997, 41, dtype=np.float32)
    sparse = angles[::5].copy()
    sparse[3] += 0.25

    with pytest.raises(ValueError, match="do not match"):
        TomographyEDSINRDataset.from_data(
            haadf_tilt_stack=haadf,
            eds_signals_tilt_stack=eds,
            sparse_view_tilt_angles=sparse,
            tilt_angles=angles,
        )


def test_eds_normalization_does_not_mutate_caller_array():
    haadf = np.ones((4, 5, 5), dtype=np.float32)
    eds = np.arange(2 * 2 * 5 * 5, dtype=np.float32).reshape(2, 2, 5, 5) + 1
    original = eds.copy()
    angles = np.linspace(-30, 30, 4, dtype=np.float32)
    TomographyEDSINRDataset.from_data(
        haadf_tilt_stack=haadf,
        eds_signals_tilt_stack=eds,
        sparse_view_tilt_angles=angles[::2],
        tilt_angles=angles,
    )
    np.testing.assert_array_equal(eds, original)


def test_device_batch_sampler_uses_eds_gather_targets_hook():
    dset = _eds_dset(n_proj=6, n=8, n_chem=2, sparse_step=3)
    sampler = DeviceBatchSampler(dset, batch_size=16, device="cpu", shuffle=False)
    first = next(iter(sampler))

    assert first["target_value"].shape == (16, 3)
    assert first["eds_mask"].dtype == torch.bool
    assert first["eds_mask"].all()

    second = next(iter(DeviceBatchSampler(dset, batch_size=16, device="cpu", shuffle=False)))
    item = dset[0]
    torch.testing.assert_close(second["target_value"][0], item["target_value"], rtol=0, atol=0)


def test_multimodal_loss_zero_chemical_contribution_and_defined_gradients():
    loss_mean = nn.MSELoss()
    loss_none = nn.MSELoss(reduction="none")

    pred = torch.tensor([[0.0, 100.0, -50.0], [0.0, 25.0, 75.0]], requires_grad=True)
    target = torch.zeros_like(pred)
    eds_mask = torch.zeros(2, dtype=torch.bool)

    loss = _multimodal_consistency_loss(
        pred,
        target,
        eds_mask,
        loss_mean,
        loss_none,
        haadf_weight=0.0,
        chem_loss_weight=1.0,
        legacy_masking=False,
    )
    assert loss.item() == 0.0

    head = nn.Linear(3, 3, bias=False)
    x = torch.randn(4, 3)
    target = torch.zeros(4, 3)
    loss = _multimodal_consistency_loss(
        head(x),
        target,
        torch.zeros(4, dtype=torch.bool),
        loss_mean,
        loss_none,
        haadf_weight=0.0,
        chem_loss_weight=1.0,
        legacy_masking=False,
    )
    loss.backward()
    assert head.weight.grad is not None
    assert head.weight.grad.shape == head.weight.shape
    assert torch.isfinite(head.weight.grad).all()


def test_multichannel_integrate_rays_matches_loop_reference():
    torch.manual_seed(0)
    batch = 5
    samples = 7
    channels = 3
    rays = torch.randn(batch * samples, channels)
    actual = TomographyEDSINRDataset._integrate_multichannel_rays(rays, samples, batch)

    step = 2.0 / (samples - 1)
    expected = torch.stack(
        [rays[b * samples : (b + 1) * samples].sum(dim=0) * step for b in range(batch)]
    )
    torch.testing.assert_close(actual, expected)


def test_legacy_masking_reproduces_aa621fc_formula_by_hand():
    pred = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    target = torch.tensor([[1.0, 4.0, 6.0], [8.0, 0.0, 0.0]])
    eds_mask = torch.tensor([True, False])
    haadf_weight = 0.25

    actual = _multimodal_consistency_loss(
        pred,
        target,
        eds_mask,
        nn.MSELoss(),
        nn.MSELoss(reduction="none"),
        haadf_weight=haadf_weight,
        chem_loss_weight=1.0,
        legacy_masking=True,
    )

    masked_pred = torch.tensor([[2.0, 3.0, 5.0], [7.0, 0.0, 0.0]])
    zero_fill_mse = (
        (1.0 + 1.0 + 1.0 + 1.0 + 0.0 + 0.0) / 6.0
    )
    coupling_mse = (
        (2.0 - 3.0) ** 2
        + (2.0 - 5.0) ** 2
        + (7.0 - 0.0) ** 2
        + (7.0 - 0.0) ** 2
    ) / 4.0
    expected = torch.tensor(zero_fill_mse + haadf_weight * coupling_mse)
    torch.testing.assert_close(actual, expected)
    assert masked_pred.shape == pred.shape


def test_s3im_soft_constraint_context_gets_haadf_only_for_eds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dset = _eds_dset(n_proj=2, n=4, n_chem=2, sparse_step=1)
    model = HSiren(in_features=3, out_features=3, hidden_layers=1, hidden_features=8)
    obj = ObjectINR.from_model(model, shape=(4, 4, 4), device="cpu")
    tomo = Tomography.from_models(dset=dset, obj_model=obj, device="cpu", verbose=False)

    captured = []

    def capture_soft_constraints(ctx):
        captured.append((ctx.pred.shape, ctx.target.shape))
        return torch.tensor(0.0, device=ctx.pred.device)

    obj.apply_soft_constraints = capture_soft_constraints
    tomo.reconstruct(
        num_iter=1,
        batch_size=len(dset),
        num_workers=0,
        num_samples_per_ray=4,
        optimizer_params={"object": {"default": OptimizerParams.Adam(lr=1e-4)}},
    )

    assert captured
    assert all(len(pred_shape) == 1 for pred_shape, _ in captured)
    assert all(len(target_shape) == 1 for _, target_shape in captured)


@pytest.mark.slow
def test_tiny_multimodal_cpu_reconstruct_decreases_loss(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    n = 16
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    haadf_proj = np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 18.0).astype(np.float32)
    chem0_proj = np.exp(-((xx - 6) ** 2 + (yy - 7) ** 2) / 12.0).astype(np.float32)
    chem1_proj = np.exp(-((xx - 10) ** 2 + (yy - 9) ** 2) / 14.0).astype(np.float32)

    angles = np.linspace(-60, 60, 8, dtype=np.float32)
    sparse = angles[[0, 4]]
    haadf = np.stack([haadf_proj for _ in angles])
    eds = np.stack(
        [
            np.stack([chem0_proj for _ in sparse]),
            np.stack([chem1_proj for _ in sparse]),
        ]
    ).astype(np.float32)

    dset = TomographyEDSINRDataset.from_data(
        haadf_tilt_stack=haadf,
        eds_signals_tilt_stack=eds,
        sparse_view_tilt_angles=sparse,
        tilt_angles=angles,
    )
    model = HSiren(in_features=3, out_features=3, hidden_layers=1, hidden_features=16)
    obj = ObjectINR.from_model(model, shape=(n, n, n), device="cpu")
    tomo = Tomography.from_models(dset=dset, obj_model=obj, device="cpu", verbose=False)

    tomo.reconstruct(
        num_iter=3,
        batch_size=128,
        num_workers=0,
        num_samples_per_ray=8,
        optimizer_params={"object": {"default": OptimizerParams.Adam(lr=5e-4)}},
        haadf_weight=0.1,
    )

    losses = tomo.consistency_losses
    assert len(losses) == 3
    assert losses[-1] < losses[0]
