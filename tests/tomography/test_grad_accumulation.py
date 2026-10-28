"""Gradient accumulation (``grad_accum_steps``) and chunked volume rendering for INR tomography.

The equivalence runs refine pose, enable pose TV and every object soft term the frozen
benchmark configurations use (S3IM, sparsity, distortion, volume TV, and for KPlanesTILTED
plane TV and l21), so batch-coupled terms, ragged box-mode sample means and ray-independent
terms are all covered. They use SGD so float summation-order differences in the accumulated
gradient are compared directly instead of being rescaled by Adam's normalisation.
"""

import numpy as np
import pytest
import torch

from quantem.core.ml.inr import HSiren
from quantem.core.ml.models.kplanes import KPlanesTILTED
from quantem.core.ml.optimizer_mixin import OptimizerParams
from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.object_models import ObjectINR, ObjectTensorDecomp
from quantem.tomography.tomography import Tomography, _chunk_bounds

N = 8
BATCH = 64
STEPS = 3
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-6

CONSTRAINTS = {
    "hsiren": {
        "name": "obj_inr",
        "tv_vol": 1e-2,
        "sparsity": 1e-2,
        "s3im_weight": 1.0,
        "distortion": 1e-2,
    },
    "kplanes": {
        "name": "obj_tensor_decomp",
        "tv_vol": 1e-2,
        "tv_plane": 1e-2,
        "sparsity": 1e-2,
        "s3im_weight": 1.0,
        "l21_plane": 1e-2,
        "distortion": 1e-2,
    },
}


def _build(kind: str) -> Tomography:
    rng = np.random.default_rng(0)
    stack = (rng.random((5, N, N)) * 5.0 + 0.5).astype(np.float32)
    angles = np.linspace(-60, 60, 5).astype(np.float32)
    torch.manual_seed(0)
    if kind == "hsiren":
        obj = ObjectINR.from_model(
            HSiren(hidden_layers=1, hidden_features=16), shape=(N, N, N), device="cpu"
        )
    else:
        model = KPlanesTILTED(
            M_features=2, resolution=(N, N, N), multiscale_res_multipliers=[0.5, 1.0], T=2
        )
        obj = ObjectTensorDecomp.from_model(model, shape=(N, N, N), device="cpu")
    dset = TomographyINRDataset.from_data(stack, angles, learn_shift=True, learn_tilt_axis=True)
    return Tomography.from_models(dset=dset, obj_model=obj, device="cpu", verbose=False)


def _optimizer_params(kind: str) -> dict:
    sgd = OptimizerParams.SGD
    obj = sgd(lr=1e-2)
    if kind == "kplanes":
        obj = {"grids": sgd(lr=1e-1), "sigma_net": sgd(lr=1e-2), "so3": sgd(lr=1e-2)}
    return {"object": obj, "pose": {"pose_shift": sgd(lr=1e-1), "pose_tilt_axis": sgd(lr=1e-1)}}


def _parameters(tomo: Tomography) -> list[torch.Tensor]:
    return list(tomo.obj_model.model.parameters()) + list(tomo.dset.parameters())


def _run(kind: str, grad_accum_steps: int, **kwargs) -> dict:
    tomo = _build(kind)
    grads, snapshots = [], []
    step_optimizers = tomo.step_optimizers

    def record_and_step():
        grads.append([None if p.grad is None else p.grad.clone() for p in _parameters(tomo)])
        step_optimizers()

    tomo.step_optimizers = record_and_step
    torch.manual_seed(1)
    tomo.reconstruct(
        num_iter=2,
        max_grad_steps=STEPS,
        batch_size=BATCH,
        num_workers=0,
        optimizer_params=_optimizer_params(kind),
        obj_constraints=CONSTRAINTS[kind],
        dset_constraints={"name": "base_tomography_dataset", "tv_zs": 1e-2, "tv_shifts": 1e-2},
        use_bfloat16=False,
        snapshot_every=2,
        snapshot_steps={3},
        snapshot_callback=lambda step, volume: snapshots.append(step),
        grad_accum_steps=grad_accum_steps,
        **kwargs,
    )
    return {
        "tomo": tomo,
        "grads": grads,
        "params": [p.detach().clone() for p in _parameters(tomo)],
        "snapshots": snapshots,
    }


def _assert_same_run(ref: dict, run: dict) -> None:
    assert len(run["grads"]) == len(ref["grads"]) == STEPS
    for step_ref, step_run in zip(ref["grads"], run["grads"]):
        for g_ref, g_run in zip(step_ref, step_run):
            if g_ref is None:
                assert g_run is None or not g_run.any()
            else:
                torch.testing.assert_close(g_run, g_ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    for p_ref, p_run in zip(ref["params"], run["params"]):
        torch.testing.assert_close(p_run, p_ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    ref_tomo, run_tomo = ref["tomo"], run["tomo"]
    assert run_tomo._grad_steps == ref_tomo._grad_steps == STEPS
    assert run_tomo.num_epochs == ref_tomo.num_epochs
    assert run["snapshots"] == ref["snapshots"] == [2, 3]
    np.testing.assert_allclose(run_tomo._epoch_losses, ref_tomo._epoch_losses, rtol=1e-5)
    np.testing.assert_allclose(
        run_tomo._consistency_losses, ref_tomo._consistency_losses, rtol=1e-5
    )
    assert run_tomo._soft_term_losses[-1].keys() == ref_tomo._soft_term_losses[-1].keys()
    for name, value in ref_tomo._soft_term_losses[-1].items():
        assert run_tomo._soft_term_losses[-1][name] == pytest.approx(value, rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("kind", ["hsiren", "kplanes"])
@pytest.mark.parametrize("grad_accum_steps", [2, 4])
def test_accumulated_steps_match_the_full_batch(kind, grad_accum_steps):
    _assert_same_run(_run(kind, 1), _run(kind, grad_accum_steps))


@pytest.mark.parametrize("kind", ["hsiren", "kplanes"])
def test_accumulated_validation_matches_the_full_batch(kind):
    ref = _run(kind, 1, holdout_fraction=0.2)
    run = _run(kind, 3, holdout_fraction=0.2)
    assert len(ref["tomo"]._val_losses) == len(run["tomo"]._val_losses) == 1
    np.testing.assert_allclose(run["tomo"]._val_losses, ref["tomo"]._val_losses, rtol=1e-5)


def test_k1_is_the_single_pass_path(monkeypatch):
    tomo = _build("hsiren")
    monkeypatch.setattr(
        tomo, "_accumulate_gradients", lambda **kw: pytest.fail("k=1 must not accumulate")
    )
    tomo.reconstruct(
        num_iter=1,
        max_grad_steps=1,
        batch_size=BATCH,
        num_workers=0,
        optimizer_params=_optimizer_params("hsiren"),
    )
    assert tomo._grad_steps == 1


@pytest.mark.parametrize("value", [0, -2, 1.5])
def test_grad_accum_steps_rejects_invalid(value):
    tomo = _build("hsiren")
    with pytest.raises(ValueError, match="grad_accum_steps"):
        tomo.reconstruct(num_iter=1, batch_size=BATCH, num_workers=0, grad_accum_steps=value)


def test_chunk_bounds_cover_every_point_in_order():
    assert _chunk_bounds(10, 3) == [(0, 3), (3, 6), (6, 10)]
    assert _chunk_bounds(2, 4) == [(0, 0), (0, 1), (1, 1), (1, 2)]
    assert _chunk_bounds(0, 2) == [(0, 0), (0, 0)]


def _meshgrid_volume(obj: ObjectINR, chunk: int) -> torch.Tensor:
    coords_1d = torch.linspace(-1, 1, N)
    x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    inputs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    with torch.no_grad():
        outputs = [
            obj.apply_hard_constraints(obj.model(inputs[start : start + chunk])).reshape(-1, 1)
            for start in range(0, N**3, chunk)
        ]
    return torch.cat(outputs).reshape(1, N, N, N)


@pytest.mark.parametrize("kind", ["hsiren", "kplanes"])
def test_default_volume_is_bitwise_the_meshgrid_path(kind):
    obj = _build(kind).obj_model
    ref = _meshgrid_volume(obj, 5 * N * N)
    torch.testing.assert_close(obj.create_volume(return_vol=True), ref, rtol=0, atol=0)
    np.testing.assert_array_equal(obj.obj_view, ref.numpy().transpose(0, 1, 3, 2))


@pytest.mark.parametrize("kind", ["hsiren", "kplanes"])
@pytest.mark.parametrize("chunk_size", [1, 37, 64, N**3])
def test_volume_chunk_size_only_changes_float_rounding(kind, chunk_size):
    obj = _build(kind).obj_model
    torch.testing.assert_close(
        obj.create_volume(return_vol=True, chunk_size=chunk_size),
        _meshgrid_volume(obj, chunk_size),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        obj.create_volume(return_vol=True, chunk_size=chunk_size),
        obj.create_volume(return_vol=True),
        rtol=1e-5,
        atol=1e-5,
    )


def test_volume_chunk_size_is_capped_and_validated(monkeypatch):
    obj = _build("hsiren").obj_model
    calls = []
    model = obj.model

    def recording_forward(coords):
        calls.append(coords.shape[0])
        return model.__class__.forward(model, coords)

    monkeypatch.setattr(model, "forward", recording_forward)
    monkeypatch.setattr(type(obj), "VOLUME_CHUNK_SIZE", 100)
    obj.create_volume()
    assert max(calls) == 100 and sum(calls) == N**3
    with pytest.raises(ValueError, match="chunk_size"):
        obj.create_volume(chunk_size=0)
