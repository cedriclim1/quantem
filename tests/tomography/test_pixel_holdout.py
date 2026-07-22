import numpy as np
import torch

from quantem.tomography.dataset_models import (
    DeviceBatchSampler,
    TomographyINRDataset,
    build_pixel_holdout_split,
)


def _stack(n_proj=3, n=10):
    stack = np.zeros((n_proj, n, n), dtype=np.float32)
    foreground = stack.reshape(-1)[::7]
    values = np.linspace(1.0, 5.0, min(20, len(foreground)), dtype=np.float32)
    foreground[: len(values)] = values
    return stack


def _flat(batch, n):
    return batch["projection_idx"] * (n * n) + batch["pixel_i"] * n + batch["pixel_j"]


def test_holdout_split_is_seeded_and_foreground_aware():
    stack = _stack()
    first = build_pixel_holdout_split(stack, holdout_fraction=0.2, holdout_seed=11)
    repeat = build_pixel_holdout_split(stack, holdout_fraction=0.2, holdout_seed=11)
    different = build_pixel_holdout_split(stack, holdout_fraction=0.2, holdout_seed=12)

    torch.testing.assert_close(first.val_indices, repeat.val_indices, rtol=0, atol=0)
    assert not torch.equal(first.val_indices, different.val_indices)
    assert first.val_fg_indices.numel() > 0
    assert first.val_bg_indices.numel() > 0


def test_holdout_fraction_arithmetic_is_exact():
    stack = _stack(n_proj=2, n=10)
    split = build_pixel_holdout_split(stack, holdout_fraction=0.15, holdout_seed=0)
    n_pixels = stack.size
    n_holdout = int(n_pixels * 0.15)

    assert split.val_indices.numel() == n_holdout
    assert split.train_indices.numel() == n_pixels - n_holdout
    assert split.val_fg_indices.numel() + split.val_bg_indices.numel() == n_holdout


def test_training_and_holdout_indices_are_disjoint():
    stack = _stack()
    split = build_pixel_holdout_split(stack, holdout_fraction=0.25, holdout_seed=3)
    all_indices = torch.cat([split.train_indices, split.val_indices])

    assert all_indices.unique().numel() == stack.size
    assert torch.isin(split.train_indices, split.val_indices).sum().item() == 0


def test_ddp_training_shards_exclude_same_holdout_pixels():
    stack = _stack(n_proj=4, n=12)
    angles = np.linspace(-60, 60, stack.shape[0], dtype=np.float32)
    dset = TomographyINRDataset.from_data(stack, angles)
    split = build_pixel_holdout_split(dset.tilt_stack, holdout_fraction=0.1, holdout_seed=9)
    holdout = set(split.val_indices.tolist())

    shards = []
    for rank in range(3):
        sampler = DeviceBatchSampler(
            dset,
            batch_size=16,
            device="cpu",
            indices=split.train_indices,
            rank=rank,
            world_size=3,
        )
        sampler.set_epoch(0)
        flats = torch.cat([_flat(batch, stack.shape[1]) for batch in sampler])
        assert not holdout.intersection(flats.tolist())
        shards.append(flats)

    all_train_seen = torch.cat(shards)
    assert all_train_seen.unique().numel() == all_train_seen.numel()


def test_validation_sampler_keeps_partial_final_batch():
    stack = _stack(n_proj=2, n=8)
    angles = np.linspace(-60, 60, stack.shape[0], dtype=np.float32)
    dset = TomographyINRDataset.from_data(stack, angles)
    split = build_pixel_holdout_split(stack, holdout_fraction=0.05, holdout_seed=4)

    sampler = DeviceBatchSampler(
        dset,
        batch_size=1024,
        device="cpu",
        indices=split.val_indices,
        shuffle=False,
        drop_last=False,
    )

    assert len(sampler) == 1
    batch = next(iter(sampler))
    assert len(batch["target_value"]) == split.val_indices.numel()


def test_per_row_angles_apply_to_all_holdout_samplers():
    stack = _stack(n_proj=4, n=12)
    angles = np.linspace(-60, 60, stack.shape[0], dtype=np.float32)
    row_angles = np.linspace(
        -62.4, 59.1, stack.shape[0] * stack.shape[1], dtype=np.float32
    ).reshape(stack.shape[:2])
    dset = TomographyINRDataset.from_data(
        stack,
        angles,
        tilt_angles_per_row=row_angles,
    )
    split = build_pixel_holdout_split(dset.tilt_stack, holdout_fraction=0.2, holdout_seed=7)
    sampler_configs = (
        (split.train_indices, True),
        (split.val_indices, False),
        (split.val_fg_indices, False),
        (split.val_bg_indices, False),
    )

    for indices, shuffle in sampler_configs:
        sampler = DeviceBatchSampler(
            dset,
            batch_size=17,
            device="cpu",
            indices=indices,
            shuffle=shuffle,
            drop_last=False,
        )
        for batch in sampler:
            flat_indices = _flat(batch, stack.shape[1])
            for batch_idx, flat_idx in enumerate(flat_indices.tolist()):
                item = dset[flat_idx]
                for key in ("projection_idx", "pixel_i", "pixel_j", "phi", "target_value"):
                    torch.testing.assert_close(
                        batch[key][batch_idx],
                        torch.as_tensor(item[key], dtype=batch[key].dtype),
                        rtol=0,
                        atol=0,
                    )
