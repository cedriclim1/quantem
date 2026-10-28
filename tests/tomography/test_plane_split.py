"""KPlanesTILTEDShard on 2 and 4 CPU (gloo) ranks reproduces the unsliced KPlanesTILTED.

Every rank builds the same full model and inputs, runs it unsliced as the reference, then runs
its shard; outputs, plane TV, l2,1, the global grad norm and all gradients (planes, decoder,
replicated bias and rotations, and a pose-like coordinate shift) must match.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from quantem.core.ml.dist_utils import find_free_port
from quantem.core.ml.models.kplanes import KPlanesTILTED, KPlanesTILTEDShard, plane_split_layout


def _reference_tv(model):
    per_level = []
    for p in model.grids:
        dh = (p[:, :, 1:, :] - p[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
        dw = (p[:, :, :, 1:] - p[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
        per_level.append((dh + dw).view(model.T, 3).sum(dim=1).mean())
    return torch.stack(per_level).sum()


def _reference_l21(model):
    norms = [torch.sqrt(g.reshape(g.shape[0], -1).pow(2).mean(dim=1) + 1e-12) for g in model.grids]
    return torch.cat(norms).mean()


def _build():
    torch.manual_seed(0)
    model = KPlanesTILTED(
        M_features=4, resolution=[12, 12, 12], multiscale_res_multipliers=[0.5, 1.0], T=4,
        c2f_warmup_frac=0.5, c2f_blur_sigma0=1.5, c2f_blur_frac=0.5,
    ).double()
    with torch.no_grad():
        model.sigma_net.weight.normal_(0, 0.3)
        model.sigma_net.bias.fill_(0.2)
    model.set_progress(0.2)
    gen = torch.Generator().manual_seed(1)
    base = torch.rand(300, 3, generator=gen, dtype=torch.float64) * 1.8 - 0.9
    target = torch.rand(300, generator=gen, dtype=torch.float64)
    return model, base, target


def _loss(model, out_fn, tv_fn, l21_fn, shift, base, target):
    coords = base + shift
    dens = out_fn(coords).squeeze(-1)
    return F.mse_loss(dens, target) + 0.3 * tv_fn() + 0.1 * l21_fn(), dens


def _worker(rank, world, port, t_shards):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        full, base, target = _build()
        shift_ref = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        loss_ref, dens_ref = _loss(full, full, lambda: _reference_tv(full), lambda: _reference_l21(full), shift_ref, base, target)
        loss_ref.backward()
        norm_ref = torch.nn.utils.clip_grad_norm_(full.parameters(), max_norm=1e9)
        so3_ref = [p.grad.clone() for p in full.so3.parameters()]

        shard = KPlanesTILTEDShard(full, rank, world, t_shards)
        shard.set_progress(0.2)
        shard.zero_grad()
        shift = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        loss, dens = _loss(shard, shard, shard.plane_tv, shard.plane_l21, shift, base, target)
        loss.backward()
        shard.sync_replicated_grads()
        norm = shard.clip_grad_norm_(1e9)

        tol = dict(rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(dens, dens_ref, **tol)
        torch.testing.assert_close(loss, loss_ref, **tol)
        torch.testing.assert_close(shard.plane_tv(), _reference_tv(full), **tol)
        torch.testing.assert_close(shard.plane_l21(), _reference_l21(full), **tol)
        torch.testing.assert_close(norm, norm_ref, **tol)
        torch.testing.assert_close(shift.grad, shift_ref.grad, **tol)
        t_sl = slice(3 * shard.t0, 3 * (shard.t0 + shard.T_local))
        c_sl = slice(shard.c0, shard.c0 + shard.M_local)
        for g_shard, g_full in zip(shard.grids, full.grids):
            torch.testing.assert_close(g_shard.grad, g_full.grad[t_sl, c_sl], **tol)
        torch.testing.assert_close(shard.bias.grad, full.sigma_net.bias.grad, **tol)
        for p_shard, g_ref in zip(shard.so3.parameters(), so3_ref):
            torch.testing.assert_close(p_shard.grad, g_ref, **tol)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,t_shards", [(2, None), (4, 2), (4, 1)])
def test_plane_split_matches_unsliced(world, t_shards):
    mp.spawn(_worker, args=(world, int(find_free_port()), t_shards), nprocs=world, join=True)


def test_layout_rotations_first():
    assert plane_split_layout(16, 8, 32) == (8, 2)
    assert plane_split_layout(4, 8, 32) == (4, 1)
    assert plane_split_layout(4, 8, 32, t_shards=2) == (2, 2)
    with pytest.raises(ValueError):
        plane_split_layout(16, 8, 32, t_shards=3)
    with pytest.raises(ValueError):
        plane_split_layout(64, 1, 32)
