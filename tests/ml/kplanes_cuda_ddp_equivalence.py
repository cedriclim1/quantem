"""Multi-rank equivalence and speed check for KPlanesTILTED CUDA dispatch.

Run on one multi-GPU node with the quantem-cuda runtime available, for example::

    torchrun --standalone --nproc-per-node=2 \
        tests/ml/kplanes_cuda_ddp_equivalence.py

The script compares identical DDP training trajectories with the fused interpolation
kernel enabled and disabled, verifies synchronized gradient norms on every rank, and
then reports paired training iterations/second on the same rank-to-GPU assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from quantem.core import config
from quantem.core.ml.models.kplanes import KPlanesTILTED


@dataclass
class TrainingTrace:
    losses: torch.Tensor
    grad_norms: torch.Tensor
    parameters: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--benchmark-batch-size", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--transforms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--rtol", type=float, default=2.0e-3)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> KPlanesTILTED:
    return KPlanesTILTED(
        M_features=args.features,
        resolution=(args.resolution,) * 3,
        multiscale_res_multipliers=[1, 2],
        T=args.transforms,
        out_features=1,
    ).to(device)


def initial_state(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    model = build_model(args, torch.device("cpu"))
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def make_batches(
    *,
    steps: int,
    batch_size: int,
    seed: int,
    rank: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1009 * rank)
    batches = []
    for _ in range(steps):
        coords = torch.rand(batch_size, 3, generator=generator, device=device) * 2 - 1
        target = (
            0.35
            + 0.20 * coords[:, :1].square()
            + 0.10 * torch.sin(2.0 * coords[:, 1:2])
            + 0.05 * coords[:, 2:3]
        )
        batches.append((coords, target))
    return batches


def gather_scalar(value: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.zeros_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value)
    return torch.stack(gathered)


def gradient_norm(model: DDP) -> torch.Tensor:
    squared = [parameter.grad.detach().float().square().sum() for parameter in model.parameters()]
    return torch.stack(squared).sum().sqrt()


def run_training_trace(
    *,
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int,
    world_size: int,
    state: dict[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    kernels_enabled: bool,
) -> TrainingTrace:
    config.set({"use_cuda_kernels": kernels_enabled})
    model = build_model(args, device)
    model.load_state_dict(state)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=2.0e-3)
    losses = []
    grad_norms = []

    for coords, target in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(ddp_model(coords), target)
        loss.backward()
        grad_norm = gradient_norm(ddp_model)
        losses.append(gather_scalar(loss.detach(), world_size))
        grad_norms.append(gather_scalar(grad_norm, world_size))
        optimizer.step()

    parameters = [parameter.detach().clone() for parameter in ddp_model.module.parameters()]
    return TrainingTrace(
        losses=torch.stack(losses),
        grad_norms=torch.stack(grad_norms),
        parameters=parameters,
    )


def assert_equivalent(
    on: TrainingTrace, off: TrainingTrace, *, rtol: float, atol: float
) -> dict[str, float]:
    torch.testing.assert_close(on.losses, off.losses, rtol=rtol, atol=atol)
    torch.testing.assert_close(on.grad_norms, off.grad_norms, rtol=rtol, atol=atol)

    for trace in (on, off):
        reference = trace.grad_norms[:, :1].expand_as(trace.grad_norms)
        torch.testing.assert_close(trace.grad_norms, reference, rtol=1.0e-6, atol=1.0e-7)

    for on_parameter, off_parameter in zip(on.parameters, off.parameters):
        torch.testing.assert_close(on_parameter, off_parameter, rtol=rtol, atol=atol)

    loss_delta = (on.losses - off.losses).abs()
    grad_delta = (on.grad_norms - off.grad_norms).abs()
    loss_rel = loss_delta / off.losses.abs().clamp_min(atol)
    grad_rel = grad_delta / off.grad_norms.abs().clamp_min(atol)
    grad_spread_on = (on.grad_norms.max(dim=1).values - on.grad_norms.min(dim=1).values).abs()
    grad_spread_off = (
        off.grad_norms.max(dim=1).values - off.grad_norms.min(dim=1).values
    ).abs()
    return {
        "loss_max_abs": float(loss_delta.max().item()),
        "loss_max_rel": float(loss_rel.max().item()),
        "grad_norm_max_abs": float(grad_delta.max().item()),
        "grad_norm_max_rel": float(grad_rel.max().item()),
        "rank_grad_spread_on": float(grad_spread_on.max().item()),
        "rank_grad_spread_off": float(grad_spread_off.max().item()),
    }


def benchmark(
    *,
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int,
    state: dict[str, torch.Tensor],
    batch: tuple[torch.Tensor, torch.Tensor],
    kernels_enabled: bool,
) -> float:
    config.set({"use_cuda_kernels": kernels_enabled})
    model = build_model(args, device)
    model.load_state_dict(state)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=2.0e-3)
    coords, target = batch

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(ddp_model(coords), target)
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(args.benchmark_steps):
        step()
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - started, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return args.benchmark_steps / float(elapsed.item())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires CUDA.")
    if not config.get("has_quantem_cuda"):
        raise RuntimeError("quantem.cuda is unavailable; load its CUDA runtime before running.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if not 2 <= world_size <= 4:
        raise RuntimeError(f"Expected 2-4 torchrun ranks, got {world_size}.")

    try:
        state = initial_state(args)
        batches = make_batches(
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed + 1,
            rank=rank,
            device=device,
        )

        import quantem.cuda.core.ml as cuda_ml

        real_kernel = cuda_ml.kplanes_tilted_fuse
        kernel_calls = 0

        def counted_kernel(pts, rotations, plane):
            nonlocal kernel_calls
            kernel_calls += 1
            return real_kernel(pts, rotations, plane)

        cuda_ml.kplanes_tilted_fuse = counted_kernel
        trace_on = run_training_trace(
            args=args,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            state=state,
            batches=batches,
            kernels_enabled=True,
        )
        calls_after_on = kernel_calls
        trace_off = run_training_trace(
            args=args,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            state=state,
            batches=batches,
            kernels_enabled=False,
        )
        if calls_after_on == 0:
            raise AssertionError("The kernels-on DDP trace never dispatched the fused kernel.")
        if kernel_calls != calls_after_on:
            raise AssertionError("The kernels-off DDP trace dispatched the fused kernel.")
        cuda_ml.kplanes_tilted_fuse = real_kernel

        equivalence = assert_equivalent(
            trace_on, trace_off, rtol=args.rtol, atol=args.atol
        )

        benchmark_batch = make_batches(
            steps=1,
            batch_size=args.benchmark_batch_size,
            seed=args.seed + 2,
            rank=rank,
            device=device,
        )[0]
        iterations_per_second_off = benchmark(
            args=args,
            device=device,
            local_rank=local_rank,
            state=state,
            batch=benchmark_batch,
            kernels_enabled=False,
        )
        iterations_per_second_on = benchmark(
            args=args,
            device=device,
            local_rank=local_rank,
            state=state,
            batch=benchmark_batch,
            kernels_enabled=True,
        )

        if rank == 0:
            result = {
                "world_size": world_size,
                "equivalence_steps": args.steps,
                "equivalence_batch_size_per_rank": args.batch_size,
                "kernel_calls_per_rank": calls_after_on,
                **equivalence,
                "benchmark_batch_size_per_rank": args.benchmark_batch_size,
                "benchmark_steps": args.benchmark_steps,
                "iterations_per_second_kernels_off": iterations_per_second_off,
                "iterations_per_second_kernels_on": iterations_per_second_on,
                "speedup": iterations_per_second_on / iterations_per_second_off,
            }
            print("DDP_KPLANES_CUDA_RESULT=" + json.dumps(result, sort_keys=True))
    finally:
        config.set({"use_cuda_kernels": True})
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
