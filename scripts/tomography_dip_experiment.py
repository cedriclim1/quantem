"""TILTED denoising / atomicity experiment on an Au-nanoparticle phantom.

Method families (select with ``--methods``) per dose, on one shared reconstruction:

    baselines    raw / gaussian / median on the final TILTED volume
    dip          detached 3D-U-Net DIP, {warm,random} x {bbox,full}, oracle F1/PSNR stops
    tilted_cv    cross-validation early stop -- the TILTED volume at the held-out (per-pixel)
                 val-loss minimum (a ground-truth-free stop)
    cv_then_dip  detached DIP applied to the CV-early-stopped volume
    joint        data-coupled CNN head: V = CNN(V0) reprojected through the INR ray forward,
                 stopped on held-out tilts (deployable, GT-free); oracle F1 logged too
    alternating  plug-and-play: INR <-> DIP with a volume-space prior (CNN as regularizer)

The phantom's ground-truth atom catalog lets every volume be scored on detection F1 and
PSNR. The headline question: does a GT-free stop (CV val loss / held-out tilts) land near
the oracle-F1 win? Results are a long-format CSV (rewritten after every row) plus per-run
metric-vs-iteration and per-volume view plots.

Quick smoke run::

    uv run python scripts/tomography_dip_experiment.py --n-vox 32 --diameter 8 --n-tilts 9 \
        --recon-iters 4 --dip-iters 6 --joint-iters 4 --alt-rounds 2 --alt-chunk 2 \
        --alt-dip-iters 3 --eval-every 2 --val-stride 3 --doses clean --inputs random \
        --num-workers 2 --outdir /tmp/dip_exp_smoke

Full run (defaults are the 300^3 phantom)::

    uv run python scripts/tomography_dip_experiment.py \
        --outdir /home/cedlim/quantem/dip_experiment/02_cv_inloop

Each ``--outdir`` is self-contained: ``results.csv`` (rewritten after every row),
``curves/`` (metric-vs-iteration plots) and ``views/`` (per-volume projections/slices).
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from quantem.core.ml.models.kplanes import KPlanesTILTED  # noqa: E402
from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams  # noqa: E402
from quantem.tomography.dataset_models import TomographyINRDataset  # noqa: E402
from quantem.tomography.object_models import ObjectTensorDecomp  # noqa: E402
from quantem.tomography.phantoms import (  # noqa: E402
    add_poisson_noise,
    make_au_nanoparticle,
    project_au_phantom,
)
from quantem.tomography.post_processing import (  # noqa: E402
    JointRefiner,
    ObjectDIPRefiner,
    detection_f1,
    gaussian_baseline,
    identity_baseline,
    median_baseline,
    psnr_mse,
    score_volume,
)
from quantem.tomography.tomography import Tomography  # noqa: E402


def _build_tomo(series, angles, *, device, num_iter):
    """Construct a TILTED (KPlanes tensor-decomposition) Tomography ready to reconstruct."""
    n = int(series.shape[1])
    model = KPlanesTILTED(M_features=8, resolution=(n, n, n), multiscale_res_multipliers=[1], T=4)
    obj = ObjectTensorDecomp.from_model(model, shape=(n, n, n), device=device)
    dset = TomographyINRDataset.from_data(series, angles)
    return Tomography.from_models(dset=dset, obj_model=obj, device=device, verbose=False)


def _recon_kwargs(num_iter, samples_per_ray, batch_size, num_workers):
    return dict(
        optimizer_params={
            "object": {
                "grids": OptimizerParams.Adam(lr=1e-2),
                "sigma_net": OptimizerParams.Adam(lr=1e-3),
                "so3": OptimizerParams.Adam(lr=1e-2),
            },
            "pose": OptimizerParams.Adam(lr=1e-2),
        },
        scheduler_params={
            "object": SchedulerParams.CosineAnnealing(T_max=num_iter),
            "pose": SchedulerParams.CosineAnnealing(T_max=num_iter),
        },
        num_iter=num_iter,
        batch_size=batch_size,
        num_samples_per_ray=samples_per_ray,
        num_workers=num_workers,
    )


def _obj_volume(tomo):
    """Densify the current object model to a (Z, Y, X) float32 array."""
    view = tomo.obj_model.obj_view
    return np.asarray(view[0] if view.ndim == 4 else view, dtype=np.float32)


def reconstruct_tilted(
    series,
    angles,
    *,
    device,
    num_iter,
    samples_per_ray,
    batch_size,
    num_workers,
    val_fraction=0.0,
    track_best_val=False,
):
    """Run a TILTED reconstruction; return ``(final_volume, tomo)``.

    With ``track_best_val`` and ``val_fraction > 0`` the tomo also carries a best-validation
    checkpoint (``tomo.load_best_val_state()``) for the cross-validation early-stop variant.
    """
    tomo = _build_tomo(series, angles, device=device, num_iter=num_iter)
    tomo.reconstruct(
        **_recon_kwargs(num_iter, samples_per_ray, batch_size, num_workers),
        val_fraction=val_fraction,
        track_best_val=track_best_val,
    )
    return _obj_volume(tomo), tomo


def alternating_refine(
    series,
    angles,
    *,
    device,
    samples_per_ray,
    batch_size,
    num_workers,
    rounds,
    chunk,
    lam,
    dip_iters,
    dip_lr,
    seed,
):
    """Plug-and-play alternation: INR reconstruct -> DIP-denoise volume -> re-train INR with
    a volume-space prior pulling it toward the denoised volume. Returns the final volume."""
    tomo = _build_tomo(series, angles, device=device, num_iter=chunk)
    opt_params = {
        "object": {
            "grids": OptimizerParams.Adam(lr=1e-2),
            "sigma_net": OptimizerParams.Adam(lr=1e-3),
            "so3": OptimizerParams.Adam(lr=1e-2),
        },
        "pose": OptimizerParams.Adam(lr=1e-2),
    }
    vol = None
    for r in range(rounds):
        prior = None if vol is None else torch.as_tensor(vol, dtype=torch.float32, device=device)
        # Optimizer is set once (keeps momentum across rounds); the cosine schedule warm-
        # restarts each round (T_max=chunk).
        tomo.reconstruct(
            num_iter=chunk,
            optimizer_params=opt_params if r == 0 else None,
            scheduler_params={"object": SchedulerParams.CosineAnnealing(T_max=chunk)},
            batch_size=batch_size,
            num_samples_per_ray=samples_per_ray,
            num_workers=num_workers,
            volume_prior=prior,
            volume_prior_weight=lam,
        )
        recon = _obj_volume(tomo)
        # Denoise the current reconstruction with a random-input DIP (oracle-free stop: take
        # the final DIP volume after a short fit -- the convolutional prior is the regularizer).
        refiner = ObjectDIPRefiner.from_volume(
            recon, input_mode="random", spatial_mode="bbox", device=device, seed=seed
        )
        res = refiner.refine(
            num_iters=dip_iters,
            optimizer_params=OptimizerParams.Adam(lr=dip_lr),
            verbose=False,
        )
        vol = res.final_volume
    return _obj_volume(tomo)


def plot_views(vol, path, title, voxel):
    """Projected potential (sum along Z), central Z slice, and a zoom on that slice."""
    cz, cy, cx = (s // 2 for s in vol.shape)
    zoom = min(30, cy, cx)  # ~6 NN spacings at 0.3 A/voxel, matching preview_phantom
    images = [vol.sum(0), vol[cz], vol[cz, cy - zoom : cy + zoom, cx - zoom : cx + zoom]]
    titles = [
        "projected potential (sum Z)",
        f"central slice Z={cz}",
        f"zoom +/-{zoom} vox (~{2 * zoom * voxel:.0f} A)",
    ]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, img, t in zip(axs, images, titles):
        im = ax.imshow(img, cmap="inferno")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history, path, title):
    """Plot F1 and PSNR against iteration on twin axes; mark the two oracle stops."""
    its = history["iter"]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("DIP iteration")
    ax1.set_ylabel("F1", color="tab:blue")
    ax1.plot(its, history["f1"], color="tab:blue", label="F1")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("PSNR (dB)", color="tab:red")
    ax2.plot(its, history["psnr"], color="tab:red", label="PSNR")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.axvline(its[int(np.argmax(history["f1"]))], color="tab:blue", ls="--", alpha=0.6)
    ax2.axvline(its[int(np.argmax(history["psnr"]))], color="tab:red", ls=":", alpha=0.6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-vox", type=int, default=300)
    p.add_argument("--voxel", type=float, default=0.3)
    p.add_argument("--diameter", type=float, default=60.0)
    p.add_argument("--n-tilts", type=int, default=41)
    p.add_argument("--tilt-range", type=float, default=70.0)
    p.add_argument(
        "--doses",
        nargs="+",
        default=["clean", "50"],
        help="'clean' and/or peak-count doses, e.g. clean 100 50 25",
    )
    p.add_argument("--inputs", nargs="+", default=["warm", "random"], choices=["warm", "random"])
    p.add_argument("--spatial", nargs="+", default=["bbox"], choices=["bbox", "full"])
    p.add_argument(
        "--methods",
        nargs="+",
        default=["baselines", "dip", "tilted_cv", "cv_then_dip", "joint", "alternating"],
        choices=["baselines", "dip", "tilted_cv", "cv_then_dip", "joint", "alternating"],
        help="which method families to run",
    )
    p.add_argument("--recon-iters", type=int, default=300)
    p.add_argument("--samples-per-ray", type=int, default=None, help="default: n-vox")
    p.add_argument("--recon-batch", type=int, default=2048)
    p.add_argument("--dip-iters", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--dip-lr", type=float, default=1e-3)
    p.add_argument("--gauss-sigma", type=float, default=1.0)
    # Cross-validation / data-coupled options.
    p.add_argument("--val-fraction", type=float, default=0.1, help="held-out ray-pixel fraction")
    p.add_argument("--val-stride", type=int, default=5, help="JointRefiner held-out tilt stride")
    p.add_argument("--joint-iters", type=int, default=1500)
    p.add_argument("--joint-input", default="random", choices=["warm", "random"])
    p.add_argument("--joint-batch", type=int, default=4096)
    # Alternating plug-and-play options.
    p.add_argument("--alt-rounds", type=int, default=5)
    p.add_argument("--alt-chunk", type=int, default=60, help="INR iters per alternation round")
    p.add_argument("--alt-lambda", type=float, default=0.1, help="volume-prior weight")
    p.add_argument("--alt-dip-iters", type=int, default=400, help="inner DIP iters per round")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="/home/cedlim/quantem/dip_experiment")
    args = p.parse_args()

    # ObjectTensorDecomp.setup_distributed needs an explicit device index (cuda:0, not cuda).
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    spr = args.samples_per_ray or args.n_vox
    rng = np.random.default_rng(args.seed)

    print(f"Building phantom ({args.n_vox}^3) and projecting {args.n_tilts} tilts...")
    phantom = make_au_nanoparticle(
        n_vox=args.n_vox, voxel_size=args.voxel, diameter_ang=args.diameter
    )
    angles = np.linspace(-args.tilt_range, args.tilt_range, args.n_tilts).astype(np.float32)
    clean_series = project_au_phantom(phantom, angles, device=device)
    gt = phantom.volume
    score_kw = dict(voxel_size=phantom.voxel_size, nn_dist_ang=phantom.nn_dist)

    def full_score(vol):
        return score_volume(vol, phantom.atom_coords, gt, **score_kw)

    def cheap_cb(vol):
        d = detection_f1(vol, phantom.atom_coords, **score_kw)
        psnr, mse = psnr_mse(vol, gt)
        return {"f1": d.f1, "psnr": psnr, "mse": mse}

    views_dir = os.path.join(args.outdir, "views")
    curves_dir = os.path.join(args.outdir, "curves")
    os.makedirs(views_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    plot_views(gt, os.path.join(views_dir, "ground_truth.png"), "ground truth", args.voxel)

    rows: list[dict] = []
    csv_path = os.path.join(args.outdir, "results.csv")

    def write_csv():
        # Rewrite the whole CSV after every record so a crash never loses scored rows.
        lead = ["dose", "method", "stop", "f1", "precision", "recall", "valley_contrast", "psnr"]
        extra = [k for k in {k for r in rows for k in r} if k not in lead]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=lead + extra)
            writer.writeheader()
            writer.writerows(rows)

    def record(dose, method, stop, vol):
        rows.append({"dose": dose, "method": method, "stop": stop, **full_score(vol)})
        stem = f"{dose}_{method}" + ("" if stop == "n/a" else f"_{stop}")
        plot_views(vol, os.path.join(views_dir, f"{stem}.png"), stem, args.voxel)
        write_csv()

    def run_dip(vol, tag, name, input_mode, spatial_mode):
        """Detached DIP refinement of ``vol`` with oracle F1/PSNR stop reporting."""
        refiner = ObjectDIPRefiner.from_volume(
            vol, input_mode=input_mode, spatial_mode=spatial_mode, device=device, seed=args.seed
        )
        res = refiner.refine(
            num_iters=args.dip_iters,
            optimizer_params=OptimizerParams.Adam(lr=args.dip_lr),
            metric_callback=cheap_cb,
            eval_every=args.eval_every,
            track_best=(("f1", "max"), ("psnr", "max")),
            verbose=False,
        )
        if "f1" in res.best:
            record(tag, name, "f1opt", res.best["f1"]["volume"])
        if "psnr" in res.best:
            record(tag, name, "psnropt", res.best["psnr"]["volume"])
        record(tag, name, "final", res.final_volume)
        plot_curves(
            res.history, os.path.join(curves_dir, f"{tag}_{name}.png"), f"{tag} {name}"
        )

    need_cv = any(m in args.methods for m in ("tilted_cv", "cv_then_dip"))

    for dose_str in args.doses:
        dose = None if dose_str == "clean" else float(dose_str)
        series = add_poisson_noise(clean_series, dose=dose, rng=rng)
        tag = "clean" if dose is None else f"dose{dose:.0f}"
        print(f"\n=== {tag}: TILTED reconstruction ({args.recon_iters} iters) ===")
        recon, tomo = reconstruct_tilted(
            series,
            angles,
            device=device,
            num_iter=args.recon_iters,
            samples_per_ray=spr,
            batch_size=args.recon_batch,
            num_workers=args.num_workers,
            val_fraction=args.val_fraction if need_cv else 0.0,
            track_best_val=need_cv,
        )

        if "baselines" in args.methods:
            record(tag, "raw", "n/a", identity_baseline(recon))
            record(tag, "gaussian", "n/a", gaussian_baseline(recon, args.gauss_sigma))
            record(tag, "median", "n/a", median_baseline(recon, 3))

        if "dip" in args.methods:
            for input_mode in args.inputs:
                for spatial_mode in args.spatial:
                    name = f"dip_{input_mode}_{spatial_mode}"
                    print(f"  -> {name}")
                    run_dip(recon, tag, name, input_mode, spatial_mode)

        # Cross-validation early-stopped INR (GT-free stop = val-loss minimum).
        cv_vol = None
        if need_cv and tomo.best_val_epoch is not None:
            tomo.load_best_val_state()
            cv_vol = _obj_volume(tomo)
            if "tilted_cv" in args.methods:
                record(tag, "tilted_cv", f"ep{tomo.best_val_epoch}", cv_vol)
            if "cv_then_dip" in args.methods:
                print("  -> cv_then_dip")
                run_dip(cv_vol, tag, "cv_then_dip", "random", "bbox")

        # Data-coupled CNN head; deployable stop = held-out-tilt loss.
        if "joint" in args.methods:
            print("  -> joint (CNN head + held-out-tilt stop)")
            jr = JointRefiner.from_volume_and_dset(
                recon,
                tomo.dset,
                val_stride=args.val_stride,
                input_mode=args.joint_input,
                spatial_mode="bbox",
                batch_size=args.joint_batch,
                num_samples_per_ray=spr,
                device=device,
                seed=args.seed,
            )
            res = jr.refine(
                num_iters=args.joint_iters,
                optimizer_params=OptimizerParams.Adam(lr=args.dip_lr),
                metric_callback=cheap_cb,
                eval_every=args.eval_every,
                track_best=(("val_data", "min"), ("f1", "max")),
                early_stop=("val_data", "min", 8),
                verbose=False,
            )
            if "val_data" in res.best:
                record(tag, "joint", "valstop", res.best["val_data"]["volume"])
            if "f1" in res.best:
                record(tag, "joint", "f1opt", res.best["f1"]["volume"])
            record(tag, "joint", "final", res.final_volume)
            plot_curves(
                res.history, os.path.join(curves_dir, f"{tag}_joint.png"), f"{tag} joint"
            )

        # Alternating plug-and-play (fresh reconstruction; mutates its own tomo).
        if "alternating" in args.methods:
            print("  -> alternating (plug-and-play)")
            alt_vol = alternating_refine(
                series,
                angles,
                device=device,
                samples_per_ray=spr,
                batch_size=args.recon_batch,
                num_workers=args.num_workers,
                rounds=args.alt_rounds,
                chunk=args.alt_chunk,
                lam=args.alt_lambda,
                dip_iters=args.alt_dip_iters,
                dip_lr=args.dip_lr,
                seed=args.seed,
            )
            record(tag, "alternating", "final", alt_vol)

    print(f"\nWrote {len(rows)} rows to {csv_path}")
    print("Plots + CSV in:", args.outdir)


if __name__ == "__main__":
    main()
