"""Preview the synthetic Au-nanoparticle tomography phantom before any training.

Builds the phantom, projects it into a tilt series (clean and Poisson-noised), and writes
slice/MIP/zoom images plus a text summary to an output directory so the phantom can be
eyeballed before the DIP refinement experiment is run.

Usage
-----
    uv run python scripts/preview_phantom.py
    uv run python scripts/preview_phantom.py --n-vox 300 --voxel 0.3 --diameter 60 \
        --dose 50 --outdir /home/cedlim/quantem/phantom_preview
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from quantem.tomography.phantoms import (  # noqa: E402
    add_poisson_noise,
    make_au_nanoparticle,
    project_au_phantom,
)

_VOL_CMAP = "inferno"
_PROJ_CMAP = "gray"


def _save_row(images, titles, path, cmap, suptitle):
    """Save a single row of images with shared per-image colorbars."""
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axs = [axs]
    for ax, img, title in zip(axs, images, titles):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vox", type=int, default=300, help="cubic volume edge in voxels")
    parser.add_argument("--voxel", type=float, default=0.3, help="Angstrom per voxel")
    parser.add_argument("--diameter", type=float, default=60.0, help="NP diameter (Angstrom)")
    parser.add_argument("--lattice-a", type=float, default=4.078, help="FCC lattice constant")
    parser.add_argument("--sigma", type=float, default=0.45, help="atom Gaussian sigma (Angstrom)")
    parser.add_argument("--n-tilts", type=int, default=5, help="number of preview tilt angles")
    parser.add_argument("--tilt-range", type=float, default=70.0, help="+/- tilt range (degrees)")
    parser.add_argument("--dose", type=float, default=50.0, help="peak counts for Poisson preview")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for Poisson noise")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/cedlim/quantem/phantom_preview",
        help="output directory for preview images",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Building Au nanoparticle phantom ({args.n_vox}^3, {args.voxel} A/voxel)...")
    phantom = make_au_nanoparticle(
        n_vox=args.n_vox,
        voxel_size=args.voxel,
        diameter_ang=args.diameter,
        lattice_a=args.lattice_a,
        atom_sigma_ang=args.sigma,
    )
    vol = phantom.volume
    nz, ny, nx = vol.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    angles = np.linspace(-args.tilt_range, args.tilt_range, args.n_tilts).astype(np.float32)
    print(f"Projecting {args.n_tilts} tilts on {device}...")
    proj_clean = project_au_phantom(phantom, angles, device=device)
    proj_noisy = add_poisson_noise(proj_clean, dose=args.dose, rng=args.seed)

    print("Writing preview images...")
    # Volume: central orthogonal slices.
    _save_row(
        [vol[cz], vol[:, cy], vol[:, :, cx]],
        [f"central Z={cz}", f"central Y={cy}", f"central X={cx}"],
        os.path.join(args.outdir, "volume_central_slices.png"),
        _VOL_CMAP,
        "Phantom volume - central slices",
    )
    # Volume: max-intensity projections (atom columns add up; checks lattice regularity).
    _save_row(
        [vol.max(0), vol.max(1), vol.max(2)],
        ["MIP along Z", "MIP along Y", "MIP along X"],
        os.path.join(args.outdir, "volume_mips.png"),
        _VOL_CMAP,
        "Phantom volume - max-intensity projections",
    )
    # Volume: a few off-center Z slices through the particle.
    half_extent = int(0.35 * args.diameter / args.voxel)
    z_slices = np.linspace(cz - half_extent, cz + half_extent, 5).astype(int)
    _save_row(
        [vol[z] for z in z_slices],
        [f"Z={z}" for z in z_slices],
        os.path.join(args.outdir, "volume_zslices.png"),
        _VOL_CMAP,
        "Phantom volume - off-center Z slices",
    )
    # Volume: zoom on the central Z slice so individual atoms are visible/resolvable.
    zoom = 30  # voxels each side of center (~6 nearest-neighbor spacings)
    crop = vol[cz, cy - zoom : cy + zoom, cx - zoom : cx + zoom]
    _save_row(
        [crop],
        [f"central Z={cz}, +/-{zoom} vox (~{2 * zoom * args.voxel:.1f} A)"],
        os.path.join(args.outdir, "volume_zoom.png"),
        _VOL_CMAP,
        "Phantom volume - zoom (atom resolvability)",
    )
    # Projections: clean and Poisson-noised.
    _save_row(
        [proj_clean[i] for i in range(len(angles))],
        [f"{a:.0f} deg" for a in angles],
        os.path.join(args.outdir, "projections_clean.png"),
        _PROJ_CMAP,
        "Simulated tilt-series projections (clean)",
    )
    _save_row(
        [proj_noisy[i] for i in range(len(angles))],
        [f"{a:.0f} deg" for a in angles],
        os.path.join(args.outdir, f"projections_poisson_dose{args.dose:.0f}.png"),
        _PROJ_CMAP,
        f"Simulated tilt-series projections (Poisson, peak dose={args.dose:.0f})",
    )

    summary = os.path.join(args.outdir, "phantom_summary.txt")
    with open(summary, "w") as fh:
        fh.write("Au nanoparticle tomography phantom\n")
        fh.write("==================================\n")
        fh.write(f"volume shape         : {vol.shape}\n")
        fh.write(f"voxel size           : {args.voxel} Angstrom/voxel\n")
        fh.write(f"box size             : {args.n_vox * args.voxel:.1f} Angstrom\n")
        fh.write(f"NP diameter          : {args.diameter:.1f} Angstrom\n")
        fh.write(f"lattice constant a   : {phantom.lattice_a:.4f} Angstrom (FCC)\n")
        fh.write(
            f"nearest-neighbor     : {phantom.nn_dist:.3f} Angstrom = "
            f"{phantom.nn_dist_vox:.2f} voxels\n"
        )
        fh.write(
            f"atom sigma           : {args.sigma} Angstrom = "
            f"{args.sigma / args.voxel:.2f} voxels\n"
        )
        fh.write(f"n atoms              : {phantom.n_atoms}\n")
        fh.write(f"volume min/max       : {vol.min():.4f} / {vol.max():.4f}\n")
        fh.write(f"tilt angles (deg)    : {np.array2string(angles, precision=1)}\n")
        fh.write(f"Poisson peak dose    : {args.dose:.0f} counts\n")
        fh.write(f"clean proj min/max   : {proj_clean.min():.3f} / {proj_clean.max():.3f}\n")
        fh.write(f"noisy proj min/max   : {proj_noisy.min():.3f} / {proj_noisy.max():.3f}\n")
    print(f"  wrote {summary}")
    print(f"\nDone. {phantom.n_atoms} atoms. Review images in: {args.outdir}")


if __name__ == "__main__":
    main()
