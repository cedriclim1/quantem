# quantem

This is the home repository for the quantitative electron microscopy (quantem) data analysis toolkit.

## Installation Instructions

The package is available on the Python Package Index (PyPi), as [quantem](https://pypi.org/project/quantem/).

You can install it using `pip install quantem`.

For a developer install, please refer to [CONTRIBUTORS.md](CONTRIBUTORS.md).

### Optional extras

#### Tomography (`tomography`)

Installs [FastTomo](https://github.com/cedriclim1/FastTomo) and a
self-contained CUDA 13.0 toolkit (nvcc, cccl, runtime, crt, nvvm) as
PyPI wheels for GPU-accelerated tomographic reconstruction. No system
or conda CUDA install is required; FastTomo's CMake auto-detects the
wheel layout under `site-packages/nvidia/cu13/` and falls back to a
CPU-only build on platforms without those wheels (e.g. macOS).

Either of the following works:

```bash
# uv (developer install)
uv sync --python 3.14 --extra tomography
```

```bash
# plain pip (clean conda env, Colab, or any Python 3.14 venv)
pip install -e ".[tomography]"

# or as a one-liner from git (e.g. on Colab):
pip install "git+https://github.com/electronmicroscopy/quantem[tomography]"
```

The pip path is the recommended option for Google Colab notebooks —
Colab provides `nvcc` at `/usr/local/cuda/bin/nvcc`, so FastTomo's
CMake `check_language(CUDA)` picks it up and builds the GPU kernels;
the `nvidia-cuda-*` wheels installed alongside FastTomo provide the
runtime libraries.

> [!IMPORTANT]
> The tomography extra requires **Python 3.14** (gated via PEP 508
> markers, pending validation on older interpreters). On Python 3.12
> or 3.13 the marker filters out — `uv sync --extra tomography` or
> `pip install ".[tomography]"` succeeds but installs no tomography
> deps and emits no warning. Pass `--python 3.14` to `uv sync` (or
> create a Python 3.14 venv before `pip install`) to actually get the
> extra.
>
> CUDA wheels resolve only on `linux_x86_64`, `linux_aarch64`, and
> `win_amd64`. Other platforms get a CPU-only FastTomo build.
>
> On a fully clean venv with no `nvcc` available (no system CUDA, no
> conda `cuda-nvcc`, not Colab), pip's build isolation prevents
> FastTomo from seeing the `nvidia-cuda-nvcc` wheel during its CMake
> build, so it silently produces a CPU-only build. Either install
> conda `cuda-nvcc` first, or use the `uv sync` path which mirrors
> the CUDA wheels into the FastTomo build environment.

## License

quantem is free and open source software, distributed under the [MIT License](LICENSE).
