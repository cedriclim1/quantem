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

```bash
uv sync --python 3.14 --extra tomography
```

> [!IMPORTANT]
> The tomography extra requires **Python 3.14** (gated via PEP 508
> markers, pending validation on older interpreters). On Python 3.12
> or 3.13 the marker filters out — `uv sync --extra tomography`
> succeeds but installs no tomography deps and emits no warning.
> Pass `--python 3.14` (or set `.python-version` to 3.14) to actually
> get the extra.
>
> CUDA wheels resolve only on `linux_x86_64`, `linux_aarch64`, and
> `win_amd64`. Other platforms get a CPU-only FastTomo build.

## License

quantem is free and open source software, distributed under the [MIT License](LICENSE).
