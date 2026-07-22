import contextlib
import copy
import datetime
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping, Self, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch._tensor import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from quantem.core.io.serialize import AutoSerialize, load

"""Logger class for AD/ML reconstruction methods."""


class LoggerBase(AutoSerialize):
    """Logger for AD/ML reconstruction methods.

    Parameters
    ----------
    mode : {"tensorboard", "wandb"}, optional
        Logging backend. TensorBoard preserves the historical behavior. WandB runs default to
        offline mode unless ``WANDB_MODE`` is already set, and their files are written under
        ``<log_dir>/wandb/``. Upload offline runs later with ``wandb sync <log_dir>/wandb/<run>``.
    wandb_config : Mapping, optional
        Configuration attached to ``wandb.init(config=...)``. Ignored by TensorBoard mode.
    """

    def __init__(
        self,
        base_log_dir: os.PathLike | str,
        run_prefix: str,
        run_suffix: str = "",
        log_images_every: int = 10,
        mode: Literal["tensorboard", "wandb"] | str = "tensorboard",
        wandb_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize LoggerBase.

        Parameters
        ----------
        base_log_dir : os.PathLike or str
            Base directory for log files.
        run_prefix : str
            Prefix for run directory name.
        run_suffix : str, optional
            Suffix for run directory name, by default ""
        log_images_every : int, optional
            Frequency for logging images, by default 10
        mode : {"tensorboard", "wandb"}, optional
            Logging backend, by default "tensorboard"
        wandb_config : Mapping, optional
            Configuration attached to WandB runs. No-op for TensorBoard.
        """
        self._timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # This should never be reinstantiated.
        # DDP: only the main rank owns the writer and the on-disk run directory.
        # Non-main ranks build a fully inert logger -- no SummaryWriter/WandB run, no
        # directory created, and every log_*/flush/close call is a no-op -- so a DDP
        # run yields a single run subdirectory instead of one per rank. RANK is unset
        # for single-process runs, so this defaults to rank 0 and preserves behavior.
        self._is_writer_rank = int(os.environ.get("RANK", "0")) == 0
        self.run_prefix = run_prefix
        self.run_suffix = run_suffix
        self.log_dir = base_log_dir
        self.log_images_every = log_images_every
        self.mode = mode
        self._wandb_run = None
        self._wandb = None
        self.writer = None

        if self._is_writer_rank:
            if self.mode == "tensorboard":
                self.writer = SummaryWriter(str(self.log_dir))
            else:
                self._init_wandb(wandb_config)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
        step_domain: str = "epoch",
        extra_steps: dict[str, int] | None = None,
    ) -> None:
        if not self._is_writer_rank:
            return
        if self.mode == "tensorboard":
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        else:
            self._log_wandb(tag, float(value), step, step_domain, extra_steps=extra_steps)

    def log_image(
        self,
        tag: str,
        image: NDArray | Tensor,
        step: int,
        cmap: str = "turbo",
        step_domain: str = "epoch",
        extra_steps: dict[str, int] | None = None,
    ) -> None:
        if not self._is_writer_rank:
            return
        cmap_image = self.apply_colormap(image, cmap_name=cmap)
        if self.mode == "tensorboard":
            self.writer.add_image(tag, cmap_image, step)
        else:
            image_hwc = np.moveaxis(cmap_image, 0, -1)
            self._log_wandb(
                tag, self._wandb.Image(image_hwc), step, step_domain, extra_steps=extra_steps
            )

    def log_figure(
        self,
        tag: str,
        fig: Figure,
        step: int,
        step_domain: str = "epoch",
        extra_steps: dict[str, int] | None = None,
    ) -> None:
        if not self._is_writer_rank:
            return
        if self.mode == "tensorboard":
            self.writer.add_figure(tag, fig, step)
        else:
            self._log_wandb(tag, self._wandb.Image(fig), step, step_domain, extra_steps=extra_steps)

    def log_histogram(
        self,
        tag: str,
        values: NDArray | Tensor,
        step: int,
        step_domain: str = "epoch",
        extra_steps: dict[str, int] | None = None,
    ) -> None:
        """Log histogram of values for monitoring distributions.

        Parameters
        ----------
        tag : str
            Tag for the histogram.
        values : NDArray or Tensor
            Values to create histogram from.
        step : int
            Step number.
        """
        if not self._is_writer_rank:
            return
        if isinstance(values, Tensor):
            values = values.detach().cpu().numpy()
        if self.mode == "tensorboard":
            self.writer.add_histogram(tag, values, step)
        else:
            self._log_wandb(
                tag, self._wandb.Histogram(values), step, step_domain, extra_steps=extra_steps
            )

    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
        step_domain: str = "epoch",
        extra_steps: dict[str, int] | None = None,
    ) -> None:
        """Log text for configuration, hyperparameters, or notes.

        Parameters
        ----------
        tag : str
            Tag for the text.
        text : str
            Text to log.
        step : int
            Step number.
        """
        if not self._is_writer_rank:
            return
        if self.mode == "tensorboard":
            self.writer.add_text(tag, text, step)
        else:
            self._log_wandb(tag, text, step, step_domain, extra_steps=extra_steps)

    def attach_config(self, config: Mapping[str, Any]) -> None:
        """Attach a resolved run configuration to WandB.

        This is intentionally a no-op for TensorBoard mode so callers can use it unconditionally.
        """
        if not self._is_writer_rank:
            return
        if self.mode == "wandb":
            self._wandb_run.config.update(dict(config), allow_val_change=True)

    def flush(self) -> None:
        if self._is_writer_rank and self.mode == "tensorboard":
            self.writer.flush()

    def close(self) -> None:
        if not self._is_writer_rank:
            return
        if self.mode == "tensorboard":
            self.writer.flush()
            self.writer.close()
        elif self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def new_timestamp(self) -> None:
        """Create new timestamp and reinitialize writer with new log directory."""
        self.close()
        self._timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.run_prefix + "_" + self._timestamp
        if self.run_suffix:
            name += f"_{self.run_suffix}"
        new_log_dir = self.log_dir.parent / name
        self._log_dir = new_log_dir
        if not self._is_writer_rank:
            return
        new_log_dir.mkdir(exist_ok=True)
        if self.mode == "tensorboard":
            self.writer = SummaryWriter(str(self.log_dir))
        else:
            self._init_wandb()

    def clone(self) -> Self:
        """Create a cloned logger with new timestamp.

        Returns
        -------
        Self
            Cloned logger instance.
        """
        try:
            cloned: Self = copy.deepcopy(self)
        except Exception:
            # using tempfile saving as fallback
            tmp_path = Path(tempfile.gettempdir()) / f"logger_clone_{self.run_prefix}.zip"
            try:
                self.save(
                    tmp_path,
                    mode="o",
                    store="zip",
                )
                cloned = cast(Self, load(tmp_path))
            finally:
                with contextlib.suppress(Exception):
                    tmp_path.unlink()
        cloned.new_timestamp()

        # copy old log file to new log dir
        files = list(self.log_dir.glob("events.out.tfevents.*"))
        for file in files:
            shutil.copy(file, cloned.log_dir)
        return cloned

    # --- Properties ---

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @log_dir.setter
    def log_dir(self, dir: str | os.PathLike) -> None:
        if not isinstance(dir, str | os.PathLike):
            raise TypeError("Log directory must be a str or Path.")

        dir = Path(dir)
        name = self.run_prefix + "_" + self._timestamp
        if self.run_suffix:
            name += f"_{self.run_suffix}"

        full_path = dir / name
        # Non-main DDP ranks never materialize a run directory (see __init__). getattr
        # guards the first setter call, which runs before _is_writer_rank is assigned.
        if getattr(self, "_is_writer_rank", True):
            full_path.mkdir(parents=True, exist_ok=True)

        self._log_dir = full_path

    @property
    def run_prefix(self) -> str:
        return self._run_prefix

    @run_prefix.setter
    def run_prefix(self, prefix: str) -> None:
        if not isinstance(prefix, str):
            raise TypeError("Prefix must be a string")

        self._run_prefix = prefix

    @property
    def run_suffix(self) -> str:
        return self._run_suffix

    @run_suffix.setter
    def run_suffix(self, suffix: str) -> None:
        if not isinstance(suffix, str):
            raise TypeError("Suffix must be a string")

        self._run_suffix = suffix

    @property
    def log_images_every(self) -> int:
        return self._log_images_every

    @log_images_every.setter
    def log_images_every(self, value: int) -> None:
        self._log_images_every = int(value)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in {"tensorboard", "wandb"}:
            raise ValueError("Logger mode must be 'tensorboard' or 'wandb'.")
        self._mode = value

    # --- Helper Functions ---

    @staticmethod
    def apply_colormap(tensor_2d: Tensor | NDArray, cmap_name: str = "turbo") -> NDArray:
        """Apply colormap to a 2D tensor and return a [3, H, W] NumPy float32 array in [0, 1].

        Parameters
        ----------
        tensor_2d : Tensor or NDArray
            2D tensor to apply colormap to.
        cmap_name : str, optional
            Colormap name, by default "turbo"

        Returns
        -------
        NDArray
            Colored array with shape [3, H, W] and dtype float32 in range [0, 1].
        """
        if isinstance(tensor_2d, Tensor):
            tensor_2d = tensor_2d.detach().cpu().numpy()

        tensor_2d = (tensor_2d - np.min(tensor_2d)) / (np.ptp(tensor_2d) + 1e-8)
        cmap = plt.get_cmap(cmap_name)
        colored = cmap(tensor_2d)[..., :3].transpose(2, 0, 1)  # type: ignore # [3, H, W]
        return colored.astype(np.float32)

    def _init_wandb(self, config: Mapping[str, Any] | None = None) -> None:
        """Initialize a WandB run rooted inside this logger's run directory."""
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb_dir = self.log_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        for env_name, path in {
            "WANDB_DIR": wandb_dir,
            "WANDB_CONFIG_DIR": wandb_dir / "config",
            "WANDB_CACHE_DIR": wandb_dir / "cache",
            "WANDB_DATA_DIR": wandb_dir / "data",
        }.items():
            os.environ[env_name] = str(path)
            path.mkdir(parents=True, exist_ok=True)

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "LoggerBase(mode='wandb') requires the 'wandb' package. "
                "The NERSC quantem environment at "
                "/global/common/software/m5020/cedlim/conda/quantem/bin/python3 "
                "includes wandb 0.28.0."
            ) from exc

        self._wandb = wandb
        self._wandb_run = wandb.init(
            project="quantem-tomography",
            name=self.log_dir.name,
            dir=str(self.log_dir),
            config=dict(config) if config is not None else None,
            reinit=True,
        )
        self._wandb_run.define_metric("*", step_metric="epoch")
        self._wandb_run.define_metric("snapshots/*", step_metric="grad_step")

    def _log_wandb(
        self,
        tag: str,
        value: Any,
        step: int,
        step_domain: str,
        extra_steps: dict[str, int] | None = None,
    ) -> None:
        """Log WandB data with named step metrics instead of the global step."""
        payload = {tag: value}
        if extra_steps is not None:
            payload.update(extra_steps)
        payload[step_domain] = step
        self._wandb_run.log(payload)
