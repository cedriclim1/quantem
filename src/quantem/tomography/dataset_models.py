import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.optimizer_mixin import OptimizerMixin, OptimizerParams, OptimizerParamsType
from quantem.tomography.utils import tv_loss_1d

# --- Constraints ---


class DatasetConstraintParams:
    """
    Namespace class for dataset constraint parameter dataclasses and parsing utilities.

    Contains constraint definitions for different tomography dataset types and a
    factory method for instantiating the appropriate constraint class from a dict.

    Supported constraint types
    --------------------------
    BaseTomographyDatasetConstraints
        Base soft constraints for z-position and lateral shift regularization.
    ThroughFocalDatasetConstraints
        Inherits base constraints; not yet implemented.

    Examples
    --------
    >>> DatasetConstraintParams.parse_dict({"name": "base_tomography_dataset", "tv_zs": 0.1})
    BaseTomographyDatasetConstraints(tv_zs=0.1, tv_shifts=0.0)
    >>> DatasetConstraintParams.parse_dict({"type": "base_tomography_dataset"})
    BaseTomographyDatasetConstraints(tv_zs=0.0, tv_shifts=0.0)
    """

    @dataclass
    class BaseTomographyDatasetConstraints(Constraints):
        """
        Soft constraints for a base tomography dataset.

        Attributes
        ----------
        tv_zs : float
            Total variation regularization weight for Z1 and Z3 Euler angles.
        tv_shifts : float
            Total variation regularization weight for X and Y shifts.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly (none for this class).
        """

        tv_zs: float = 0.0
        tv_shifts: float = 0.0
        _name: str = "base_tomography_dataset"

        soft_constraint_keys = ["tv_zs", "tv_shifts"]
        hard_constraint_keys = []

    @dataclass
    class ThroughFocalDatasetConstraints(BaseTomographyDatasetConstraints):
        """
        Constraints for a through-focal tomography dataset.

        Inherits all constraints from ``BaseTomographyDatasetConstraints``.
        Currently not implemented — instantiation will raise ``NotImplementedError``.
        """

        pass

    @classmethod
    def parse_dict(
        cls, d: dict
    ) -> "DatasetConstraintParams.BaseTomographyDatasetConstraints | DatasetConstraintParams.ThroughFocalDatasetConstraints":
        """
        Instantiate a dataset constraint dataclass from a configuration dictionary.

        The dictionary must contain a ``'name'`` or ``'type'`` key identifying
        which constraint class to construct. All remaining keys are forwarded as
        keyword arguments to the selected dataclass.

        Parameters
        ----------
        d : dict
            Configuration dictionary. Must include ``'name'`` or ``'type'``
            with one of the following values (case-insensitive):

            - ``'base_tomography_dataset'`` → :class:`BaseTomographyDatasetConstraints`
            - ``'through_focal_dataset'`` → :class:`ThroughFocalDatasetConstraints`
              *(not yet implemented)*

            The value may also be a class ``type`` object, in which case its
            ``__name__`` is used after lower-casing.

        Returns
        -------
        BaseTomographyDatasetConstraints or ThroughFocalDatasetConstraints
            An instance of the appropriate constraint dataclass.

        Raises
        ------
        ValueError
            If neither ``'name'`` nor ``'type'`` is present, if the value is not
            a string or type, or if the name does not match any known dataset
            constraint type.
        NotImplementedError
            If ``'through_focal_dataset'`` is requested, as it is not yet implemented.
        """
        d = dict(d)
        name = d.pop("name", None)
        type_ = d.pop("type", None)
        name = name or type_
        if name is None:
            raise ValueError("Must provide either 'name' or 'type' key")
        if isinstance(name, type):
            name = name.__name__.lower()
        elif isinstance(name, str):
            name = name.lower()
        else:
            raise ValueError(f"Unknown dataset constraint type: {name}")
        if name == "base_tomography_dataset":
            return DatasetConstraintParams.BaseTomographyDatasetConstraints(**d)
        elif name == "through_focal_dataset":
            raise NotImplementedError("Through focal dataset constraints are not implemented yet.")
        else:
            raise ValueError(f"Unknown dataset constraint type: {name.lower()}")


DatasetConstraintsType = (
    DatasetConstraintParams.BaseTomographyDatasetConstraints
    | DatasetConstraintParams.ThroughFocalDatasetConstraints
)


@dataclass
class DatasetValue:
    """
    Class for storing the forward call for both PixDataset and INRDataset.
    """

    target: torch.Tensor
    tilt_angle: int | float
    pixel_loc: tuple[int, int] | None = None  # Only for INRDataset
    projection_idx: int | None = None  # Only for INRDataset
    pose: tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter] | None = (
        None  # If there is pose optimization.  # Pose is tuple (shifts, z1, z3)
    )


@dataclass(frozen=True)
class PixelHoldoutSplit:
    """Flat pixel indices for train and held-out validation rays."""

    train_indices: torch.Tensor
    val_indices: torch.Tensor
    val_fg_indices: torch.Tensor
    val_bg_indices: torch.Tensor


def _allocate_pixel_holdout_counts(group_sizes: list[int], n_holdout: int) -> list[int]:
    """Allocate an exact holdout count across groups by largest remainder."""
    if n_holdout < 0:
        raise ValueError("n_holdout must be >= 0.")
    total = sum(group_sizes)
    if n_holdout > total:
        raise ValueError("n_holdout cannot exceed the total number of pixels.")
    if total == 0 or n_holdout == 0:
        return [0 for _ in group_sizes]

    quotas = [n_holdout * size / total for size in group_sizes]
    counts = [min(size, int(quota)) for size, quota in zip(group_sizes, quotas)]
    remaining = n_holdout - sum(counts)
    order = sorted(
        range(len(group_sizes)),
        key=lambda i: (quotas[i] - int(quotas[i]), group_sizes[i]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for i in order:
            if counts[i] < group_sizes[i]:
                counts[i] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate holdout counts.")
    return counts


def _sample_intensity_strata(
    *,
    indices: torch.Tensor,
    intensities: torch.Tensor,
    n_holdout: int,
    seed: int,
    num_strata: int,
) -> torch.Tensor:
    """Sample held-out indices from intensity-quantile strata."""
    if n_holdout == 0 or indices.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    if n_holdout > indices.numel():
        raise ValueError("n_holdout cannot exceed the number of candidate pixels.")
    n_strata = max(1, min(int(num_strata), int(indices.numel())))
    order = torch.argsort(intensities[indices], stable=True)
    sorted_indices = indices[order]
    strata = [s for s in torch.tensor_split(sorted_indices, n_strata) if s.numel() > 0]
    counts = _allocate_pixel_holdout_counts([int(s.numel()) for s in strata], n_holdout)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for stratum, count in zip(strata, counts):
        if count == 0:
            continue
        perm = torch.randperm(stratum.numel(), generator=generator)
        selected.append(stratum[perm[:count]])

    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(selected).to(dtype=torch.long)


def build_pixel_holdout_split(
    tilt_stack: Dataset3d | NDArray | torch.Tensor,
    holdout_fraction: float,
    holdout_seed: int = 0,
    *,
    num_strata: int = 8,
    foreground_threshold: float = 0.0,
) -> PixelHoldoutSplit:
    """Build a seeded foreground-aware train/holdout split over flat pixel indices.

    The split is rank-independent: all work happens on detached CPU tensors with a
    local generator seeded only by ``holdout_seed``. Foreground/background groups are
    split proportionally, then each group is sampled from intensity-quantile strata.
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must satisfy 0 <= holdout_fraction < 1.")
    stack = torch.as_tensor(tilt_stack).detach().cpu()
    intensities = stack.to(torch.float32).abs().flatten()
    n_pixels = int(intensities.numel())
    all_indices = torch.arange(n_pixels, dtype=torch.long)
    n_holdout = int(n_pixels * float(holdout_fraction))

    if n_holdout == 0:
        empty = torch.empty(0, dtype=torch.long)
        return PixelHoldoutSplit(
            train_indices=all_indices,
            val_indices=empty,
            val_fg_indices=empty,
            val_bg_indices=empty,
        )

    fg_mask = intensities > float(foreground_threshold)
    fg_indices = all_indices[fg_mask]
    bg_indices = all_indices[~fg_mask]
    bg_count, fg_count = _allocate_pixel_holdout_counts(
        [int(bg_indices.numel()), int(fg_indices.numel())], n_holdout
    )

    if (
        n_holdout >= 2
        and bg_indices.numel() > 0
        and fg_indices.numel() > 0
        and (bg_count == 0 or fg_count == 0)
    ):
        if bg_count == 0 and fg_count > 1:
            bg_count, fg_count = 1, fg_count - 1
        elif fg_count == 0 and bg_count > 1:
            bg_count, fg_count = bg_count - 1, 1

    bg_val = _sample_intensity_strata(
        indices=bg_indices,
        intensities=intensities,
        n_holdout=bg_count,
        seed=int(holdout_seed) * 2 + 1,
        num_strata=num_strata,
    )
    fg_val = _sample_intensity_strata(
        indices=fg_indices,
        intensities=intensities,
        n_holdout=fg_count,
        seed=int(holdout_seed) * 2 + 2,
        num_strata=num_strata,
    )
    val_indices = torch.sort(torch.cat([bg_val, fg_val])).values
    train_mask = torch.ones(n_pixels, dtype=torch.bool)
    train_mask[val_indices] = False
    train_indices = all_indices[train_mask]

    return PixelHoldoutSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        val_fg_indices=torch.sort(fg_val).values,
        val_bg_indices=torch.sort(bg_val).values,
    )


class TomographyDatasetBase(AutoSerialize, OptimizerMixin, nn.Module):
    """
    Base tomography dataset class for all tomography datasets to inherit from.
    """

    _token = object()

    DEFAULT_LRS = {
        "pose_lr": 5e-2,
    }

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        _token: object | None = None,
    ):
        AutoSerialize.__init__(self)
        OptimizerMixin.__init__(self)
        nn.Module.__init__(self)
        if _token is not self._token:
            raise RuntimeError("Use TomographyPixDataset.from_* to instantiate this class.")

        if tilt_stack.shape[0] != len(tilt_angles):
            raise ValueError(
                "The number of tilt projections should be in the first dimension of the dataset."
            )

        if type(tilt_stack) is not torch.Tensor:
            tilt_stack = torch.from_numpy(tilt_stack)
        if type(tilt_angles) is not torch.Tensor:
            tilt_angles = torch.from_numpy(tilt_angles)
        max_val = torch.quantile(tilt_stack, 0.95)
        # A sparse stack (>95% zeros) has a zero 95th quantile; dividing by it
        # would turn the targets into inf/NaN and poison the first backward.
        if max_val <= 0:
            max_val = tilt_stack.abs().max()
        if max_val <= 0:
            raise ValueError("tilt_stack is all zeros; cannot normalize.")

        # Tilt stack normalization
        tilt_stack = tilt_stack / max_val

        self.tilt_stack = tilt_stack
        self.tilt_angles = tilt_angles
        self.learn_shift = learn_shift
        self.learn_tilt_axis = learn_tilt_axis

        # The reference tilt angle is the one with the smallest absolute tilt angle.
        # I.e, the pose will not be optimized for the reference tilt angle.
        self._reference_tilt_angle_idx = torch.argmin(torch.abs(self.tilt_angles))
        # TODO: Implement AuxParams from old tomography_dataset.py here.

        # TODO: The parameters won't be initialized unless .to(device) is called.
        self._z1_angles = torch.zeros(self.learnable_tilts)
        self._z3_angles = torch.zeros(self.learnable_tilts)
        self._shifts = torch.zeros(self.learnable_tilts, 2)

        # Fixed zeros for reference tilt
        self._z1_ref = torch.zeros(1)
        self._z3_ref = torch.zeros(1)
        self._shifts_ref = torch.zeros(1, 2)

    # --- Class methods ---
    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            _token=cls._token,
        )

    # --- Optimization Parameters ---

    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        """Single param group keyed by DEFAULT_OPTIMIZER_KEY.

        Hyperparameters are baked by ``set_optimizer``, not here — return only the tensors,
        matching the ``dict[str, list[tensor]]`` contract the object models use.
        """
        return {self.DEFAULT_OPTIMIZER_KEY: list(self.parameters())}

    def _materialize_pose_parameters(self, device: str | torch.device):
        """Create the learnable pose parameters, or move the existing ones.

        Once the parameters exist, their *current* (possibly trained) values are
        moved; the initial-value buffers are only used on first materialization.
        Rebuilding from the buffers on every call silently reset learned poses
        whenever the dataset changed device (e.g. ``from_file(...).to(device)``).
        """
        z1 = self._z1_params.data if hasattr(self, "_z1_params") else self._z1_angles
        z3 = self._z3_params.data if hasattr(self, "_z3_params") else self._z3_angles
        shifts = self._shifts_params.data if hasattr(self, "_shifts_params") else self._shifts
        self._z1_params = nn.Parameter(z1.detach().to(device))
        self._z3_params = nn.Parameter(z3.detach().to(device))
        self._shifts_params = nn.Parameter(shifts.detach().to(device))

    # --- Forward pass ---
    @abstractmethod
    def forward(
        self,
        dummy_input: Any = None,  # Note all nn.Modules require some input.
    ):
        """
        Forward pass should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    # --- Properties ---
    @property
    def tilt_stack(self) -> torch.Tensor:
        return self._tilt_stack

    @tilt_stack.setter
    def tilt_stack(self, tilt_stack: torch.Tensor):
        if type(tilt_stack) is not torch.Tensor:
            print("Converting tilt stack to torch.Tensor")
            tilt_stack = torch.from_numpy(tilt_stack)

        self._tilt_stack = tilt_stack

    @property
    def tilt_angles(self) -> torch.Tensor:
        return self._tilt_angles

    @tilt_angles.setter
    def tilt_angles(self, tilt_angles: torch.Tensor):
        if type(tilt_angles) is not torch.Tensor:
            print("Converting tilt angles to torch.Tensor")
            tilt_angles = torch.from_numpy(tilt_angles)

        self._tilt_angles = tilt_angles

    @property
    def learn_shift(self) -> bool:
        return self._learn_shift

    @learn_shift.setter
    def learn_shift(self, learn_shift: bool):
        self._learn_shift = learn_shift

    @property
    def learn_tilt_axis(self) -> bool:
        return self._learn_tilt_axis

    @learn_tilt_axis.setter
    def learn_tilt_axis(self, learn_tilt_axis: bool):
        self._learn_tilt_axis = learn_tilt_axis

    @property
    def reference_tilt_idx(self) -> int:
        return int(self._reference_tilt_angle_idx)

    @reference_tilt_idx.setter
    def reference_tilt_idx(self, reference_tilt_idx: int):
        self._reference_tilt_angle_idx = reference_tilt_idx

    @property
    def learnable_tilts(self) -> int:
        # Derived from the tilt series (all tilts minus the fixed reference); there is
        # deliberately no setter -- the old one wrote a private attribute this getter
        # never read, so assignments appeared to succeed while doing nothing.
        return self.tilt_angles.shape[0] - 1

    @property
    def z1_params(self) -> torch.nn.Parameter:
        return self._z1_params

    @z1_params.setter
    def z1_params(self, z1_angles: torch.Tensor, device: str):
        self._z1_params = nn.Parameter(z1_angles.to(device))

    @property
    def z3_params(self) -> torch.nn.Parameter:
        return self._z3_params

    @z3_params.setter
    def z3_params(self, z3_angles: torch.Tensor, device: str):
        self._z3_params = nn.Parameter(z3_angles.to(device))

    @property
    def shifts_params(self) -> torch.nn.Parameter:
        return self._shifts_params

    @shifts_params.setter
    def shifts_params(self, shifts: torch.Tensor, device: str):
        self._shifts_params = nn.Parameter(shifts.to(device))

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device

    # --- Helper Functions ---
    @abstractmethod
    def to(self, device: torch.device | str):  # type: ignore
        """
        Moves the dataset to the device, and also insantiates the aux params to the device.
        """

        raise NotImplementedError("This method should be implemented in subclasses.")


class TomographyDatasetConstraints(BaseConstraints, TomographyDatasetBase):
    DEFAULT_CONSTRAINTS = DatasetConstraintParams.BaseTomographyDatasetConstraints()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints: DatasetConstraintParams.BaseTomographyDatasetConstraints = (
            self.DEFAULT_CONSTRAINTS.copy()
        )

    def apply_soft_constraints(self) -> torch.Tensor:
        soft_loss = torch.zeros((), device=self.z1_params.device)
        if self.constraints.tv_zs > 0:
            tv_loss_zs = tv_loss_1d(self.z1_params)
            tv_loss_zs += tv_loss_1d(self.z3_params)
            tv_loss_zs = self.constraints.tv_zs * tv_loss_zs
            soft_loss += tv_loss_zs

        if self.constraints.tv_shifts > 0:
            # Shift params is of shape (N, 2)
            tv_loss_shifts = tv_loss_1d(self.shifts_params[:, 0])
            tv_loss_shifts += tv_loss_1d(self.shifts_params[:, 1])
            tv_loss_shifts = self.constraints.tv_shifts * tv_loss_shifts
            soft_loss += tv_loss_shifts
        return soft_loss

    def apply_hard_constraints(self) -> torch.Tensor:
        """
        No hard constraints have been implemented yet.
        """
        return torch.tensor(0.0)


class TomographyPixDataset(TomographyDatasetConstraints):
    """
    Dataset class for pixel-based tomography, i.e AD, SIRT, WBP, etc...

    These algorithms only require the tilt image in the forward call.
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        _token: object | None = None,
    ):
        super().__init__(
            tilt_stack=tilt_stack,
            tilt_angles=-tilt_angles,  # TODO: Flip the tilt angles to be negative to match the convention of INR.
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            _token=_token,
        )

    def forward(  # type:ignore
        self,
        proj_idx: int,
    ) -> DatasetValue:
        """
        Forward pass for pixel-based tomography.
        Returns the full tilt image for the given projection index, and the tilt angle.
        """

        return DatasetValue(
            target=self.tilt_stack[proj_idx],
            tilt_angle=self.tilt_angles[proj_idx].item(),
            pixel_loc=None,
        )

    def to(self, device: str | torch.device):
        """
        Moves the tilt stack and tilt_angles to the device, along with other nn.Parameters to the device.
        """
        self.tilt_stack = self.tilt_stack.to(device)
        self.tilt_angles = self.tilt_angles.to(device)

        self._materialize_pose_parameters(device)

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device


class DeviceBatchSampler:
    """Epoch iterator that builds INR training batches directly on a device.

    Replaces the per-pixel DataLoader path: the tilt stack and angles are
    made resident on ``device`` once, so producing a batch is index
    arithmetic plus two tensor lookups instead of ``batch_size`` Python
    ``__getitem__`` calls, a collate, and a host-to-device copy per step.
    On a GPU this removes the CPU dataloader bottleneck entirely.

    Yields the same batch dicts as ``TomographyINRDataset.__getitem__``
    under a DataLoader collate (``projection_idx``, ``pixel_i``,
    ``pixel_j``, ``phi``, ``target_value``), with the train loader's
    ``drop_last=True`` semantics.

    Distributed runs: pass ``rank``/``world_size`` and every rank derives
    the *same* epoch permutation from ``seed + epoch`` (CPU generator, so
    it is identical across ranks and reproducible), then takes an
    equal-size contiguous shard — equal so per-rank batch counts match and
    DDP gradient sync cannot hang on a ragged tail. The training loop's
    ``sampler.set_epoch(epoch)`` drives reshuffling, exactly like
    ``DistributedSampler``; without ``set_epoch`` the epoch advances
    automatically on each ``__iter__``.
    """

    def __init__(
        self,
        dset: "TomographyINRDataset",
        batch_size: int,
        device: torch.device | str,
        indices: torch.Tensor | None = None,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0
        self._stack = dset.tilt_stack.to(self.device)
        self._angles = dset.tilt_angles.to(self.device)
        self._angles_per_row = (
            dset.tilt_angles_per_row.to(self.device)
            if dset.tilt_angles_per_row is not None
            else None
        )
        self._angles_per_col = (
            dset.tilt_angles_per_col.to(self.device)
            if dset.tilt_angles_per_col is not None
            else None
        )
        self._s1 = dset.tilt_stack.shape[1]
        self._s2 = dset.tilt_stack.shape[2]
        if indices is None:
            indices = torch.arange(len(dset), dtype=torch.int64)
        self._indices = indices.to(self.device)
        self._per_rank = len(self._indices) // world_size

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed this epoch's shared permutation."""
        self._epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self._per_rank // self.batch_size
        return (self._per_rank + self.batch_size - 1) // self.batch_size

    def _epoch_shard(self) -> torch.Tensor:
        idx = self._indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(len(idx), generator=g).to(self.device)
            idx = idx[perm]
            self._epoch += 1  # auto-advance; set_epoch overrides per epoch
        if self.world_size > 1:
            idx = idx[self.rank * self._per_rank : (self.rank + 1) * self._per_rank]
        return idx

    def __iter__(self):
        idx = self._epoch_shard()
        per_proj = self._s1 * self._s2
        for k in range(len(self)):
            sel = idx[k * self.batch_size : min((k + 1) * self.batch_size, len(idx))]
            proj = sel // per_proj
            rem = sel - proj * per_proj
            pixel_i = rem // self._s2
            pixel_j = rem - pixel_i * self._s2
            if self._angles_per_row is not None:
                phi = self._angles_per_row[proj, pixel_i]
            elif self._angles_per_col is not None:
                phi = self._angles_per_col[proj, pixel_j]
            else:
                phi = self._angles[proj]
            yield {
                "projection_idx": proj,
                "pixel_i": pixel_i,
                "pixel_j": pixel_j,
                "phi": phi,
                "target_value": self._stack[proj, pixel_i, pixel_j],
            }


class TomographyINRDataset(TomographyDatasetConstraints, Dataset):
    """
    Dataset class for INR-based tomography.

    The two main methods here are that the `forward` call will return the relative pose parameters,
    while `__getitem__` will actually return the pixel values of the tilt stack.

    TODO: I think TomographyINRDataset shouldn't handle the train/val split and will be handled later? Yea this is handled in setup_dataloader in DDP
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        tilt_angles_per_row: NDArray | torch.Tensor | None = None,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        ray_sampling: str = "box_fixed_ds",
        ray_ds: float | None = None,
        seed: int = 42,
        tilt_angles_per_col: NDArray | torch.Tensor | None = None,
        _token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token=_token)

        if tilt_angles_per_row is not None and tilt_angles_per_col is not None:
            raise ValueError("tilt_angles_per_row and tilt_angles_per_col are mutually exclusive")
        if tilt_angles_per_row is not None:
            if type(tilt_angles_per_row) is not torch.Tensor:
                tilt_angles_per_row = torch.from_numpy(tilt_angles_per_row)
            if tilt_angles_per_row.ndim != 2 or tuple(tilt_angles_per_row.shape) != tuple(
                tilt_stack.shape[:2]
            ):
                raise ValueError(
                    "tilt_angles_per_row must have shape "
                    f"{tuple(tilt_stack.shape[:2])}, got {tuple(tilt_angles_per_row.shape)}."
                )
        self.tilt_angles_per_row = tilt_angles_per_row
        if tilt_angles_per_col is not None:
            if type(tilt_angles_per_col) is not torch.Tensor:
                tilt_angles_per_col = torch.from_numpy(tilt_angles_per_col)
            expected_shape = (tilt_stack.shape[0], tilt_stack.shape[2])
            if tilt_angles_per_col.ndim != 2 or tuple(tilt_angles_per_col.shape) != tuple(
                expected_shape
            ):
                raise ValueError(
                    "tilt_angles_per_col must have shape "
                    f"{tuple(expected_shape)}, got {tuple(tilt_angles_per_col.shape)}."
                )
        self.tilt_angles_per_col = tilt_angles_per_col

        # --- Ray-sampling scheme ---
        # "legacy"       : detector-frame z in [-1, 1] swept with a FIXED count of points,
        #                  then rotated (create_batch_rays / transform_batch_rays). The
        #                  segment is rotated with the object, so long diagonal chords are
        #                  clipped and other rays waste samples outside the [-1,1]^3 cube.
        # "box_fixed_ds" : per-ray ray-box intersection with the [-1,1]^3 cube, then a
        #                  CONSTANT physical step `ds` along the true chord (variable count
        #                  per ray). Gives identical physical sampling spacing on every ray
        #                  regardless of tilt -- the consistent discretization of the line
        #                  integral. Ragged, so integrate_rays uses a scatter-add.
        self.ray_sampling: str = ray_sampling
        # Physical step between samples for "box_fixed_ds". None => derive per call from
        # num_samples_per_ray as 2/(num_samples_per_ray - 1) so the existing samples_per_ray
        # knob stays meaningful (a centered ray reproduces the legacy sample count).
        self.ray_ds: float | None = ray_ds
        # Per-batch ragged metadata stashed by get_coords for integrate_rays to consume.
        self._ray_meta: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        tilt_angles_per_row: NDArray | torch.Tensor | None = None,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        ray_sampling: str = "box_fixed_ds",
        ray_ds: float | None = None,
        tilt_angles_per_col: NDArray | torch.Tensor | None = None,
    ):

        if ray_sampling == "box_fixed_ds":
            if ray_ds is None:
                ray_ds = 2.0 / max(tilt_stack.shape)
            else:
                ray_ds = float(ray_ds)

        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            tilt_angles_per_row=tilt_angles_per_row,
            tilt_angles_per_col=tilt_angles_per_col,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            ray_sampling=ray_sampling,
            ray_ds=ray_ds,
            _token=cls._token,
        )

    # --- Forward Pass w/ Params Method for OptimizerMixin ---
    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        """Return independently tunable shift and tilt-axis parameter groups."""
        groups = {}
        if self.learn_shift:
            groups["pose_shift"] = [self._shifts_params]
        if self.learn_tilt_axis:
            groups["pose_tilt_axis"] = [self._z1_params, self._z3_params]
        return groups

    def _normalize_optimizer_params(
        self, params: OptimizerParamsType | dict[str, Any]
    ) -> dict[str, OptimizerParamsType]:
        """Expand a legacy shared pose optimizer over the active pose groups."""
        normalized = super()._normalize_optimizer_params(params)
        if set(normalized) == {self.DEFAULT_OPTIMIZER_KEY}:
            spec = normalized[self.DEFAULT_OPTIMIZER_KEY]
            if not isinstance(spec, OptimizerParams.NoneOptimizer):
                normalized = {
                    key: spec
                    for key in ("pose_shift", "pose_tilt_axis")
                    if (key == "pose_shift" and self.learn_shift)
                    or (key == "pose_tilt_axis" and self.learn_tilt_axis)
                }
        return normalized

    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based tomography. In the forward pass, the only parameters that
        are passed will be the shifts, z1 and z3 Euler angles.
        """

        first_half_shifts = self.shifts_params[: self.reference_tilt_idx]
        second_half_shifts = self.shifts_params[self.reference_tilt_idx :]
        shifts = torch.cat([first_half_shifts, self._shifts_ref, second_half_shifts], dim=0)

        first_half_z1 = self.z1_params[: self.reference_tilt_idx]
        second_half_z1 = self.z1_params[self.reference_tilt_idx :]
        z1 = torch.cat([first_half_z1, self._z1_ref, second_half_z1], dim=0)

        first_half_z3 = self.z3_params[: self.reference_tilt_idx]
        second_half_z3 = self.z3_params[self.reference_tilt_idx :]
        z3 = torch.cat([first_half_z3, self._z3_ref, second_half_z3], dim=0)

        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3)
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3)

    def get_coords(
        self, batch: dict[str, torch.Tensor], N: int, num_samples_per_ray: int
    ) -> torch.Tensor:
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        # target_values = batch["target_value"].to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)

        shifts, z1_params, z3_params = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)

        if getattr(self, "ray_sampling", "legacy") == "box_fixed_ds":
            return self._get_coords_box_fixed_ds(
                pixel_i=pixel_i,
                pixel_j=pixel_j,
                phis=phis,
                batch_z1=batch_z1,
                batch_z3=batch_z3,
                batch_shifts=batch_shifts,
                N=N,
                num_samples_per_ray=num_samples_per_ray,
            )

        with torch.no_grad():
            batch_ray_coords = self.create_batch_rays(pixel_i, pixel_j, N, num_samples_per_ray)

        transformed_rays = self.transform_batch_rays(
            batch_ray_coords,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def create_batch_rays(
        pixel_i: torch.Tensor, pixel_j: torch.Tensor, N: int, num_samples_per_ray: int
    ) -> torch.Tensor:
        batch_size = len(pixel_i)
        x_coords = (pixel_j / (N - 1)) * 2 - 1
        y_coords = (pixel_i / (N - 1)) * 2 - 1
        z_coords = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

        rays = torch.zeros(batch_size, num_samples_per_ray, 3, device=pixel_i.device)

        rays[:, :, 0] = x_coords.unsqueeze(1)
        rays[:, :, 1] = y_coords.unsqueeze(1)
        rays[:, :, 2] = z_coords.unsqueeze(0)

        return rays

    @staticmethod
    def transform_batch_rays(
        rays: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> torch.Tensor:
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        shifted = torch.stack(
            [rays[:, :, 0] - shift_x_norm, rays[:, :, 1] - shift_y_norm, rays[:, :, 2]],
            dim=2,
        )

        rot = TomographyINRDataset._compose_euler_rotation(z1, x, z3)  # (B, 3, 3)

        return shifted @ rot.transpose(1, 2)

    @staticmethod
    def _compose_euler_rotation(
        z1: torch.Tensor, x: torch.Tensor, z3: torch.Tensor
    ) -> torch.Tensor:
        """Compose the three Euler rotations Rz(-z1) @ Rx(x) @ Rz(-z3) into a single
        (B, 3, 3) matrix. A point row-vector ``v`` is mapped to the object frame by
        ``v @ rot.transpose`` (equivalently ``rot @ v``)."""
        a = torch.deg2rad(-z3).view(-1)
        b = torch.deg2rad(x).view(-1)
        g = torch.deg2rad(-z1).view(-1)
        zero = torch.zeros_like(a)
        one = torch.ones_like(a)

        cos_a, sin_a = torch.cos(a), torch.sin(a)
        cos_b, sin_b = torch.cos(b), torch.sin(b)
        cos_g, sin_g = torch.cos(g), torch.sin(g)

        rot_a = torch.stack(
            [cos_a, -sin_a, zero, sin_a, cos_a, zero, zero, zero, one], dim=-1
        ).view(-1, 3, 3)
        rot_b = torch.stack(
            [one, zero, zero, zero, cos_b, -sin_b, zero, sin_b, cos_b], dim=-1
        ).view(-1, 3, 3)
        rot_g = torch.stack(
            [cos_g, -sin_g, zero, sin_g, cos_g, zero, zero, zero, one], dim=-1
        ).view(-1, 3, 3)

        return rot_g @ rot_b @ rot_a  # (B, 3, 3)

    # ------------------------------------------------------------------
    # Box-intersection, fixed-distance ("box_fixed_ds") ray sampling.
    # Ported from ray_sampling/geometry.py (ParallelBeamRayProjector), reusing the
    # dataset's own Euler pose so gradients keep flowing to z1/z3/shifts.
    # ------------------------------------------------------------------
    def _resolve_ray_ds(self, num_samples_per_ray: int) -> float:
        """Physical step `ds` between samples along every ray (object-space units,
        where the cube spans [-1, 1])."""
        if self.ray_ds is not None:
            ds = float(self.ray_ds)
            if ds <= 0.0:
                raise ValueError(f"ray_ds must be > 0, got {ds}")
            return ds
        # Derive from the samples_per_ray knob so it stays meaningful: an untilted ray
        # (chord length 2) then gets exactly num_samples_per_ray points, matching legacy.
        return 2.0 / (num_samples_per_ray - 1)

    def _build_ray_origins_directions(
        self,
        *,
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ray origin/direction in the object frame for each detector pixel.

        Rotates the two legacy endpoints (x, y, +/-1) by the same Euler pose used in
        transform_batch_rays, so ``point(t) = origin + t * direction`` reproduces the
        legacy z in [-1, 1] segment at t in [-1, 1] but lets t run past it to capture the
        full chord through the cube. ``direction`` is unit-norm (rotation of (0,0,1)), so
        t is true arc length and ``ds`` is a physical distance.
        """
        x_coords = (pixel_j / (N - 1)) * 2 - 1
        y_coords = (pixel_i / (N - 1)) * 2 - 1
        shift_x_norm = (shifts[:, 0] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1] * sampling_rate * 2) / (N - 1)
        x_base = x_coords - shift_x_norm
        y_base = y_coords - shift_y_norm

        rot = self._compose_euler_rotation(z1, x, z3)  # (B, 3, 3)
        ones = torch.ones_like(x_base)
        p_plus_local = torch.stack((x_base, y_base, ones), dim=-1)
        p_minus_local = torch.stack((x_base, y_base, -ones), dim=-1)
        p_plus = torch.einsum("bij,bj->bi", rot, p_plus_local)
        p_minus = torch.einsum("bij,bj->bi", rot, p_minus_local)

        origins = 0.5 * (p_plus + p_minus)
        directions = 0.5 * (p_plus - p_minus)
        return origins, directions

    @staticmethod
    def _compute_ray_box_intersections(
        origins: torch.Tensor,
        directions: torch.Tensor,
        eps: float = 1.0e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slab method: entry/exit t into the [-1, 1]^3 cube and a validity mask."""
        parallel = directions.abs() < eps
        safe_dirs = torch.where(parallel, torch.ones_like(directions), directions)

        t0 = (-1.0 - origins) / safe_dirs
        t1 = (1.0 - origins) / safe_dirs
        t_min_dim = torch.minimum(t0, t1)
        t_max_dim = torch.maximum(t0, t1)

        neg_inf = torch.full_like(t_min_dim, -float("inf"))
        pos_inf = torch.full_like(t_max_dim, float("inf"))
        in_slab = (origins >= -1.0) & (origins <= 1.0)
        valid_parallel = (~parallel) | in_slab

        t_min_dim = torch.where(parallel, neg_inf, t_min_dim)
        t_max_dim = torch.where(parallel, pos_inf, t_max_dim)

        t_enter = t_min_dim.max(dim=1).values
        t_exit = t_max_dim.min(dim=1).values
        valid = valid_parallel.all(dim=1) & (t_exit > t_enter)
        return t_enter, t_exit, valid

    def _sample_ray_segment_coords_fixed_ds(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        t_enter: torch.Tensor,
        t_exit: torch.Tensor,
        ds: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Place points every ``ds`` along each ray's in-cube chord.

        Variable count per ray (padded to the batch max), so a ``sample_mask`` marks the
        real samples. Returns (coords, sample_mask, n_samples, lengths). The sample COUNT
        is decided from detached lengths (ceil is non-differentiable anyway); the sample
        POSITIONS stay differentiable in t_enter/lengths, so pose gradients flow.
        """
        lengths = t_exit - t_enter
        n_samples = torch.ceil(lengths.detach() / float(ds)).to(torch.long) + 1
        n_samples = torch.clamp(n_samples, min=2)

        max_samples = int(n_samples.max().item())
        sample_idx = torch.arange(max_samples, device=origins.device).unsqueeze(0)
        sample_mask = sample_idx < n_samples.unsqueeze(1)

        denom = (n_samples - 1).to(origins.dtype).unsqueeze(1)
        u = sample_idx.to(origins.dtype) / denom
        t_vals = t_enter.unsqueeze(1) + lengths.unsqueeze(1) * u
        coords = origins.unsqueeze(1) + t_vals.unsqueeze(2) * directions.unsqueeze(1)
        return coords, sample_mask, n_samples, lengths

    def _get_coords_box_fixed_ds(
        self,
        *,
        pixel_i: torch.Tensor,
        pixel_j: torch.Tensor,
        phis: torch.Tensor,
        batch_z1: torch.Tensor,
        batch_z3: torch.Tensor,
        batch_shifts: torch.Tensor,
        N: int,
        num_samples_per_ray: int,
    ) -> torch.Tensor:
        origins, directions = self._build_ray_origins_directions(
            pixel_i=pixel_i,
            pixel_j=pixel_j,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        t_enter, t_exit, valid = self._compute_ray_box_intersections(origins, directions)

        if getattr(self, "_cuda_graph_static_sampling", False):
            return self._get_coords_box_fixed_ds_cuda_graph(
                origins=origins,
                directions=directions,
                t_enter=t_enter,
                t_exit=t_exit,
                valid=valid,
                num_samples_per_ray=num_samples_per_ray,
            )

        if not bool(valid.any()):
            self._ray_meta = {
                "valid": valid,
                "local_ray_ids": torch.zeros(0, dtype=torch.long, device=origins.device),
                "step_sizes": origins.new_zeros(0),
                "num_valid": 0,
            }
            return origins.new_zeros((0, 3))

        origins_v = origins[valid]
        directions_v = directions[valid]
        t_enter_v = t_enter[valid]
        t_exit_v = t_exit[valid]

        ds = self._resolve_ray_ds(num_samples_per_ray)
        (
            coords_v,
            sample_mask_v,
            n_samples_v,
            lengths_v,
        ) = self._sample_ray_segment_coords_fixed_ds(
            origins_v, directions_v, t_enter_v, t_exit_v, ds
        )

        # nonzero() walks row-major, so local_ray_ids lines up with coords_v[sample_mask_v].
        local_ray_ids = sample_mask_v.nonzero(as_tuple=False)[:, 0]
        step_sizes_v = lengths_v / (n_samples_v.to(lengths_v.dtype) - 1.0)

        self._ray_meta = {
            "valid": valid,
            "local_ray_ids": local_ray_ids,
            "step_sizes": step_sizes_v,
            "num_valid": int(origins_v.shape[0]),
        }

        all_coords = coords_v[sample_mask_v]
        return all_coords.to(self.device, dtype=torch.float32)

    def _get_coords_box_fixed_ds_cuda_graph(
        self,
        *,
        origins: torch.Tensor,
        directions: torch.Tensor,
        t_enter: torch.Tensor,
        t_exit: torch.Tensor,
        valid: torch.Tensor,
        num_samples_per_ray: int,
    ) -> torch.Tensor:
        """Build a fixed-capacity ray representation suitable for CUDA Graph replay.

        Eager fixed-ds sampling compacts valid rays and sizes its padded dimension from a
        GPU reduction. Both make allocation shapes depend on batch contents. During graph
        capture, retain every ray and use the cube's maximum possible chord to choose one
        conservative, host-known padded width. Invalid/padded coordinates are placed
        outside the cube, so object-model masks make their densities and gradients zero.
        """
        ds = self._resolve_ray_ds(num_samples_per_ray)
        max_samples = math.ceil((2.0 * math.sqrt(3.0)) / float(ds)) + 1
        lengths = t_exit - t_enter
        n_samples = torch.ceil(lengths.detach() / float(ds)).to(torch.long) + 1
        n_samples = torch.clamp(n_samples, min=2, max=max_samples)

        sample_idx = torch.arange(max_samples, device=origins.device).unsqueeze(0)
        sample_mask = valid.unsqueeze(1) & (sample_idx < n_samples.unsqueeze(1))
        denom = (n_samples - 1).to(origins.dtype).unsqueeze(1)
        u = sample_idx.to(origins.dtype) / denom
        t_vals = t_enter.unsqueeze(1) + lengths.unsqueeze(1) * u
        coords = origins.unsqueeze(1) + t_vals.unsqueeze(2) * directions.unsqueeze(1)
        coords = torch.where(sample_mask.unsqueeze(2), coords, coords.new_full((), 2.0))

        step_sizes = lengths / (n_samples.to(lengths.dtype) - 1.0)
        self._ray_meta = {
            "valid": valid,
            "sample_mask": sample_mask,
            "step_sizes": step_sizes,
            "num_valid_samples": sample_mask.sum(),
        }
        return coords.reshape(-1, 3).to(self.device, dtype=torch.float32)

    def integrate_rays(
        self, rays: torch.Tensor, num_samples_per_ray: int, target_values_len: int
    ) -> torch.Tensor:
        if getattr(self, "ray_sampling", "legacy") == "box_fixed_ds":
            return self._integrate_rays_box_fixed_ds(rays, target_values_len)
        return self._integrate_rays_legacy(rays, num_samples_per_ray, target_values_len)

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def _integrate_rays_legacy(
        rays: torch.Tensor, num_samples_per_ray: int, target_values_len: int
    ) -> torch.Tensor:
        ray_densities = rays.view(
            target_values_len,
            num_samples_per_ray,
        )
        step_size = 2.0 / (num_samples_per_ray - 1)

        predicted_values = ray_densities.sum(dim=1) * step_size

        return predicted_values

    def _integrate_rays_box_fixed_ds(
        self, densities: torch.Tensor, target_values_len: int
    ) -> torch.Tensor:
        """Riemann-sum integrate ragged fixed-ds samples back to one value per ray.

        ``densities`` are the INR outputs for the flattened, mask-selected samples that
        ``get_coords`` produced (same order). Scatter-add them per ray and scale by the
        per-ray physical step, writing into a zero (B,) output at the rays that actually
        intersected the cube.
        """
        meta = self._ray_meta
        if meta is None:
            raise RuntimeError(
                "integrate_rays called in 'box_fixed_ds' mode without ray metadata; "
                "get_coords must run first."
            )
        if "sample_mask" in meta:
            sample_mask = meta["sample_mask"]
            ray_densities = densities.reshape(sample_mask.shape)
            ray_sums = (ray_densities * sample_mask.to(densities.dtype)).sum(dim=1)
            predicted = ray_sums * meta["step_sizes"].to(densities.dtype)
            return torch.where(meta["valid"], predicted, torch.zeros_like(predicted))

        predicted = torch.zeros(target_values_len, device=densities.device, dtype=densities.dtype)
        valid = meta["valid"]
        if not bool(valid.any()):
            self._ray_meta = None
            return predicted

        local_ray_ids = meta["local_ray_ids"]
        step_sizes = meta["step_sizes"].to(densities.dtype)
        num_valid = int(meta["num_valid"])

        ray_sums = torch.zeros(num_valid, device=densities.device, dtype=densities.dtype)
        ray_sums.index_add_(0, local_ray_ids, densities)
        predicted[valid] = ray_sums * step_sizes
        # Consumed; clear so a stale batch can never be silently reused.
        self._ray_meta = None
        return predicted

    def graph_constraint_densities(self, densities: torch.Tensor) -> torch.Tensor:
        """Match eager sparsity normalization for graph-padded fixed-ds densities."""
        meta = self._ray_meta
        if meta is None or "sample_mask" not in meta:
            return densities
        num_valid_samples = meta["num_valid_samples"].clamp_min(1).to(densities.dtype)
        return densities * (densities.numel() / num_valid_samples)

    # --- Torch Dataset Methods ---
    def __getitem__(
        self,
        idx: int,
    ) -> dict:
        """
        Gets the item for INR i.e, the project index, pixel value at (i, j), and the tilt angle.
        """

        actual_idx = idx

        projection_idx = actual_idx // (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])
        remaining = actual_idx % (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])

        pixel_i = remaining // self.tilt_stack.shape[2]
        pixel_j = remaining % self.tilt_stack.shape[2]
        if self.tilt_angles_per_row is not None:
            phi = self.tilt_angles_per_row[projection_idx, pixel_i]
        elif self.tilt_angles_per_col is not None:
            phi = self.tilt_angles_per_col[projection_idx, pixel_j]
        else:
            phi = self.tilt_angles[projection_idx]

        # Plain ints for the index fields: default_collate builds one int64 tensor per
        # batch either way, but wrapping each index in torch.tensor() here allocates
        # three scalar tensors per item on the dataloader hot path.
        return {
            "projection_idx": projection_idx,
            "pixel_i": pixel_i,
            "pixel_j": pixel_j,
            "phi": phi,  # tensor
            "target_value": self.tilt_stack[projection_idx, pixel_i, pixel_j],  # tensor
        }

    def __len__(
        self,
    ):
        """
        Returns the number of pixels in the tilt stack.
        """
        return self.tilt_stack.shape[0] * self.tilt_stack.shape[1] * self.tilt_stack.shape[2]

    def to(self, device: torch.device | str):
        self._materialize_pose_parameters(device)

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()

    # --- Save learned parameters ---

    def save_parameters(self, path: str):
        """
        Saves the learned parameters to a file.
        """
        torch.save(
            {
                "z1": self._z1_params.detach().cpu(),
                "z3": self._z3_params.detach().cpu(),
                "shifts": self._shifts_params.detach().cpu(),
            },
            path,
        )

    def load_parameters(self, path: str):
        """
        Loads the learned parameters from a file.
        """
        data = torch.load(path)
        self._z1_params = nn.Parameter(data["z1"]).to(self.device)
        self._z3_params = nn.Parameter(data["z3"]).to(self.device)
        self._shifts_params = nn.Parameter(data["shifts"]).to(self.device)
        if self.optimizer is not None:
            self.reconnect_optimizer_to_parameters()


class TomographyINRPretrainDataset(Dataset):
    """
    Dataset class for pretraining INR models.
    """

    def __init__(
        self,
        pretrain_target: torch.Tensor,
    ):
        data = pretrain_target.float()

        total_elements = data.numel()
        if total_elements > 1e6:
            sample_size = min(int(1e6), total_elements)
            flat_data = data.flatten()
            indices = torch.randperm(total_elements)[:sample_size]
            sampled_data = flat_data[indices]
            data_quantile = torch.quantile(sampled_data, 0.95)
        else:
            data_quantile = torch.quantile(data, 0.95)

        # Same guard as TomographyDatasetBase: a >95%-zero target has a zero
        # 95th quantile and would normalize to inf/NaN.
        if data_quantile <= 0:
            data_quantile = data.abs().max()
        if data_quantile <= 0:
            raise ValueError("pretrain_target is all zeros; cannot normalize.")

        data = data / data_quantile
        data = torch.permute(data, (0, 3, 2, 1))
        # data = torch.flip(data, dims=(2,))

        self.volume = data.cpu()
        self.N = pretrain_target.shape[1]  # Assumes cubic volume.
        self.total_samples = pretrain_target.shape[1] ** 3

        coords_1d = torch.linspace(-1, 1, self.N)
        x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        self.coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3).cpu()
        self.targets = self.volume.reshape(-1).cpu()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {"coords": self.coords[idx], "target": self.targets[idx]}


DatasetModelType = TomographyINRDataset | TomographyPixDataset
