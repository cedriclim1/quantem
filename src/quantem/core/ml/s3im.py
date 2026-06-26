from math import exp, isqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    center = window_size // 2
    values = [exp(-((x - center) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    kernel = torch.tensor(values, dtype=torch.float32)
    return kernel / kernel.sum()


def _create_window(window_size: int, channel: int, *, device, dtype) -> torch.Tensor:
    window_1d = _gaussian(window_size, 1.5).to(device=device, dtype=dtype).unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    return window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()


def _auto_patch_shape(num_pixels: int) -> tuple[int, int]:
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be >= 1, got {num_pixels}")

    root = isqrt(num_pixels)
    for height in range(root, 0, -1):
        if num_pixels % height == 0:
            return height, num_pixels // height
    return 1, num_pixels


class SSIM(nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 4,
        stride: int = 4,
        value_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.value_range = float(value_range)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self._cached_channel = 0
        self._cached_window: torch.Tensor | None = None

        if self.window_size <= 0:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.value_range <= 0.0:
            raise ValueError(f"value_range must be > 0, got {self.value_range}")

    def _window(self, x: torch.Tensor) -> torch.Tensor:
        channel = int(x.shape[1])
        if (
            self._cached_window is None
            or self._cached_channel != channel
            or self._cached_window.device != x.device
            or self._cached_window.dtype != x.dtype
        ):
            self._cached_window = _create_window(
                self.window_size,
                channel,
                device=x.device,
                dtype=x.dtype,
            )
            self._cached_channel = channel
        return self._cached_window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"Expected matching shapes, got {tuple(x.shape)} and {tuple(y.shape)}")
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

        window = self._window(x)
        channel = int(x.shape[1])
        padding = (self.window_size - 1) // 2

        mu_x = F.conv2d(x, window, padding=padding, groups=channel, stride=self.stride)
        mu_y = F.conv2d(y, window, padding=padding, groups=channel, stride=self.stride)

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(x * x, window, padding=padding, groups=channel, stride=self.stride) - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y * y, window, padding=padding, groups=channel, stride=self.stride) - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, window, padding=padding, groups=channel, stride=self.stride) - mu_xy
        )

        sigma_x_sq = sigma_x_sq.clamp_min(0.0)
        sigma_y_sq = sigma_y_sq.clamp_min(0.0)

        c1 = (self.k1 * self.value_range) ** 2
        c2 = (self.k2 * self.value_range) ** 2

        ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ).clamp_min(1.0e-12)
        ssim_map = ssim_map.clamp(min=-1.0, max=1.0)
        return ssim_map.mean(dim=(1, 2, 3))


class S3IMLoss(nn.Module):
    """Paper-faithful S3IM for scalar or multi-channel tomography ray batches."""

    def __init__(
        self,
        *,
        kernel_size: int = 4,
        stride: int | None = None,
        repeat_time: int = 10,
        patch_height: int | None = None,
        patch_width: int | None = None,
        value_range: float = 1.0,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else self.kernel_size
        self.repeat_time = int(repeat_time)
        self.patch_height = None if patch_height is None else int(patch_height)
        self.patch_width = None if patch_width is None else int(patch_width)
        self.value_range = float(value_range)
        self.ssim = SSIM(
            window_size=self.kernel_size,
            stride=self.stride,
            value_range=self.value_range,
        )

        if self.repeat_time <= 0:
            raise ValueError(f"repeat_time must be >= 1, got {self.repeat_time}")
        if self.patch_height is not None and self.patch_height <= 0:
            raise ValueError(f"patch_height must be >= 1, got {self.patch_height}")
        if self.patch_width is not None and self.patch_width <= 0:
            raise ValueError(f"patch_width must be >= 1, got {self.patch_width}")

    @staticmethod
    def _canonicalize(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 2:
            raise ValueError(
                "Expected batched pixel values shaped [B] or [B, C], "
                f"got {tuple(values.shape)}"
            )
        return values.float()

    def resolve_patch_shape(self, batch_size: int) -> tuple[int, int]:
        if self.patch_height is None and self.patch_width is None:
            return _auto_patch_shape(batch_size)
        if self.patch_height is None or self.patch_width is None:
            raise ValueError("patch_height and patch_width must both be set or both be omitted.")
        return self.patch_height, self.patch_width

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        *,
        return_similarity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        predicted = self._canonicalize(predicted)
        target = self._canonicalize(target)
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted and target must have matching shapes, got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}"
            )

        batch_size, channels = predicted.shape
        patch_height, patch_width = self.resolve_patch_shape(batch_size)
        expected_size = patch_height * patch_width
        if expected_size != batch_size:
            raise ValueError(
                "S3IM expects patch_height * patch_width == batch size, "
                f"got {patch_height} * {patch_width} != {batch_size}"
            )

        index_list = [torch.arange(batch_size, device=predicted.device)]
        for _ in range(1, self.repeat_time):
            index_list.append(torch.randperm(batch_size, device=predicted.device))
        res_index = torch.cat(index_list, dim=0)

        pred_all = predicted[res_index]
        target_all = target[res_index]
        pred_patch = pred_all.transpose(0, 1).reshape(
            1,
            channels,
            patch_height,
            patch_width * self.repeat_time,
        )
        target_patch = target_all.transpose(0, 1).reshape(
            1,
            channels,
            patch_height,
            patch_width * self.repeat_time,
        )

        similarity = self.ssim(pred_patch, target_patch).mean().clamp(min=-1.0, max=1.0)
        loss = (1.0 - similarity).clamp(min=0.0, max=2.0)
        if return_similarity:
            return loss, similarity
        return loss
