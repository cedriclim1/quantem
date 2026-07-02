"""Tests for ``quantem.tomography.logger_tomography``.

``LoggerTomography`` is a thin tensorboard wrapper that the reconstruction loop only drives
when a ``log_dir`` is passed, so the end-to-end recon tests never exercise it. These CPU,
always-on tests construct a logger against a ``tmp_path`` and drive each method with small
stubs that expose only the attributes the logger reads, asserting the calls run and write
event files. Matplotlib backend is ``Agg`` (set in the root conftest), so figure logging is
headless.
"""

import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from quantem.tomography.logger_tomography import LoggerTomography


def _make_logger(tmp_path) -> LoggerTomography:
    return LoggerTomography(
        log_dir=str(tmp_path),
        run_prefix="test_tomo",
        run_suffix="",
        log_images_every=1,
    )


def test_init_creates_log_dir(tmp_path):
    logger = _make_logger(tmp_path)
    try:
        assert logger.log_dir.exists()
        assert logger.log_dir.name.startswith("test_tomo_")
    finally:
        logger.close()


def test_log_epoch_writes_events(tmp_path):
    logger = _make_logger(tmp_path)
    try:
        logger.log_epoch(epoch=0, loss=1.0, tilt_series_loss=0.8, soft_loss=0.2)
        logger.flush()
        events = list(logger.log_dir.glob("events.out.tfevents.*"))
        assert events, "log_epoch should have written a tensorboard event file"
    finally:
        logger.close()


def test_log_iter_unpacks_learning_rates(tmp_path):
    logger = _make_logger(tmp_path)
    obj_model = SimpleNamespace(_soft_constraint_losses=[0.3])
    try:
        logger.log_iter(
            object_model=obj_model,
            iter=2,
            consistency_loss=0.5,
            total_loss=0.7,
            learning_rates={"object": 1e-3, "pose": 1e-2},
            num_samples_per_ray=16,
            val_loss=0.4,
        )
        logger.flush()
        assert list(logger.log_dir.glob("events.out.tfevents.*"))
    finally:
        logger.close()


def test_log_iter_without_val_loss(tmp_path):
    logger = _make_logger(tmp_path)
    obj_model = SimpleNamespace(_soft_constraint_losses=[0.1])
    try:
        # val_loss defaults to None -> the val branch must be skipped without error.
        logger.log_iter(
            object_model=obj_model,
            iter=0,
            consistency_loss=0.5,
            total_loss=0.6,
            learning_rates={},
            num_samples_per_ray=8,
        )
        logger.flush()
    finally:
        logger.close()


def test_log_iter_images(tmp_path):
    logger = _make_logger(tmp_path)
    n_tilts = 5
    dataset_model = SimpleNamespace(
        z1_params=torch.linspace(-1.0, 1.0, n_tilts),
        z3_params=torch.linspace(1.0, -1.0, n_tilts),
        shifts_params=torch.zeros(n_tilts, 2),
    )
    pred_volume = np.random.default_rng(0).random((2, 6, 6, 6)).astype(np.float32)
    try:
        logger.log_iter_images(
            pred_volume=pred_volume,
            dataset_model=dataset_model,
            iter=1,
        )
        logger.flush()
        assert list(logger.log_dir.glob("events.out.tfevents.*"))
    finally:
        logger.close()


def test_invalid_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="tensorboard.*wandb"):
        LoggerTomography(
            log_dir=str(tmp_path),
            run_prefix="test_tomo",
            mode="invalid",
        )


def test_wandb_mode_logs_under_run_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    init_calls = []
    logged = []
    defined_metrics = []

    class FakeConfig(dict):
        def update(self, values, allow_val_change=False):
            super().update(values)

    class FakeRun:
        def __init__(self):
            self.config = FakeConfig()
            self.finished = False

        def define_metric(self, name, **kwargs):
            defined_metrics.append((name, kwargs))

        def log(self, data, **kwargs):
            logged.append((data, kwargs))

        def finish(self):
            self.finished = True

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    fake_wandb = SimpleNamespace(
        Image=lambda image: ("image", image),
        Histogram=lambda values: ("histogram", values),
        init=fake_init,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = LoggerTomography(
        log_dir=str(tmp_path),
        run_prefix="test_tomo",
        run_suffix="wandb",
        log_images_every=1,
        mode="wandb",
        wandb_config={"batch_size": 4},
    )
    try:
        assert logger.mode == "wandb"
        assert os.environ["WANDB_MODE"] == "offline"
        assert logger.log_dir.exists()
        assert (logger.log_dir / "wandb").exists()
        assert init_calls[-1]["dir"] == str(logger.log_dir)
        assert init_calls[-1]["config"] == {"batch_size": 4}

        logger.attach_config({"num_iter": 2})
        logger.log_scalar("loss/total", 1.0, 1)
        logger.log_image("volume/sum_z_0", np.ones((2, 2), dtype=np.float32), 1)
        logger.log_histogram("weights/object", np.array([0.0, 1.0], dtype=np.float32), 1)
        logger.log_text("config/notes", "offline test", 1)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])
        logger.log_figure("figures/test", fig, 1)
        plt.close(fig)
        logger.log_scalar(
            "snapshots/last_grad_step", 48.0, 48, step_domain="grad_step"
        )
        logger.log_scalar("loss/total", 0.8, 2)
        logger.flush()
    finally:
        logger.close()

    assert defined_metrics == [
        ("*", {"step_metric": "epoch"}),
        ("snapshots/*", {"step_metric": "grad_step"}),
    ]
    assert all("step" not in kwargs for _, kwargs in logged)
    assert logged[0] == ({"loss/total": 1.0, "epoch": 1}, {})
    assert "volume/sum_z_0" in logged[1][0]
    assert logged[1][0]["epoch"] == 1
    assert logged[2][0]["epoch"] == 1
    assert logged[3] == ({"config/notes": "offline test", "epoch": 1}, {})
    assert "figures/test" in logged[4][0]
    assert logged[4][0]["epoch"] == 1
    assert logged[5] == ({"snapshots/last_grad_step": 48.0, "grad_step": 48}, {})
    assert logged[6] == ({"loss/total": 0.8, "epoch": 2}, {})
