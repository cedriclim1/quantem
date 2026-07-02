"""Pure tests for tomography grad-step snapshot triggering and plumbing."""

from pathlib import Path

import numpy as np
import pytest

from quantem.tomography import tomography


class _ObjModel:
    def __init__(self):
        self.calls = 0
        self.volume = np.arange(8, dtype=np.float64).reshape(1, 2, 2, 2)

    @property
    def obj_view(self):
        self.calls += 1
        return self.volume


class _Logger:
    def __init__(self):
        self.scalars = []

    def log_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


@pytest.mark.parametrize(
    ("snapshot_every", "steps"),
    [
        (0, []),
        (1, [1, 2, 3, 4, 5, 6]),
        (2, [2, 4, 6]),
        (3, [3, 6]),
        (10, []),
    ],
)
def test_should_take_grad_step_snapshot(snapshot_every, steps):
    fired = [
        step
        for step in range(1, 7)
        if tomography._should_take_grad_step_snapshot(step, snapshot_every)
    ]
    assert fired == steps


def test_take_grad_step_snapshot_main_rank_writes_logs_and_callbacks(monkeypatch):
    mkdir_calls = []
    saved = {}

    def fake_mkdir(self, parents=False, exist_ok=False):
        mkdir_calls.append((self, parents, exist_ok))

    def fake_save(path, array):
        saved["path"] = path
        saved["array"] = array

    monkeypatch.setattr(tomography.Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(tomography.np, "save", fake_save)

    obj_model = _ObjModel()
    logger = _Logger()
    callback_calls = []

    tomography._take_grad_step_snapshot(
        obj_model=obj_model,
        grad_step=4,
        global_rank=0,
        logger=logger,
        snapshot_dir="snapshots",
        snapshot_callback=lambda step, volume: callback_calls.append((step, volume)),
    )

    assert obj_model.calls == 1
    assert mkdir_calls == [(Path("snapshots"), True, True)]
    assert saved["path"] == Path("snapshots") / "step_4.npy"
    assert saved["array"].dtype == np.float32
    np.testing.assert_array_equal(saved["array"], obj_model.volume.astype(np.float32))
    assert logger.scalars == [("snapshots/last_grad_step", 4.0, 4)]
    assert callback_calls == [(4, obj_model.volume)]


def test_take_grad_step_snapshot_non_main_rank_gets_collective_but_no_volume(monkeypatch):
    monkeypatch.setattr(
        tomography.np,
        "save",
        lambda *args, **kwargs: pytest.fail("non-main ranks must not write snapshots"),
    )

    obj_model = _ObjModel()
    logger = _Logger()
    callback_calls = []

    tomography._take_grad_step_snapshot(
        obj_model=obj_model,
        grad_step=6,
        global_rank=1,
        logger=logger,
        snapshot_dir="snapshots",
        snapshot_callback=lambda step, volume: callback_calls.append((step, volume)),
    )

    assert obj_model.calls == 1
    assert logger.scalars == []
    assert callback_calls == [(6, None)]


def test_take_grad_step_snapshot_callback_exception_propagates():
    obj_model = _ObjModel()

    def raise_callback(step, volume):
        raise RuntimeError(f"stop at {step}")

    with pytest.raises(RuntimeError, match="stop at 8"):
        tomography._take_grad_step_snapshot(
            obj_model=obj_model,
            grad_step=8,
            global_rank=1,
            logger=None,
            snapshot_dir=None,
            snapshot_callback=raise_callback,
        )
