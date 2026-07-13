import os
from contextlib import contextmanager

import torch.cuda.nvtx as nvtx


@contextmanager
def nvtx_range(enabled: bool, name: str):
    if enabled:
        nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            nvtx.range_pop()


# --- Bounded nsys capture (project-scoped, bench-profiling lineage) -----------------
#
# When QUANTEM_NSYS_CAPTURE="<start_step>:<n_steps>" is set (e.g. "10:5"), every rank
# calls torch.cuda.profiler.start() at grad step <start_step> and .stop() after
# <n_steps> more steps. Paired with
#   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop
# this bounds the trace to a few steady-state iterations so per-rank reports stay
# small enough to open. Inert (single early return) when the env var is unset.

_NSYS_SPEC: tuple[int, int] | None = None
_NSYS_STATE: str = "unparsed"  # unparsed -> idle -> started -> stopped / disabled


def nsys_capture_tick(grad_step: int) -> None:
    global _NSYS_SPEC, _NSYS_STATE
    if _NSYS_STATE in ("stopped", "disabled"):
        return
    if _NSYS_STATE == "unparsed":
        raw = os.environ.get("QUANTEM_NSYS_CAPTURE", "")
        if not raw:
            _NSYS_STATE = "disabled"
            return
        try:
            start_s, n_s = raw.split(":")
            start, n = int(start_s), int(n_s)
            if start < 1 or n < 1:
                raise ValueError
        except ValueError:
            print(f"QUANTEM_NSYS_CAPTURE malformed ({raw!r}); expected 'start:n' -> disabled")
            _NSYS_STATE = "disabled"
            return
        _NSYS_SPEC = (start, n)
        _NSYS_STATE = "idle"
    assert _NSYS_SPEC is not None
    start, n = _NSYS_SPEC
    if _NSYS_STATE == "idle" and grad_step >= start:
        import torch.cuda.profiler as _prof

        _prof.start()
        _NSYS_STATE = "started"
    elif _NSYS_STATE == "started" and grad_step >= start + n:
        import torch.cuda.profiler as _prof

        _prof.stop()
        _NSYS_STATE = "stopped"
