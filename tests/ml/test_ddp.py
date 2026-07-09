import torch

from quantem.core.ml.ddp import DDPMixin


class _DistributedOwner(DDPMixin):
    pass


def test_setup_distributed_normalizes_indexless_cuda_to_current_device(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    set_device_calls = []
    monkeypatch.setattr(torch.cuda, "set_device", set_device_calls.append)

    owner = _DistributedOwner()
    owner.setup_distributed(device="cuda")

    assert owner.device == torch.device("cuda:2")
    assert set_device_calls == [2]


def test_cuda_kernel_gate_allows_initialized_multirank_process_group(monkeypatch):
    from quantem.core import config

    monkeypatch.setattr(config, "get", lambda *args, **kwargs: True)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    assert config.cuda_kernels_enabled()


def test_cuda_kernel_gate_respects_the_global_kill_switch(monkeypatch):
    from quantem.core import config

    def fake_get(key, default=None):
        return key == "has_quantem_cuda"

    monkeypatch.setattr(config, "get", fake_get)

    assert not config.cuda_kernels_enabled()
