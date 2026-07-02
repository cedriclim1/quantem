import torch
from torch import nn

from quantem.tomography.tomography import Tomography


class _CountingValidationLoader:
    def __init__(self):
        self.iter_calls = 0
        self.batch_counts: list[int] = []

    def __iter__(self):
        self.iter_calls += 1
        batches = [
            {
                "x": torch.tensor([[1.0], [2.0]]),
                "target_value": torch.tensor([0.0, 0.0]),
            },
            {
                "x": torch.tensor([[3.0]]),
                "target_value": torch.tensor([0.0]),
            },
        ]
        self.batch_counts.append(len(batches))
        return iter(batches)


class _ValidationDataset(nn.Module):
    def get_coords(
        self, batch: dict[str, torch.Tensor], object_extent: int, num_samples_per_ray: int
    ) -> torch.Tensor:
        return batch["x"]

    def integrate_rays(
        self, densities: torch.Tensor, num_samples_per_ray: int, target_values_len: int
    ) -> torch.Tensor:
        return densities.reshape(target_values_len)


class _ValidationObject(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.model(coords).squeeze(-1)


def _tomography_for_validation() -> Tomography:
    tomography = object.__new__(Tomography)
    tomography.device = torch.device("cpu")
    tomography.world_size = 1
    tomography._dset = _ValidationDataset()
    tomography._obj_model = _ValidationObject()
    return tomography


def _evaluate(tomography: Tomography, loader: _CountingValidationLoader) -> float:
    loss = tomography._evaluate_validation_loss(
        dataloader=loader,
        num_samples_per_ray=1,
        object_extent=1,
        loss_func=nn.MSELoss(),
    )
    assert loss is not None
    return loss


def test_validation_loss_depends_on_current_model_parameters():
    tomography = _tomography_for_validation()
    loader = _CountingValidationLoader()

    with torch.no_grad():
        tomography.obj_model.model.weight.fill_(1.0)
    first_loss = _evaluate(tomography, loader)

    with torch.no_grad():
        tomography.obj_model.model.weight.fill_(2.0)
    second_loss = _evaluate(tomography, loader)

    assert first_loss != second_loss


def test_validation_loss_reiterates_fresh_dataloader_each_call():
    tomography = _tomography_for_validation()
    loader = _CountingValidationLoader()

    _evaluate(tomography, loader)
    _evaluate(tomography, loader)

    assert loader.iter_calls == 2
    assert loader.batch_counts == [2, 2]


def test_validation_loss_restores_train_mode():
    tomography = _tomography_for_validation()
    loader = _CountingValidationLoader()
    tomography.obj_model.model.train()
    tomography.dset.train()

    _evaluate(tomography, loader)

    assert tomography.obj_model.model.training
    assert tomography.dset.training
