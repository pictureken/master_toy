import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class GenerateToyDataset(Dataset):
    def __init__(
        self,
        transforms,
        num_samples: int,
        num_classes: int,
        center: Tuple[float, float],
        radius: float = 0.25,
        train: bool = True,
        noise: float = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms

        random.seed(2022)
        if train and noise is None:
            raise ValueError("Noise must be specified when train is True.")
        else:
            # テストデータの場合はNoiseを入れず別のシードに固定
            noise = 0
            random.seed(2023)
        samples = np.zeros((num_samples * num_classes, 2))
        targets = np.zeros(num_samples * num_classes, dtype="uint8")
        for j in range(num_classes):
            idx = range(num_samples * j, num_samples * (j + 1))
            r = np.linspace(0.0, 1, num_samples)  # radius
            t = (
                np.linspace(j * 4, (j + 1) * 4, num_samples)
                + np.random.randn(num_samples) * noise
            )  # theta
            samples[idx] = np.c_[
                r * np.sin(t) * radius + center[0],
                r * np.cos(t) * radius + center[1],
            ]
            targets[idx] = j
        self.samples = samples
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        label = self.targets[index]
        # data = self.transforms(sample)
        data = torch.tensor(sample).float()
        return data, label

    def __len__(self) -> int:
        return len(self.samples)


class GenerateToyOodDataset(Dataset):
    def __init__(
        self,
        transforms,
        num_samples: int,
        center: Tuple[float, float],
        inner_radius: float = 0.35,
        outer_radius: float = 0.5,
    ) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        label = self.targets[index]

        data = self.transforms(sample)

        return data, label

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == "__main__":
    from torchvision import transforms

    transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = GenerateToyDataset(
        transforms,
        num_samples=10000,
        num_classes=3,
        center=(0.5, 0.5),
        train=True,
        noise=0.5,
    )
