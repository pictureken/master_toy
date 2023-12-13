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
        radius: float = 0.15,
        inner_radius: int = 0.30,
        outer_radius: int = 0.45,
        train: bool = True,
        noise: float = 0.4,
        dim: int = 2,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.radius = radius
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.center = center
        self.dim = dim

        if train:
            # train dataset
            random.seed(2022)
            self.noise = noise
            samples, targets = self._gen_train()

        else:
            # test dataset
            random.seed(2023)
            samples, targets = self._gen_test()

        self.samples = samples
        self.targets = targets

    def _gen_train(self):
        samples = np.zeros((self.num_samples * self.num_classes, self.dim))
        targets = np.zeros(self.num_samples * self.num_classes, dtype="uint8")
        for j in range(self.num_classes - 1):
            idx = range(self.num_samples * j, self.num_samples * (j + 1))
            r = np.linspace(0.0, self.radius, self.num_samples)  # radius
            t = (
                np.linspace(j * 4, (j + 1) * 4, self.num_samples)
                + np.random.randn(self.num_samples) * self.noise
            )  # theta
            samples[idx] = np.c_[
                r * np.sin(t) + self.center[0],
                r * np.cos(t) + self.center[1],
            ]
            targets[idx] = j
        t = 2 * np.pi * np.random.rand(self.num_samples)
        r = self.inner_radius + (
            self.outer_radius - self.inner_radius
        ) * np.random.rand(self.num_samples)
        idx = range(self.num_samples * (j + 1), self.num_samples * (j + 2))
        samples[idx] = np.c_[
            r * np.cos(t) + self.center[0], r * np.sin(t) + self.center[1]
        ]
        targets[idx] = j + 1
        return (samples, targets)

    def _gen_test(self):
        SPLIT = 100
        CLASS_LIST = [2, 0, 1]
        samples = np.zeros((self.num_samples * self.num_classes, 2))
        targets = np.zeros(self.num_samples * self.num_classes, dtype="uint8")
        split_samples = self.num_samples // SPLIT
        total = 0
        for j in range(self.num_classes - 1):
            for i in range(SPLIT):
                idx = range(total, total + split_samples)
                r = np.linspace(0.0, self.radius, split_samples)  # radius
                t = np.linspace(
                    (j + (0.25 + i * 0.005)) * 4.2,
                    ((j + (0.25 + i * 0.005)) + 1) * 4.2,
                    split_samples,
                )  # theta
                samples[idx] = np.c_[
                    r * np.sin(t) + self.center[0],
                    r * np.cos(t) + self.center[1],
                ]
                targets[idx] = CLASS_LIST[j]
                total += split_samples
        np.random.seed(2023)
        theta = 2 * np.pi * np.random.rand(self.num_samples)
        r = self.inner_radius + (
            self.outer_radius - self.inner_radius
        ) * np.random.rand(self.num_samples)
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        idx = range(self.num_samples * (j + 1), self.num_samples * (j + 2))
        samples[idx] = np.c_[x, y]
        targets[idx] = j + 1
        return (samples, targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        label = self.targets[index]
        data = torch.tensor(sample).float()
        return data, label

    def __len__(self) -> int:
        return len(self.samples)


class TTAGenerateToyDataset(Dataset):
    def __init__(
        self,
        transforms,
        num_samples: int,
        num_classes: int,
        center: Tuple[float, float],
        radius: float = 0.15,
        train: bool = True,
        noise: float = 0.4,
    ) -> None:
        super().__init__(
            transforms, num_samples, num_classes, radius, center, noise, train
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        label = self.targets[index]
        data = torch.tensor(sample).float()
        return data, label


class GenerateToyCircleDataset(Dataset):
    def __init__(
        self,
        transforms,
        num_samples: int,
        center: Tuple[float, float],
        inner_radius: float = 0.35,
        outer_radius: float = 0.5,
    ) -> None:
        theta = 2 * np.pi * np.random.rand(num_samples)
        r = inner_radius + (outer_radius - inner_radius) * np.random.rand(num_samples)
        x = r * np.cos(theta) + center[0]
        y = r * np.sin(theta) + center[1]
        x = x.reshape(len(x), 1)
        y = y.reshape(len(y), 1)
        self.samples = np.concatenate((x, y), axis=1)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.samples[index]
        data = torch.tensor(sample).float()

        return data

    def __len__(self) -> int:
        return len(self.samples)


class GenerateGridDataset(Dataset):
    def __init__(self, num_samples) -> None:
        x = np.linspace(0, 1, num_samples)
        y = np.linspace(0, 1, num_samples)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten()
        yv = yv.flatten()
        xv = xv.reshape(len(xv), 1)
        yv = yv.reshape(len(yv), 1)
        self.samples = np.concatenate((xv, yv), axis=1)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.samples[index]
        data = torch.tensor(sample).float()

        return data

    def __len__(self) -> int:
        return len(self.samples)


class TTAGenerateGridDataset(GenerateGridDataset):
    def __init__(self, num_samples, k, sigma) -> None:
        super().__init__(num_samples)
        self.k = k
        self.sigma = sigma
        self.mean = 0

    def __getitem__(self, index: int) -> torch.Tensor:
        img_sum = torch.Tensor(self.k, 2).zero_()
        sample = self.samples[index]
        for i in range(self.k):
            if i == 0:
                img_sum[i, :] += torch.tensor(sample).float()
                continue
            ipsilon = np.random.normal(loc=self.mean, scale=self.sigma, size=(2))
            sample_aug = sample + ipsilon
            img_sum[i, :] += torch.tensor(sample_aug).float()
        return img_sum


if __name__ == "__main__":
    from torchvision import transforms

    transforms = transforms.Compose([transforms.ToTensor()])
    # train_dataset = GenerateToyDataset(
    #     transforms,
    #     num_samples=10000,
    #     num_classes=3,
    #     center=(0.5, 0.5),
    #     train=True,
    #     noise=0.5,
    # )
    train_dataset = TTAGenerateGridDataset(num_samples=100, k=10, sigma=1)
    a = train_dataset.__getitem__(0)
    print(a.shape)
