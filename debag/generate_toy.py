from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GenerateToyData:
    center: Tuple[int, int] = (0, 0)

    # スパイラル状の三種類のIDデータを生成
    def id(
        self,
        num_samples: int,
        num_classes: int,
        train: bool = True,
        noise: float = None,
        radius: int = 4,
    ) -> Tuple[list, list]:
        if not train:
            noise = 0
        samples = np.zeros((num_samples * num_classes, 2))
        targets = np.zeros(num_samples * num_classes, dtype="uint8")
        for j in range(num_classes):
            idx = range(num_samples * j, num_samples * (j + 1))
            r = np.linspace(0.0, radius, num_samples)  # radius
            t = (
                np.linspace(j * 4, (j + 1) * 4, num_samples)
                + np.random.randn(num_samples) * noise
            )  # theta
            samples[idx] = np.c_[
                r * np.sin(t) + self.center[0],
                r * np.cos(t) + self.center[1],
            ]
            targets[idx] = j
        return (samples, targets)

    def id_gt(
        self,
        num_samples: int,
        num_classes: int,
        train: bool = True,
        noise: float = None,
        radius: int = 4,
    ) -> Tuple[list, list]:
        samples = np.zeros((num_samples * num_classes, 2))
        targets = np.zeros(num_samples * num_classes, dtype="uint8")
        split = 100
        num_samples = num_samples // split
        total = 0
        class_list = [2, 0, 1]
        for j in range(num_classes):
            for i in range(split):
                idx = range(total, total + num_samples)
                r = np.linspace(0.0, radius, num_samples, endpoint=False)  # radius
                t = np.linspace(
                    (j + (0.25 + i * 0.005)) * 4.2,
                    ((j + (0.25 + i * 0.005)) + 1) * 4.2,
                    num_samples,
                    endpoint=False,
                )  # theta
                samples[idx] = np.c_[
                    r * np.sin(t) + self.center[0],
                    r * np.cos(t) + self.center[1],
                ]
                targets[idx] = class_list[j]
                total += num_samples
        return (samples, targets)

    # ドーナツ状のOODデータを生成
    def ood(
        self,
        num_samples: int,
        inner_radius: int = 0.35,
        outer_radius: int = 0.5,
    ) -> Tuple[list, list]:
        theta = 2 * np.pi * np.random.rand(num_samples)
        r = inner_radius + (outer_radius - inner_radius) * np.random.rand(num_samples)
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        return (x, y)

    def grid(self):
        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1, 1000)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten()
        yv = yv.flatten()
        xv = xv.reshape(len(xv), 1)
        yv = yv.reshape(len(yv), 1)
        return xv, yv


if __name__ == "__main__":
    generate = GenerateToyData(center=(0.5, 0.5))
    samples, targets = generate.id(
        num_samples=10000, num_classes=3, train=True, noise=0.4, radius=0.25
    )
    plt.scatter(samples[:, 0], samples[:, 1], s=1, c=targets)
    plt.axis("square")
    plt.show()
