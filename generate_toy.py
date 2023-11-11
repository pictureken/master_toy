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
            r = np.linspace(0.0, 1, num_samples)  # radius
            t = (
                np.linspace(j * 4, (j + 1) * 4, num_samples)
                + np.random.randn(num_samples) * noise
            )  # theta
            samples[idx] = np.c_[
                r * np.sin(t) * radius + self.center[0],
                r * np.cos(t) * radius + self.center[1],
            ]
            targets[idx] = j
        return (samples, targets)

    # ドーナツ状のOODデータを生成
    def ood(
        self, num_sample: int, inner_radius: int, outer_radius: int
    ) -> Tuple[list, list]:
        theta = 2 * np.pi * np.random.rand(num_sample)
        r = inner_radius + (outer_radius - inner_radius) * np.random.rand(num_sample)
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        return (x, y)


if __name__ == "__main__":
    generate = GenerateToyData(center=(0.5, 0.5))
    samples, targets = generate.id(
        num_samples=10000, num_classes=3, train=True, noise=0.5, radius=0.25
    )
    plt.scatter(samples[:, 0], samples[:, 1], s=0.1, c=targets)
    x, y = generate.ood(num_sample=10000, inner_radius=0.35, outer_radius=0.5)
    plt.scatter(x, y, s=0.05)
    plt.axis("square")
    plt.show()
    plt.clf()
    samples, targets = generate.id(
        num_samples=10000, num_classes=3, train=False, radius=0.25
    )
    plt.scatter(samples[:, 0], samples[:, 1], s=0.05, c=targets)
    plt.axis("square")
    plt.show()
    plt.clf()
