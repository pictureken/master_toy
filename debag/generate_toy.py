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
        radius: int = 0.15,
        inner_radius: int = 0.30,
        outer_radius: int = 0.45,
    ) -> Tuple[list, list]:
        if not train:
            noise = 0
        samples = np.zeros((num_samples * 4, 2))
        targets = np.zeros(num_samples * 4, dtype="uint8")
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
        theta = 2 * np.pi * np.random.rand(num_samples)
        r = inner_radius + (outer_radius - inner_radius) * np.random.rand(num_samples)
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        idx = range(num_samples * (j + 1), num_samples * (j + 2))
        print(samples[idx].shape, np.c_[x, y].shape)
        samples[idx] = np.c_[x, y]
        targets[idx] = j + 1
        return (samples, targets)

    def generate_spiral_dataset(
        self, num_samples: int, num_classes: int, noise: float = 0.2
    ):
        samples = np.zeros((num_samples * num_classes, 2))
        targets = np.zeros(num_samples * num_classes, dtype="uint8")
        for j in range(num_classes):
            ix = range(num_samples * j, num_samples * (j + 1))
            r = np.linspace(0.0, 1, num_samples)  # radius
            t = (
                np.linspace(j * 6, (j + 1) * 6 * np.pi, num_samples)
                + np.random.randn(num_samples) * noise
            )  # theta
            samples[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            targets[ix] = j
        return samples, targets

    def id_gt(
        self,
        num_samples: int,
        num_classes: int,
        train: bool = True,
        noise: float = None,
        radius: int = 0.25,
        inner_radius: int = 0.30,
        outer_radius: int = 0.45,
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
        theta = 2 * np.pi * np.random.rand(num_samples)
        np.random.seed(2023)
        r = inner_radius + (outer_radius - inner_radius) * np.random.rand(num_samples)
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        idx = range(num_samples * (j + 1), num_samples * (j + 2))
        samples[idx] = np.c_[x, y]
        targets[idx] = j + 1
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

    def grid(self, num_samples):
        x = np.linspace(0, 1, num_samples)
        y = np.linspace(0, 1, num_samples)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten()
        yv = yv.flatten()
        xv = xv.reshape(len(xv), 1)
        yv = yv.reshape(len(yv), 1)
        return xv, yv


if __name__ == "__main__":
    generate = GenerateToyData(center=(0.5, 0.5))
    samples, targets = generate.id(num_samples=10000, num_classes=3, noise=0.4)
    plt.scatter(
        samples[0:10000, 0], samples[0:10000, 1], s=2, label="class" + str(targets[0])
    )
    plt.scatter(
        samples[10000:20000, 0],
        samples[10000:20000, 1],
        s=2,
        label="class" + str(targets[10000]),
    )
    plt.scatter(
        samples[20000:30000, 0],
        samples[20000:30000, 1],
        s=2,
        label="class" + str(targets[20000]),
    )
    plt.scatter(
        samples[30000:, 0], samples[30000:, 1], s=2, label="class" + str(targets[30000])
    )
    print(samples.mean(axis=0))
    plt.axis("square")
    plt.xlim([0, 1.0])
    plt.xlabel("x1", fontsize=13)
    plt.ylim([0, 1.0])
    plt.ylabel("x2", fontsize=13)
    plt.legend(fontsize=13)
    plt.savefig("./dataset.png", bbox_inches="tight", pad_inches=0)
