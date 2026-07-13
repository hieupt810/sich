from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms.v2 import ColorJitter
from torchvision.transforms.v2 import functional as F

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_IMAGE_INTERPOLATION = F.InterpolationMode.BILINEAR
DEFAULT_MASK_INTERPOLATION = F.InterpolationMode.NEAREST


class TrainTransform:
    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        *,
        seed: int = 42,
        image_interpolation: F.InterpolationMode = DEFAULT_IMAGE_INTERPOLATION,
        mask_interpolation: F.InterpolationMode = DEFAULT_MASK_INTERPOLATION,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.rng = np.random.default_rng(seed=seed)
        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = F.to_image(image), F.to_image(mask)
        image, mask = F.to_dtype(image, dtype=torch.float32), F.to_dtype(mask, dtype=torch.long)

        image = F.resize(image, size=self.image_size, interpolation=self.image_interpolation)
        mask = F.resize(mask, size=self.image_size, interpolation=self.mask_interpolation)

        if self.rng.random() > 0.5:
            image, mask = F.vflip(image), F.vflip(mask)

        if self.rng.random() > 0.5:
            angle = self.rng.integers(-15, 16)
            image = F.rotate(image, angle=angle, interpolation=self.image_interpolation)
            mask = F.rotate(mask, angle=angle, interpolation=self.mask_interpolation)

        if self.rng.random() > 0.5:
            image = self.color_jitter(image)

        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask


class TestTransform:
    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        *,
        image_interpolation: F.InterpolationMode = DEFAULT_IMAGE_INTERPOLATION,
        mask_interpolation: F.InterpolationMode = DEFAULT_MASK_INTERPOLATION,
    ) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.image_interpolation = image_interpolation
        self.mask_interpolation = mask_interpolation

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = F.to_image(image), F.to_image(mask)
        image, mask = F.to_dtype(image, dtype=torch.float32), F.to_dtype(mask, dtype=torch.long)

        image = F.resize(image, size=self.image_size, interpolation=self.image_interpolation)
        mask = F.resize(mask, size=self.image_size, interpolation=self.mask_interpolation)

        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask
