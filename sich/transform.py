from __future__ import annotations

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_IMAGE_INTERPOLATION = v2.InterpolationMode.BILINEAR


class TrainTransform:
    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        *,
        image_interpolation: v2.InterpolationMode = DEFAULT_IMAGE_INTERPOLATION,
    ) -> None:
        self.transforms = v2.Compose(
            [
                # 1. Resize
                v2.Resize(size=image_size, interpolation=image_interpolation, antialias=True),
                # 2. Geometric Augmentations
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=[-15, 15], interpolation=image_interpolation),
                # 3. Photometric Augmentations
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.5
                ),
                # 4. Type Conversions
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.long,
                        "others": None,
                    },
                    scale=True,
                ),
                # 5. Noise Augmentation
                v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.05, clip=True)], p=0.5),
                # 5. Normalization: Automatically applies to Image and skips the Mask.
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)

        # Apply the entire joint pipeline in one highly-optimized pass
        return self.transforms(img_tv, mask_tv)


class TestTransform:
    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        *,
        image_interpolation: v2.InterpolationMode = DEFAULT_IMAGE_INTERPOLATION,
    ) -> None:
        self.transforms = v2.Compose(
            [
                v2.Resize(size=image_size, interpolation=image_interpolation, antialias=True),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.long,
                        "others": None,
                    },
                    scale=True,
                ),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)

        return self.transforms(img_tv, mask_tv)
