from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        transform: Callable | None = None,
        *,
        seed: int = 42,
        train_ratio: float = 0.7,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.transform = transform

        if not self.root.is_dir():
            raise ValueError(f"Root path {self.root} is not a directory.")

        if self.split not in ["train", "val", "all"]:
            raise ValueError(f"Split must be 'train', 'val', or 'all', got '{self.split}'.")

        rng = np.random.default_rng(seed=seed)

        # Get the patient IDs
        patient_ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()])

        # Patient-level splitting
        rng.shuffle(patient_ids)
        split_index = int(len(patient_ids) * train_ratio)

        if self.split == "train":
            patient_ids = patient_ids[:split_index]
        elif self.split == "val":
            patient_ids = patient_ids[split_index:]

        self.samples = self._load_samples(patient_ids)

    def _load_samples(self, patient_ids: list[str]) -> list[tuple[Path, Path]]:
        samples: list[tuple[Path, Path]] = []
        for patient_id in patient_ids:
            patient_dir = self.root / patient_id
            for image_path in patient_dir.glob("*_img.npy"):
                mask_path = image_path.with_name(image_path.name.replace("_img.npy", "_mask.npy"))
                if mask_path.exists():
                    samples.append((image_path, mask_path))
                else:
                    LOGGER.warning(
                        "Mask file '%s' does not exist for image '%s'.", mask_path, image_path
                    )

        LOGGER.info("Loaded %d samples for split '%s'.", len(samples), self.split)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = np.load(image_path)
        mask = np.load(mask_path)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def compute_stats(dataset: SegmentationDataset) -> tuple[np.float32, np.float32]:
    """Computes the mean and standard deviation of the dataset."""
    pixel_sum = np.float32(0.0)
    pixel_sq_sum = np.float32(0.0)
    num_pixels = np.float32(0.0)

    # Ensure transforms are off to compute raw stats
    dataset.transforms = None

    for image, _ in tqdm(dataset, desc="Computing dataset statistics"):
        image = image.astype(np.float32)

        pixel_sum += image.sum()
        pixel_sq_sum += (image**2).sum()
        num_pixels += image.size

    if num_pixels == 0:
        raise ValueError("No pixels found in the dataset.")

    mean = pixel_sum / num_pixels
    variance = (pixel_sq_sum / num_pixels) - (mean**2)
    std = np.sqrt(variance)

    return mean, std


@dataclass
class DatasetRuntimeConfig:
    root: str = "processed_data"
    verbose: bool = False


def get_runtime_config() -> DatasetRuntimeConfig:
    """Parses command-line arguments and returns a DatasetRuntimeConfig instance."""
    parser = argparse.ArgumentParser(description="Compute dataset statistics.")
    defaults = DatasetRuntimeConfig()

    parser.add_argument(
        "--root", type=str, default=defaults.root, help="Root directory of the dataset."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()
    return DatasetRuntimeConfig(**vars(args))


def main(config: DatasetRuntimeConfig | None = None) -> None:
    config = config or get_runtime_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    LOGGER.info("Computing dataset statistics with config: %s", asdict(config))
    try:
        dataset = SegmentationDataset(root=config.root)
        mean, std = compute_stats(dataset)
        LOGGER.info("Dataset mean: %.4f, std: %.4f", mean, std)
    except Exception as e:
        LOGGER.error("Error computing dataset statistics: %s", e)


if __name__ == "__main__":
    main()
