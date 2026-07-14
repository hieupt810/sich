import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# Configure concise, structured logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        transform: Callable | None = None,
        train_ratio: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.transform = transform

        if not self.root.is_dir():
            raise NotADirectoryError(f"Missing root directory | Path: {self.root}")

        if self.split not in {"train", "val", "all"}:
            raise ValueError(f"Invalid split | Expected train/val/all | Got: {self.split}")

        # Collect valid patient directories
        patient_ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not patient_ids:
            raise ValueError(f"No patient data found | Path: {self.root}")

        # Patient-level splitting ensuring deterministic reproducibility
        if self.split != "all":
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(patient_ids)
            split_idx = int(len(patient_ids) * train_ratio)

            if self.split == "train":
                patient_ids = patient_ids[:split_idx]
            else:  # val
                patient_ids = patient_ids[split_idx:]

        self.samples = self._load_samples(patient_ids)

    def _load_samples(self, patient_ids: list[str]) -> list[tuple[Path, Path]]:
        samples: list[tuple[Path, Path]] = []

        for pid in patient_ids:
            patient_dir = self.root / pid
            for img_path in patient_dir.glob("*_img.npy"):
                # Fast string slicing instead of `.replace`
                msk_path = patient_dir / f"{img_path.name[:-8]}_msk.npy"

                if msk_path.exists():
                    samples.append((img_path, msk_path))
                else:
                    LOGGER.debug("Missing mask | Img: %s", img_path.name)

        LOGGER.info("Dataset loaded | Split: %s | Samples: %d", self.split, len(samples))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img_path, msk_path = self.samples[index]

        image = np.load(img_path)
        mask = np.load(msk_path)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def compute_stats(dataset: SegmentationDataset) -> tuple[float, float]:
    """Computes dataset mean & std using robust Float64 accumulators to prevent overflow."""

    original_transform = dataset.transform
    dataset.transform = None

    pixel_sum = np.float64(0.0)
    pixel_sq_sum = np.float64(0.0)
    num_pixels = 0

    try:
        for img, _ in tqdm(dataset, desc="Stats Computation", leave=False):
            # Upcast to float64 for stable accumulation over thousands of images
            img = img.astype(np.float64)
            pixel_sum += img.sum()
            pixel_sq_sum += (img**2).sum()
            num_pixels += img.size
    finally:
        dataset.transform = original_transform

    if num_pixels == 0:
        raise ValueError("Dataset is empty. Cannot compute stats.")

    mean = pixel_sum / num_pixels
    variance = (pixel_sq_sum / num_pixels) - (mean**2)
    std = np.sqrt(max(0.0, variance))

    return float(mean), float(std)


@dataclass(frozen=True)
class DatasetRuntimeConfig:
    root: str
    split: str
    verbose: bool


def get_runtime_config() -> DatasetRuntimeConfig:
    parser = argparse.ArgumentParser(description="Dataset utility and statistics.")
    parser.add_argument("--root", type=str, default="processed_data", help="Root dir of dataset.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()
    return DatasetRuntimeConfig(root=args.root, split=args.split, verbose=args.verbose)


def main(config: DatasetRuntimeConfig | None = None) -> int:
    config = config or get_runtime_config()

    if config.verbose:
        LOGGER.setLevel(logging.DEBUG)

    LOGGER.info("Starting run | Root: %s | Split: %s", config.root, config.split)

    try:
        dataset = SegmentationDataset(root=config.root, split=config.split)
        mean, std = compute_stats(dataset)
        LOGGER.info("Stats calculated | Mean: %.4f | Std: %.4f", mean, std)
        return 0
    except Exception as e:
        LOGGER.error("Execution failed | Err: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())
