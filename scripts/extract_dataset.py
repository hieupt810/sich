import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumePair:
    subject_id: str
    nii_path: Path
    msk_path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    nii_dir: str
    msk_dir: str
    out_dir: str
    window_center: float
    window_width: float
    workers: int
    axis: int


def get_runtime_config() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Extract dataset from raw NIfTI files.")
    parser.add_argument("--nii_dir", type=str, required=True, help="Dir containing NIfTI files.")
    parser.add_argument("--msk_dir", type=str, required=True, help="Dir containing mask files.")
    parser.add_argument("--out_dir", type=str, default="processed_data", help="Output directory.")
    parser.add_argument("--window_center", type=float, default=40.0, help="Window center.")
    parser.add_argument("--window_width", type=float, default=400.0, help="Window width.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--axis", type=int, default=2, help="Axis to slice the volumes (0, 1, 2).")

    args = parser.parse_args()
    return RuntimeConfig(
        nii_dir=args.nii_dir,
        msk_dir=args.msk_dir,
        out_dir=args.out_dir,
        window_center=args.window_center,
        window_width=args.window_width,
        workers=args.workers,
        axis=args.axis,
    )


def extract_subject_id(path: Path) -> str:
    """Extracts the subject ID strictly from the second part of the filename."""
    name = path.name.replace(".nii.gz", "").replace(".nii", "")
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid format (no 2nd part) | File: {path.name}")

    return parts[1]


def find_valid_mask(subject_id: str, msk_dir: Path) -> Path:
    """Finds and validates a mask file for the given subject. Must be in range [0, 5]."""
    for match in msk_dir.glob(f"*{subject_id}*.nrrd"):
        try:
            data, _ = nrrd.read(str(match))
            if np.min(data) >= 0 and np.max(data) <= 5:
                return match
        except Exception as e:
            LOGGER.debug("Mask validation failed | File: %s | Err: %s", match.name, e)

    raise FileNotFoundError(f"No valid mask [0, 5] found | ID: {subject_id}")


def build_pairs(nii_dir: Path, msk_dir: Path) -> list[VolumePair]:
    pairs: list[VolumePair] = []
    for nii_path in sorted(nii_dir.glob("*.nii*")):
        if not nii_path.is_file():
            continue

        try:
            subject_id = extract_subject_id(nii_path)
            msk_path = find_valid_mask(subject_id, msk_dir)
            pairs.append(VolumePair(subject_id=subject_id, nii_path=nii_path, msk_path=msk_path))
        except (ValueError, FileNotFoundError) as e:
            LOGGER.warning(str(e))

    return pairs


def apply_window(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    """Vectorized, robust windowing normalized to [0, 1]."""
    lower, upper = center - (width / 2), center + (width / 2)
    return np.clip((volume - lower) / (upper - lower), 0.0, 1.0)


def process_pair(pair: VolumePair, config: RuntimeConfig) -> bool:
    """Processes a single subject pair. Returns True if successful."""
    subject_dir = Path(config.out_dir) / pair.subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = nib.load(str(pair.nii_path)).get_fdata(dtype=np.float32)
        msk, _ = nrrd.read(str(pair.msk_path))

        # Core sanity checks
        if img.shape != msk.shape:
            LOGGER.error(
                "Mismatch shape | ID: %s | Img: %s | Msk: %s", pair.subject_id, img.shape, msk.shape
            )
            return False

        if config.axis < 0 or config.axis >= img.ndim:
            LOGGER.error(
                "Invalid axis | ID: %s | Axis: %d | Shape: %s",
                pair.subject_id,
                config.axis,
                img.shape,
            )
            return False

        # Optimized slicing: move target axis to index 0 and iterate directly (zero copy logic)
        img_arr = np.moveaxis(img, config.axis, 0)
        msk_arr = np.moveaxis(msk, config.axis, 0)

        for i, (img_slice, msk_slice) in enumerate(zip(img_arr, msk_arr, strict=False)):
            img_slice = apply_window(img_slice, config.window_center, config.window_width)

            stem = f"{pair.subject_id}_{i:03d}"
            np.save(subject_dir / f"{stem}_img.npy", img_slice)
            np.save(subject_dir / f"{stem}_msk.npy", msk_slice.astype(np.uint8))

        return True
    except Exception as e:
        LOGGER.error("Process crashed | ID: %s | Err: %s", pair.subject_id, e)
        return False


def main(config: RuntimeConfig) -> int:
    nii_dir = Path(config.nii_dir)
    msk_dir = Path(config.msk_dir)

    if not nii_dir.is_dir():
        LOGGER.error("Missing NIfTI dir | Path: %s", nii_dir)
        return 1

    LOGGER.info("Discovering and validating pairs...")
    pairs = build_pairs(nii_dir, msk_dir)

    if not pairs:
        LOGGER.error("Abort | No valid pairs found.")
        return 0

    success, failure = 0, 0
    LOGGER.info("Starting processing | Pairs: %d | Workers: %d", len(pairs), config.workers)

    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(process_pair, p, config): p for p in pairs}
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                failure += 1

    LOGGER.info("Finished | Success: %d | Failed: %d", success, failure)
    return 0


if __name__ == "__main__":
    exit(main(get_runtime_config()))
