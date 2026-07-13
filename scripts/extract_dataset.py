import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumePair:
    subject_id: str
    nii_path: Path
    msk_path: Path


@dataclass
class RuntimeConfig:
    nii_dir: str
    msk_dirs: list[str]
    out_dir: str
    id_parts: int
    window_center: float
    window_width: float
    workers: int
    axis: int


def get_runtime_config():
    """Parses command-line arguments and returns a RuntimeConfig object."""
    parser = argparse.ArgumentParser(description="Extract dataset from raw NIfTI files.")

    parser.add_argument(
        "--nii_dir", type=str, required=True, help="Directory containing raw NIfTI files."
    )
    parser.add_argument(
        "--msk_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of directories containing mask files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="processed_data",
        help="Directory to save processed data.",
    )
    parser.add_argument(
        "--id_parts", type=int, default=2, help="Number of parts to split the subject ID into."
    )
    parser.add_argument(
        "--window_center",
        type=float,
        default=40.0,
        help="Center of the window for intensity normalization.",
    )
    parser.add_argument(
        "--window_width",
        type=float,
        default=300.0,
        help="Width of the window for intensity normalization.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes for parallel processing."
    )
    parser.add_argument(
        "--axis", type=int, default=2, help="Axis along which to slice the volumes (0, 1, or 2)."
    )

    args = parser.parse_args()
    return RuntimeConfig(
        nii_dir=args.nii_dir,
        msk_dirs=args.msk_dirs,
        out_dir=args.out_dir,
        id_parts=args.id_parts,
        window_center=args.window_center,
        window_width=args.window_width,
        workers=args.workers,
        axis=args.axis,
    )


def extract_subject_id(path: Path, id_parts: int) -> str:
    """Extracts the subject ID from the file path based on the specified number of parts."""
    stem = path.stem.replace(".nii.gz", "").replace(".nii", "")
    parts = stem.split("_")
    if len(parts) < id_parts:
        raise ValueError(
            f"File name {path.name} does not contain enough parts to extract subject ID."
        )
    return "_".join(parts[:id_parts])


def load_nii(path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    """Loads a NIfTI file and returns its data as a NumPy array."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=dtype)
    return data


def load_nrrd(path: Path, dtype: np.dtype = np.long, max_value: int = 6) -> np.ndarray:
    """Loads a NRRD file and returns its data as a NumPy array."""
    data, _ = nrrd.read(str(path))
    if np.max(data) > max_value:
        raise ValueError(f"Mask file '{path}' contains values greater than {max_value}.")
    return data.astype(dtype)


def find_mask_path(subject_id: str, msk_dirs: list[Path]) -> Path:
    """Finds the corresponding mask file for a given subject ID in the provided directories."""
    matches: list[Path] = []
    pattern = f"{subject_id}*.seg.nrrd"
    for msk_dir in msk_dirs:
        matches.extend(msk_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No mask file found for subject ID '{subject_id}' in provided directories."
        )

    taken = None
    if len(matches) > 1:
        LOGGER.warning("Multiple mask files found for subject ID '%s'.", subject_id)
        for match in matches:
            try:
                _ = load_nrrd(match)
            except Exception as e:
                LOGGER.warning(e)
                continue

            # If we reach here, the mask file is valid
            taken = match
            break

    if taken is None:
        raise FileNotFoundError(
            f"No valid mask file found for subject ID '{subject_id}' in provided directories."
        )
    return taken


def discover_volumes(directory: Path, id_parts: int) -> dict[str, Path]:
    files = sorted([p for p in directory.glob("*.nii*") if p.is_file()])
    volumes: dict[str, Path] = {}
    for file in files:
        try:
            subject_id = extract_subject_id(file, id_parts=id_parts)
            previous = volumes.get(subject_id)
            if previous:
                LOGGER.warning(
                    "Duplicate subject ID '%s' found. Previous file: '%s', Current file: '%s'",
                    subject_id,
                    previous.stem,
                    file.stem,
                )
            volumes[subject_id] = file
        except ValueError as e:
            LOGGER.warning(e)
    return volumes


def build_pairs(nii_dir: Path, msk_dirs: list[Path], id_parts: int) -> list[VolumePair]:
    volumes = discover_volumes(nii_dir, id_parts)
    pairs: list[VolumePair] = []
    for subject_id, nii_path in volumes.items():
        try:
            msk_path = find_mask_path(subject_id, msk_dirs)
            pairs.append(VolumePair(subject_id=subject_id, nii_path=nii_path, msk_path=msk_path))
        except FileNotFoundError as e:
            LOGGER.warning(e)
    return pairs


def apply_window(volume: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Applies windowing to the volume data."""
    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)
    volume = np.clip(volume, lower_bound, upper_bound)
    volume = (volume - lower_bound) / (upper_bound - lower_bound)  # Normalize to [0, 1]
    return volume


def process_pair(pair: VolumePair, config: RuntimeConfig) -> bool:
    subject_dir = Path(config.out_dir) / pair.subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    img = load_nii(pair.nii_path)
    msk = load_nrrd(pair.msk_path)
    if img.shape != msk.shape:
        LOGGER.warning(
            "Shape mismatch for subject ID '%s': NIfTI shape %s, Mask shape %s. Skipping.",
            pair.subject_id,
            img.shape,
            msk.shape,
        )
        return False

    if config.axis < 0 or config.axis >= img.ndim:
        LOGGER.warning(
            "Invalid axis %d for subject ID '%s' with image shape %s. Skipping.",
            config.axis,
            pair.subject_id,
            img.shape,
        )
        return False

    img_slices = [np.take(img, indices=i, axis=config.axis) for i in range(img.shape[config.axis])]
    msk_slices = [np.take(msk, indices=i, axis=config.axis) for i in range(msk.shape[config.axis])]
    if len(img_slices) != len(msk_slices):
        LOGGER.warning(
            "Slice count mismatch for subject ID '%s': NIfTI slices %d, Mask slices %d. Skipping.",
            pair.subject_id,
            len(img_slices),
            len(msk_slices),
        )
        return False

    for slice_index, (img_slice, msk_slice) in enumerate(zip(img_slices, msk_slices, strict=False)):
        img_slice = apply_window(
            img_slice, window_center=config.window_center, window_width=config.window_width
        )

        stem = f"{pair.subject_id}_{slice_index:03d}"
        np.save(subject_dir / f"{stem}_img.npy", img_slice)
        np.save(subject_dir / f"{stem}_msk.npy", msk_slice)


def main(config: RuntimeConfig) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate input directories
    nii_dir = Path(config.nii_dir)
    msk_dirs = [Path(d) for d in config.msk_dirs if Path(d).is_dir()]
    out_dir = Path(config.out_dir)

    if not nii_dir.is_dir():
        LOGGER.error("NIfTI directory '%s' does not exist.", nii_dir)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(nii_dir, msk_dirs, config.id_parts)
    if not pairs:
        LOGGER.error("No valid volume pairs found. Exiting.")
        return 0

    success_count, failure_count = 0, 0
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(process_pair, pair, config): pair for pair in pairs}
        for future in futures:
            pair = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                    LOGGER.info("Successfully processed subject ID '%s'.", pair.subject_id)
            except Exception as e:
                LOGGER.error("Error processing subject ID '%s': %s", pair.subject_id, e)
                failure_count += 1

    LOGGER.info(
        "Processing complete. Successfully processed %d subjects, failed to process %d subjects.",
        success_count,
        failure_count,
    )
    return 0


if __name__ == "__main__":
    config = get_runtime_config()
    exit(main(config))
