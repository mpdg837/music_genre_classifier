from pathlib import Path
import zipfile

import gdown
from loguru import logger


def ensure_dataset(
    file_id: str,
    dataset_dir: str | Path,
    temp_zip_path: str | Path,
    force_download: bool = False,
    delete_zip: bool = True,
) -> Path:
    dataset_dir = Path(dataset_dir)

    if dataset_exists(dataset_dir) and not force_download:
        logger.info(f"Dataset already present in {dataset_dir}, skipping download and extraction.")
        return dataset_dir

    zip_path = download_google_drive_zip(
        file_id=file_id,
        output_path=temp_zip_path,
        force=force_download,
    )

    extract_zip_to_dir(
        zip_path=zip_path,
        dataset_dir=dataset_dir,
        delete_zip=delete_zip,
    )

    if not dataset_exists(dataset_dir):
        raise RuntimeError(f"Extraction completed, but no .midi files found in {dataset_dir}")

    logger.info(f"Dataset ready in {dataset_dir}")
    return dataset_dir


def download_google_drive_zip(
    file_id: str,
    output_path: str | Path,
    force: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        output_path.exists()
        and output_path.is_file()
        and output_path.stat().st_size > 0
        and not force
    ):
        logger.info(f"Archive already exists at {output_path}, skipping download.")
        return output_path

    logger.info(f"Downloading archive to {output_path}")
    gdown.download(id=file_id, output=str(output_path), quiet=False, fuzzy=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or produced an empty file: {output_path}")

    logger.info(f"Download finished: {output_path}")
    return output_path


def dataset_exists(dataset_dir: Path) -> bool:
    return dataset_dir.exists() and dataset_dir.is_dir() and any(dataset_dir.rglob("*.midi"))


def extract_zip_to_dir(
    zip_path: str | Path,
    dataset_dir: str | Path,
    delete_zip: bool = True,
) -> Path:
    zip_path = Path(zip_path)
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")

    logger.info(f"Extracting {zip_path} to {dataset_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    if delete_zip:
        logger.info(f"Removing archive {zip_path}")
        zip_path.unlink(missing_ok=True)

    return dataset_dir
