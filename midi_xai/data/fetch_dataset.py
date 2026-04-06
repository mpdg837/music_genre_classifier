from pathlib import Path
import re
import zipfile

from loguru import logger
import requests

GOOGLE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download"


def ensure_dataset(
    url: str,
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
        url=url,
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


def download_google_drive_zip(url: str, output_path: str | Path, chunk_size: int = 32768) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_id = _extract_google_drive_file_id(url)

    session = requests.Session()
    base_url = GOOGLE_DOWNLOAD_URL

    response = session.get(base_url, params={"id": file_id}, stream=True)
    response.raise_for_status()

    token = _get_confirm_token(response)
    if token:
        response.close()
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": token},
            stream=True,
        )
        response.raise_for_status()

    _save_response_content(response, output_path, chunk_size=chunk_size)

    if output_path.suffix.lower() != ".zip":
        raise ValueError(f"Downloaded file does not end with .zip: {output_path}")

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


def _extract_google_drive_file_id(url: str) -> str | None:
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/uc\?export=download&id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
    return match.group(1) if match else None


def _save_response_content(
    response: requests.Response, destination: Path, chunk_size: int
) -> None:
    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
