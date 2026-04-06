from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from midi_xai.data.fetch_dataset import ensure_dataset
from midi_xai.data.preprocess import preprocess_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Running data preparation with config:\n{}", OmegaConf.to_yaml(cfg))

    dataset_dir = Path(cfg.data.paths.dataset_dir)
    processed_dir = Path(cfg.data.paths.processed_dir)
    labels_output_path = Path(cfg.data.paths.labels_output_path)
    temp_zip_path = Path(cfg.data.paths.temp_zip_path)

    dataset_dir = ensure_dataset(
        file_id=cfg.data.download.file_id,
        dataset_dir=dataset_dir,
        temp_zip_path=temp_zip_path,
        force_download=cfg.data.download.force,
        delete_zip=cfg.data.download.delete_zip,
    )

    logger.info("Preprocessing dataset at: {}", dataset_dir)
    labels_df = preprocess_dataset(dataset_dir, processed_dir)

    labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(labels_output_path, index=False)

    logger.info("Labels saved to: {}", labels_output_path)
    logger.info("Processed dataset saved to: {}", processed_dir)


if __name__ == "__main__":
    main()