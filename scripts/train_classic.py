from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import wandb

from midi_xai.data.create_dataset import build_sklearn_dataset
from midi_xai.models.classic_model import log_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(
        "Running classic model training script with config:\n{}", OmegaConf.to_yaml(cfg)
    )

    model = hydra.utils.instantiate(cfg.model)

    metadata_csv = Path(cfg.data.paths.labels_output_path)
    note_array_dir = Path(cfg.data.paths.processed_dir)

    X, y, _ = build_sklearn_dataset(
        note_array_dir=note_array_dir, metadata_csv=metadata_csv
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,
    )

    with wandb.init(
        project="music-genre-xai",
        config={
            "model_name": cfg.model.model._target_,
        },
    ):

        logger.info("Training the model...")
        model.fit(X_train, y_train)

        logger.info("Evaluating train metrics.")
        log_metrics(model, X_train, y_train, "train")
        logger.info("Evaluating validation metrics.")
        log_metrics(model, X_val, y_val, "val")


if __name__ == "__main__":
    main()
