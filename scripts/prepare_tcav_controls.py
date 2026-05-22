from pathlib import Path

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split

from midi_xai.interpretability.tcav.concepts import write_random_control_manifests


def select_candidate_indices(cfg: DictConfig, metadata: pd.DataFrame) -> np.ndarray | None:
    split = str(OmegaConf.select(cfg, "tcav.random_controls.source_split", default="train"))
    if split == "all":
        return None

    indices = np.arange(len(metadata))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=cfg.model.training.validation_size,
        random_state=cfg.seed,
        stratify=metadata["genre"],
    )
    if split == "train":
        return train_indices
    if split in ("validation", "val"):
        return val_indices
    raise ValueError(f"Unknown random control source_split: {split}")


@hydra.main(version_base=None, config_path="../configs", config_name="tcav_config")
def main(cfg: DictConfig) -> None:
    logger.info("Preparing TCAV random controls with config:\n{}", OmegaConf.to_yaml(cfg))

    metadata_csv = Path(cfg.data.paths.labels_output_path)
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_csv}. Run scripts/prepare_data.py first."
        )

    metadata = pd.read_csv(metadata_csv)
    output_paths = write_random_control_manifests(
        metadata=metadata,
        output_dir=Path(cfg.tcav.random_controls.output_dir),
        n_controls=int(cfg.tcav.random_controls.n_controls),
        samples_per_control=int(cfg.tcav.random_controls.samples_per_control),
        seed=int(cfg.seed),
        candidate_indices=select_candidate_indices(cfg, metadata),
    )

    for output_path in output_paths:
        logger.info("Wrote random control manifest: {}", output_path)


if __name__ == "__main__":
    main()
