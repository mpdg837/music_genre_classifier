from pathlib import Path

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split

from midi_xai.interpretability.tcav.concepts import write_feature_concept_manifests


def select_candidate_indices(cfg: DictConfig, metadata: pd.DataFrame) -> np.ndarray | None:
    split = str(OmegaConf.select(cfg, "tcav.concepts.source_split", default="train"))
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
    raise ValueError(f"Unknown concept source_split: {split}")


@hydra.main(version_base=None, config_path="../configs", config_name="tcav_config")
def main(cfg: DictConfig) -> None:
    logger.info("Preparing TCAV concept manifests with config:\n{}", OmegaConf.to_yaml(cfg))

    metadata_csv = Path(cfg.data.paths.labels_output_path)
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_csv}. Run scripts/prepare_data.py first."
        )

    concept_definitions = OmegaConf.select(cfg, "tcav.concepts.feature_concepts", default=[])
    if not concept_definitions:
        logger.info("No feature concepts configured; skipping concept manifest generation.")
        return

    metadata = pd.read_csv(metadata_csv)
    output_paths = write_feature_concept_manifests(
        metadata=metadata,
        note_array_dir=Path(cfg.data.paths.processed_dir),
        output_dir=Path(cfg.tcav.concepts.manifest_dir),
        concept_definitions=concept_definitions,
        samples_per_concept=int(cfg.tcav.concepts.samples_per_concept),
        seed=int(cfg.seed),
        candidate_indices=select_candidate_indices(cfg, metadata),
    )

    for output_path in output_paths:
        logger.info("Wrote concept manifest: {}", output_path)


if __name__ == "__main__":
    main()
