from pathlib import Path

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import torch
from tqdm.auto import tqdm

from midi_xai.data.create_dataset import build_label_mapping
from midi_xai.data.musicbert_dataset import MidiMusicBertDataset
from midi_xai.models.neural.musicbert import MusicBertGenreClassifier


@torch.no_grad()
def extract_embeddings(
    model: MusicBertGenreClassifier,
    dataset: MidiMusicBertDataset,
    indices: np.ndarray,
    device: torch.device,
    stride: int,
    max_windows: int | None,
    window_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings = []
    targets = []
    batch_size = max(1, window_batch_size)

    for index in tqdm(indices, desc="Extracting MusicBERT embeddings"):
        item = dataset.get_windowed_item(
            idx=int(index),
            stride=stride,
            max_windows=max_windows,
        )
        window_embeddings = []
        for start in range(0, item["input_ids"].shape[0], batch_size):
            input_ids = item["input_ids"][start : start + batch_size].to(device)
            attention_mask = item["attention_mask"][start : start + batch_size].to(device)
            window_embeddings.append(model.encode(input_ids, attention_mask))

        pooled = torch.cat(window_embeddings, dim=0).mean(dim=0)
        embeddings.append(pooled.detach().cpu().numpy())
        targets.append(int(item["y"].item()))

    return np.vstack(embeddings), np.asarray(targets, dtype=np.int64)


@hydra.main(version_base=None, config_path="../configs", config_name="neural_config")
def main(cfg: DictConfig) -> None:
    if cfg.model.dataset.kind != "musicbert":
        raise ValueError("Run this script with model=musicbert")

    logger.info("Running MusicBERT embedding baseline with config:\n{}", OmegaConf.to_yaml(cfg))
    metadata_csv = Path(cfg.data.paths.labels_output_path)
    dataset = MidiMusicBertDataset(
        metadata_csv=metadata_csv,
        tokenizer_name_or_path=cfg.model.dataset.tokenizer_name_or_path,
        midi_root_dir=Path(cfg.data.paths.dataset_dir),
        max_length=cfg.model.dataset.max_length,
        random_crop=False,
        add_bos_eos=cfg.model.dataset.add_bos_eos,
        cache_token_ids=cfg.model.dataset.cache_token_ids,
    )
    label_to_idx = build_label_mapping(dataset.metadata)

    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=cfg.model.training.validation_size,
        random_state=cfg.seed,
        stratify=dataset.metadata["genre"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(
        cfg.model.instance,
        num_classes=len(label_to_idx),
        freeze_encoder=True,
        gradient_checkpointing=False,
    ).to(device)

    X_train, y_train = extract_embeddings(
        model=model,
        dataset=dataset,
        indices=train_indices,
        device=device,
        stride=cfg.model.dataset.eval_stride,
        max_windows=OmegaConf.select(cfg, "model.dataset.eval_max_windows"),
        window_batch_size=cfg.model.dataset.eval_window_batch_size,
    )
    X_val, y_val = extract_embeddings(
        model=model,
        dataset=dataset,
        indices=val_indices,
        device=device,
        stride=cfg.model.dataset.eval_stride,
        max_windows=OmegaConf.select(cfg, "model.dataset.eval_max_windows"),
        window_batch_size=cfg.model.dataset.eval_window_batch_size,
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.seed,
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)

    logger.info("Validation accuracy: {:.4f}", accuracy_score(y_val, predictions))
    logger.info(
        "Validation f1 macro: {:.4f}",
        f1_score(y_val, predictions, average="macro", zero_division=0),
    )
    logger.info(
        "Validation f1 weighted: {:.4f}",
        f1_score(y_val, predictions, average="weighted", zero_division=0),
    )
    logger.info("Validation classification report:\n{}", classification_report(y_val, predictions))


if __name__ == "__main__":
    main()
