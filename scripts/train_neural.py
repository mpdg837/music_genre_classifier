from pathlib import Path
import random
from typing import Dict

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import wandb


from midi_xai.data.create_dataset import (
    MidiNoteMatrixDataset,
    MidiPianoRollDataset,
    build_label_mapping,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward_model(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "mask" in batch:
        return model(batch["x"], batch["mask"])
    return model(batch["x"])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    losses = []
    all_targets = []
    all_predictions = []

    for batch in dataloader:
        batch = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch)
        loss = criterion(logits, batch["y"])
        loss.backward()

        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        losses.append(loss.item())
        all_targets.extend(batch["y"].detach().cpu().numpy())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())

    return compute_metrics(losses, all_targets, all_predictions)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float | str]:
    model.eval()
    losses = []
    all_targets = []
    all_predictions = []

    for batch in dataloader:
        batch = batch_to_device(batch, device)
        logits = forward_model(model, batch)
        loss = criterion(logits, batch["y"])

        losses.append(loss.item())
        all_targets.extend(batch["y"].detach().cpu().numpy())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())

    metrics = compute_metrics(losses, all_targets, all_predictions)
    metrics["report"] = classification_report(
        all_targets,
        all_predictions,
        zero_division=0,
    )
    return metrics


def compute_metrics(
    losses: list[float],
    targets: list[int],
    predictions: list[int],
) -> Dict[str, float]:
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(targets, predictions),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def build_datasets(
    cfg: DictConfig,
    note_array_dir: Path,
    metadata_csv: Path,
) -> tuple[Dataset, Dataset, Dataset, Dict[str, int]]:
    dataset_kind = cfg.model.dataset.kind

    if dataset_kind == "note_matrix":
        dataset = MidiNoteMatrixDataset(
            matrix_dir=note_array_dir,
            metadata_csv=metadata_csv,
            max_notes=cfg.model.dataset.max_notes,
            normalize=True,
        )
        label_to_idx = build_label_mapping(dataset.metadata)
        return dataset, dataset, dataset, label_to_idx

    if dataset_kind == "pianoroll":
        metadata_dataset = MidiPianoRollDataset(
            note_array_dir=note_array_dir,
            metadata_csv=metadata_csv,
            frame_rate=cfg.model.dataset.frame_rate,
            max_time_steps=cfg.model.dataset.max_time_steps,
            pitch_min=cfg.model.dataset.pitch_min,
            n_pitches=cfg.model.dataset.n_pitches,
        )
        label_to_idx = build_label_mapping(metadata_dataset.metadata)

        train_dataset = MidiPianoRollDataset(
            note_array_dir=note_array_dir,
            metadata_csv=metadata_csv,
            label_to_idx=label_to_idx,
            frame_rate=cfg.model.dataset.frame_rate,
            max_time_steps=cfg.model.dataset.max_time_steps,
            pitch_min=cfg.model.dataset.pitch_min,
            n_pitches=cfg.model.dataset.n_pitches,
            random_crop=True,
        )
        val_dataset = MidiPianoRollDataset(
            note_array_dir=note_array_dir,
            metadata_csv=metadata_csv,
            label_to_idx=label_to_idx,
            frame_rate=cfg.model.dataset.frame_rate,
            max_time_steps=cfg.model.dataset.max_time_steps,
            pitch_min=cfg.model.dataset.pitch_min,
            n_pitches=cfg.model.dataset.n_pitches,
            random_crop=False,
        )
        return metadata_dataset, train_dataset, val_dataset, label_to_idx

    raise ValueError(f"Unknown neural dataset kind: {dataset_kind}")


@hydra.main(version_base=None, config_path="../configs", config_name="neural_config")
def main(cfg: DictConfig) -> None:
    logger.info("Running neural training with config:\n{}", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    metadata_csv = Path(cfg.data.paths.labels_output_path)
    note_array_dir = Path(cfg.data.paths.processed_dir)

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_csv}. Run scripts/prepare_data.py first."
        )

    metadata_dataset, train_dataset, val_dataset, label_to_idx = build_datasets(
        cfg=cfg,
        note_array_dir=note_array_dir,
        metadata_csv=metadata_csv,
    )

    indices = np.arange(len(metadata_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=cfg.model.training.validation_size,
        random_state=cfg.seed,
        stratify=metadata_dataset.metadata["genre"],
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=cfg.model.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.model.dataset.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=cfg.model.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.model.dataset.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    model = hydra.utils.instantiate(
        cfg.model.instance,
        num_classes=len(label_to_idx),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
    )

    save_dir = Path(cfg.save_weights_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg.model.name}.pt"
    best_f1 = -1.0

    with wandb.init(
        project="music-genre-xai",
        config=OmegaConf.to_container(cfg, resolve=True),
    ):
        for epoch in range(1, cfg.model.training.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                gradient_clip_norm=cfg.model.training.gradient_clip_norm,
            )
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            wandb.log(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{
                        f"val_{key}": value
                        for key, value in val_metrics.items()
                        if key != "report"
                    },
                }
            )

            logger.info(
                "Epoch {}/{} | train loss {:.4f}, f1 {:.4f} | val loss {:.4f}, f1 {:.4f}",
                epoch,
                cfg.model.training.epochs,
                train_metrics["loss"],
                train_metrics["f1_macro"],
                val_metrics["loss"],
                val_metrics["f1_macro"],
            )

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "label_to_idx": label_to_idx,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "best_val_f1_macro": best_f1,
                    },
                    save_path,
                )
                logger.info("Saved new best checkpoint to {}", save_path)

        logger.info("Validation classification report:\n{}", val_metrics["report"])


if __name__ == "__main__":
    main()
