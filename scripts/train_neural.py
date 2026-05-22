import math
from pathlib import Path
import random
from typing import Dict

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import wandb

from midi_xai.data.create_dataset import (
    MidiNoteMatrixDataset,
    MidiPianoRollDataset,
    build_label_mapping,
)
from midi_xai.data.musicbert_dataset import MidiMusicBertDataset


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
    if "input_ids" in batch:
        return model(batch["input_ids"], batch["attention_mask"])
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
    gradient_accumulation_steps: int = 1,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Dict[str, float]:
    model.train()
    losses = []
    all_targets = []
    all_predictions = []
    accumulation_steps = max(1, gradient_accumulation_steps)

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(dataloader, start=1):
        batch = batch_to_device(batch, device)

        logits = forward_model(model, batch)
        loss = criterion(logits, batch["y"])
        (loss / accumulation_steps).backward()

        should_step = step % accumulation_steps == 0 or step == len(dataloader)
        if should_step:
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())
        all_targets.extend(batch["y"].detach().cpu().numpy())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())

    metrics = compute_metrics(losses, all_targets, all_predictions)
    metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
    return metrics


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


@torch.no_grad()
def evaluate_musicbert_windows(
    model: nn.Module,
    dataset: MidiMusicBertDataset,
    indices: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    stride: int,
    max_windows: int | None,
    window_batch_size: int,
) -> Dict[str, float | str]:
    model.eval()
    losses = []
    all_targets = []
    all_predictions = []
    batch_size = max(1, window_batch_size)

    for index in indices:
        item = dataset.get_windowed_item(
            idx=int(index),
            stride=stride,
            max_windows=max_windows,
        )
        y = item["y"].to(device).unsqueeze(0)
        window_logits = []

        for start in range(0, item["input_ids"].shape[0], batch_size):
            input_ids = item["input_ids"][start : start + batch_size].to(device)
            attention_mask = item["attention_mask"][start : start + batch_size].to(device)
            window_logits.append(model(input_ids, attention_mask))

        logits = torch.cat(window_logits, dim=0).mean(dim=0, keepdim=True)
        loss = criterion(logits, y)

        losses.append(loss.item())
        all_targets.append(int(y.detach().cpu().item()))
        all_predictions.append(int(logits.argmax(dim=1).detach().cpu().item()))

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

    if dataset_kind == "musicbert":
        train_dataset = MidiMusicBertDataset(
            metadata_csv=metadata_csv,
            tokenizer_name_or_path=cfg.model.dataset.tokenizer_name_or_path,
            midi_root_dir=Path(cfg.data.paths.dataset_dir),
            max_length=cfg.model.dataset.max_length,
            random_crop=True,
            add_bos_eos=cfg.model.dataset.add_bos_eos,
            cache_token_ids=cfg.model.dataset.cache_token_ids,
        )
        label_to_idx = build_label_mapping(train_dataset.metadata)
        val_dataset = MidiMusicBertDataset(
            metadata_csv=metadata_csv,
            tokenizer_name_or_path=cfg.model.dataset.tokenizer_name_or_path,
            label_to_idx=label_to_idx,
            midi_root_dir=Path(cfg.data.paths.dataset_dir),
            max_length=cfg.model.dataset.max_length,
            random_crop=False,
            add_bos_eos=cfg.model.dataset.add_bos_eos,
            cache_token_ids=cfg.model.dataset.cache_token_ids,
        )
        return train_dataset, train_dataset, val_dataset, label_to_idx

    raise ValueError(f"Unknown neural dataset kind: {dataset_kind}")


def build_criterion(
    cfg: DictConfig,
    metadata,
    train_indices: np.ndarray,
    label_to_idx: Dict[str, int],
    device: torch.device,
) -> nn.Module:
    class_weighting = OmegaConf.select(cfg, "model.training.class_weighting", default="none")
    if class_weighting in (None, False, "none"):
        return nn.CrossEntropyLoss()

    if class_weighting != "balanced":
        raise ValueError(f"Unknown class_weighting mode: {class_weighting}")

    train_targets = metadata.iloc[train_indices]["genre"].map(label_to_idx).to_numpy()
    classes = np.arange(len(label_to_idx))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_targets)
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
    )


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "set_encoder_trainable"):
        model.set_encoder_trainable(trainable)


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    learning_rate = cfg.model.training.learning_rate
    weight_decay = cfg.model.training.weight_decay
    encoder_lr = OmegaConf.select(cfg, "model.training.encoder_learning_rate")
    classifier_lr = OmegaConf.select(cfg, "model.training.classifier_learning_rate")

    if encoder_lr is None and classifier_lr is None:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    encoder_lr = encoder_lr if encoder_lr is not None else learning_rate
    classifier_lr = classifier_lr if classifier_lr is not None else learning_rate
    encoder_params = []
    classifier_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(parameter)
        else:
            classifier_params.append(parameter)

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    scheduler_kind = OmegaConf.select(cfg, "model.training.scheduler.kind", default="none")
    if scheduler_kind in (None, "none", False):
        return None

    total_steps = int(
        OmegaConf.select(
            cfg,
            "model.training.scheduler.total_steps",
            default=steps_per_epoch * cfg.model.training.epochs,
        )
    )
    warmup_steps = OmegaConf.select(cfg, "model.training.scheduler.warmup_steps")
    if warmup_steps is None:
        warmup_ratio = float(
            OmegaConf.select(cfg, "model.training.scheduler.warmup_ratio", default=0.0)
        )
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = int(warmup_steps)
    min_lr_ratio = float(
        OmegaConf.select(cfg, "model.training.scheduler.min_lr_ratio", default=0.0)
    )

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(min_lr_ratio, float(current_step + 1) / float(warmup_steps))

        progress_denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / progress_denominator)
        if scheduler_kind == "linear_warmup":
            return max(min_lr_ratio, 1.0 - progress)
        if scheduler_kind == "cosine_warmup":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        raise ValueError(f"Unknown scheduler kind: {scheduler_kind}")

    logger.info(
        "Using {} scheduler with {} total step(s), {} warmup step(s), min_lr_ratio={}",
        scheduler_kind,
        total_steps,
        warmup_steps,
        min_lr_ratio,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_checkpoint_weights(
    checkpoint_path: Path,
    model: nn.Module,
    label_to_idx: Dict[str, int],
    device: torch.device,
    strict: bool = True,
) -> float:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_label_to_idx = checkpoint.get("label_to_idx")
        if checkpoint_label_to_idx is not None and dict(checkpoint_label_to_idx) != label_to_idx:
            raise ValueError("Checkpoint label mapping does not match the current dataset labels.")

        state_dict = checkpoint["model_state_dict"]
        best_f1 = float(checkpoint.get("best_val_f1_macro", -1.0))
    else:
        state_dict = checkpoint
        best_f1 = -1.0

    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys:
        logger.warning("Missing checkpoint keys: {}", load_result.missing_keys)
    if load_result.unexpected_keys:
        logger.warning("Unexpected checkpoint keys: {}", load_result.unexpected_keys)

    logger.info("Loaded model weights from {}", checkpoint_path)
    return best_f1


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
    val_loader = None
    if cfg.model.dataset.kind != "musicbert":
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
    train_encoder = bool(OmegaConf.select(cfg, "model.training.train_encoder", default=True))
    if not train_encoder:
        set_encoder_trainable(model, False)
        logger.info("Keeping encoder frozen for the full run")

    criterion = build_criterion(
        cfg=cfg,
        metadata=metadata_dataset.metadata,
        train_indices=train_indices,
        label_to_idx=label_to_idx,
        device=device,
    )
    optimizer = build_optimizer(model=model, cfg=cfg)
    gradient_accumulation_steps = int(
        OmegaConf.select(cfg, "model.training.gradient_accumulation_steps", default=1)
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, gradient_accumulation_steps))
    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        steps_per_epoch=steps_per_epoch,
    )
    freeze_encoder_epochs = int(
        OmegaConf.select(cfg, "model.training.freeze_encoder_epochs", default=0)
    )
    if train_encoder and freeze_encoder_epochs > 0:
        set_encoder_trainable(model, False)
        logger.info("Freezing encoder for the first {} epoch(s)", freeze_encoder_epochs)

    save_dir = Path(cfg.save_weights_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg.model.name}.pt"
    best_f1 = -1.0
    load_weights_path = OmegaConf.select(cfg, "load_weights_path")
    if load_weights_path:
        best_f1 = load_checkpoint_weights(
            checkpoint_path=Path(load_weights_path),
            model=model,
            label_to_idx=label_to_idx,
            device=device,
            strict=bool(OmegaConf.select(cfg, "strict_load_weights", default=True)),
        )
        if best_f1 >= 0.0:
            logger.info("Resuming best validation F1 from checkpoint: {:.4f}", best_f1)

    with wandb.init(
        project="music-genre-xai",
        config=OmegaConf.to_container(cfg, resolve=True),
    ):
        for epoch in range(1, cfg.model.training.epochs + 1):
            if train_encoder and freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs + 1:
                set_encoder_trainable(model, True)
                logger.info("Unfroze encoder at epoch {}", epoch)

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                gradient_clip_norm=cfg.model.training.gradient_clip_norm,
                gradient_accumulation_steps=gradient_accumulation_steps,
                scheduler=scheduler,
            )
            if cfg.model.dataset.kind == "musicbert":
                val_metrics = evaluate_musicbert_windows(
                    model=model,
                    dataset=val_dataset,
                    indices=val_indices,
                    criterion=criterion,
                    device=device,
                    stride=cfg.model.dataset.eval_stride,
                    max_windows=OmegaConf.select(cfg, "model.dataset.eval_max_windows"),
                    window_batch_size=cfg.model.dataset.eval_window_batch_size,
                )
            else:
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
