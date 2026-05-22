from pathlib import Path
import random
from typing import Any

from captum.concept import TCAV, Concept
from captum.concept._utils.common import concepts_to_str
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset

from midi_xai.data.create_dataset import MidiPianoRollDataset, build_label_mapping
from midi_xai.interpretability.tcav.concepts import (
    ConceptPianoRollDataset,
    ConceptSpec,
    load_concept_specs,
)
from midi_xai.interpretability.tcav.core import (
    SklearnLinearSVCClassifier,
    compute_captum_tcav_score,
    load_model_checkpoint,
)
from midi_xai.interpretability.tcav.reports import sanitize_name, write_json, write_score_tables


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_captum_concept(
    spec: ConceptSpec,
    cfg: DictConfig,
    note_array_dir: Path,
    device: torch.device,
) -> Concept:
    if not spec.manifest_path.exists():
        raise FileNotFoundError(f"Concept manifest not found: {spec.manifest_path}")

    dataset = ConceptPianoRollDataset(
        manifest_csv=spec.manifest_path,
        note_array_dir=note_array_dir,
        frame_rate=cfg.model.dataset.frame_rate,
        max_time_steps=cfg.model.dataset.max_time_steps,
        pitch_min=cfg.model.dataset.pitch_min,
        n_pitches=cfg.model.dataset.n_pitches,
    )

    def collate_to_device(items: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(items, dim=0).to(device)

    data_iter = DataLoader(
        dataset,
        batch_size=cfg.tcav.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_to_device,
    )
    return Concept(id=spec.id, name=spec.name, data_iter=data_iter)


def load_specs_from_config(cfg: DictConfig) -> tuple[list[ConceptSpec], list[ConceptSpec]]:
    concepts = load_concept_specs(
        definitions=OmegaConf.select(cfg, "tcav.concepts.definitions", default=[]),
        manifest_paths=OmegaConf.select(cfg, "tcav.concepts.manifest_paths", default=[]),
        manifest_dir=OmegaConf.select(cfg, "tcav.concepts.manifest_dir"),
        start_id=0,
    )
    random_controls = load_concept_specs(
        definitions=OmegaConf.select(cfg, "tcav.random_controls.definitions", default=[]),
        manifest_paths=OmegaConf.select(cfg, "tcav.random_controls.manifest_paths", default=[]),
        manifest_dir=OmegaConf.select(cfg, "tcav.random_controls.output_dir"),
        start_id=10_000,
    )

    if not concepts:
        raise ValueError(
            "No TCAV concept manifests configured. Add CSV files to tcav.concepts.manifest_dir "
            "or run scripts/prepare_tcav_concepts.py first."
        )
    if not random_controls:
        raise ValueError(
            "No random control manifests found. Run scripts/prepare_tcav_controls.py first "
            "or configure tcav.random_controls.manifest_paths."
        )
    return concepts, random_controls


def parse_target_classes(target_classes: Any, label_to_idx: dict[str, int]) -> list[int]:
    if target_classes == "all":
        return sorted(label_to_idx.values())

    parsed = []
    for target in target_classes:
        if isinstance(target, int):
            parsed.append(target)
        elif str(target).isdigit():
            parsed.append(int(target))
        elif str(target) in label_to_idx:
            parsed.append(label_to_idx[str(target)])
        else:
            raise ValueError(f"Unknown target class: {target}")
    return parsed


def select_target_indices(
    metadata,
    val_indices: np.ndarray,
    label_to_idx: dict[str, int],
    target_class: int,
    max_examples: int | None,
    seed: int,
) -> list[int]:
    selected = [
        int(index)
        for index in val_indices
        if label_to_idx[metadata.iloc[int(index)]["genre"]] == target_class
    ]
    if max_examples is not None and len(selected) > max_examples:
        rng = np.random.default_rng(seed + target_class)
        selected = sorted(rng.choice(selected, size=max_examples, replace=False).tolist())
    return selected


@hydra.main(version_base=None, config_path="../configs", config_name="tcav_config")
def main(cfg: DictConfig) -> None:
    logger.info("Running MuseResNet TCAV with config:\n{}", OmegaConf.to_yaml(cfg))
    if cfg.model.dataset.kind != "pianoroll":
        raise ValueError("scripts/run_tcav_muserenet.py currently supports pianoroll models only.")

    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    logger.info("Using device: {}", device)

    metadata_csv = Path(cfg.data.paths.labels_output_path)
    note_array_dir = Path(cfg.data.paths.processed_dir)
    dataset = MidiPianoRollDataset(
        note_array_dir=note_array_dir,
        metadata_csv=metadata_csv,
        frame_rate=cfg.model.dataset.frame_rate,
        max_time_steps=cfg.model.dataset.max_time_steps,
        pitch_min=cfg.model.dataset.pitch_min,
        n_pitches=cfg.model.dataset.n_pitches,
        random_crop=False,
    )
    label_to_idx = build_label_mapping(dataset.metadata)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    indices = np.arange(len(dataset))
    _, val_indices = train_test_split(
        indices,
        test_size=cfg.model.training.validation_size,
        random_state=cfg.seed,
        stratify=dataset.metadata["genre"],
    )

    model = hydra.utils.instantiate(cfg.model.instance, num_classes=len(label_to_idx)).to(device)
    checkpoint = load_model_checkpoint(
        model=model,
        checkpoint_path=Path(cfg.tcav.checkpoint_path),
        label_to_idx=label_to_idx,
        device=device,
        strict=bool(OmegaConf.select(cfg, "tcav.strict_load_weights", default=True)),
    )
    model.eval()

    concepts, random_controls = load_specs_from_config(cfg)
    target_classes = parse_target_classes(cfg.tcav.target_classes, label_to_idx)
    artifacts_dir = Path(cfg.tcav.artifacts_dir)
    write_json(
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "label_to_idx": label_to_idx,
            "checkpoint_best_val_f1_macro": checkpoint.get("best_val_f1_macro"),
        },
        artifacts_dir / "run_metadata.json",
    )

    captum_concepts = {
        spec.manifest_path: build_captum_concept(
            spec=spec,
            cfg=cfg,
            note_array_dir=note_array_dir,
            device=device,
        )
        for spec in [*concepts, *random_controls]
    }
    tcav = TCAV(
        model=model,
        layers=list(cfg.tcav.layers),
        model_id=sanitize_name(cfg.tcav.name),
        classifier=SklearnLinearSVCClassifier(),
        save_path=str(artifacts_dir / "captum"),
        test_split_ratio=float(cfg.tcav.cav.test_size),
        c=float(cfg.tcav.cav.c),
        max_iter=int(cfg.tcav.cav.max_iter),
        seed=int(cfg.seed),
    )

    rows: list[dict[str, Any]] = []
    max_examples = OmegaConf.select(cfg, "tcav.evaluation.max_examples_per_class")
    max_examples = None if max_examples in (None, "null") else int(max_examples)

    for layer in cfg.tcav.layers:
        logger.info("Processing TCAV layer: {}", layer)
        for concept in concepts:
            for random_control in random_controls:
                experimental_set = [
                    captum_concepts[concept.manifest_path],
                    captum_concepts[random_control.manifest_path],
                ]

                for target_class in target_classes:
                    target_indices = select_target_indices(
                        metadata=dataset.metadata,
                        val_indices=val_indices,
                        label_to_idx=label_to_idx,
                        target_class=target_class,
                        max_examples=max_examples,
                        seed=int(cfg.seed),
                    )
                    if not target_indices:
                        logger.warning(
                            "No validation examples for target class {}",
                            idx_to_label[target_class],
                        )
                        continue

                    target_loader = DataLoader(
                        Subset(dataset, target_indices),
                        batch_size=cfg.tcav.batch_size,
                        shuffle=False,
                        num_workers=cfg.tcav.num_workers,
                        pin_memory=pin_memory,
                    )
                    score = compute_captum_tcav_score(
                        tcav=tcav,
                        dataloader=target_loader,
                        experimental_set=experimental_set,
                        layer_name=layer,
                        target_class=target_class,
                        device=device,
                    )
                    cav_stats = tcav.cavs[concepts_to_str(experimental_set)][layer].stats or {}
                    cav_accuracy = cav_stats.get("accs")
                    if torch.is_tensor(cav_accuracy):
                        cav_accuracy = float(cav_accuracy.detach().cpu().item())
                    rows.append(
                        {
                            "concept": concept.name,
                            "random_control": random_control.name,
                            "layer": layer,
                            "target_class": idx_to_label[target_class],
                            "target_class_idx": target_class,
                            "sign_count": score.sign_count,
                            "magnitude": score.magnitude,
                            "n_examples": score.n_examples,
                            "cav_accuracy": cav_accuracy,
                            "n_concept": len(captum_concepts[concept.manifest_path].data_iter.dataset),
                            "n_random": len(
                                captum_concepts[random_control.manifest_path].data_iter.dataset
                            ),
                        }
                    )

    raw_path, summary_path = write_score_tables(
        rows=rows,
        output_dir=artifacts_dir / "scores",
        alpha=float(cfg.tcav.significance.alpha),
    )
    logger.info("Wrote TCAV raw scores to {}", raw_path)
    logger.info("Wrote TCAV summary to {}", summary_path)


if __name__ == "__main__":
    main()
