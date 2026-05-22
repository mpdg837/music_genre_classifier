from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import Dataset

from midi_xai.data.create_dataset import (
    build_pianoroll_from_note_arrays,
    extract_note_features,
    load_note_arrays,
)


@dataclass(frozen=True)
class ConceptSpec:
    id: int
    name: str
    manifest_path: Path


class ConceptPianoRollDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        note_array_dir: Path,
        frame_rate: float = 20.0,
        max_time_steps: int = 1024,
        pitch_min: int = 0,
        n_pitches: int = 128,
    ):
        self.manifest_csv = manifest_csv
        self.manifest = pd.read_csv(manifest_csv)
        if "sample_id" not in self.manifest.columns:
            raise ValueError(f"Concept manifest must contain a sample_id column: {manifest_csv}")

        self.note_array_dir = note_array_dir
        self.frame_rate = frame_rate
        self.max_time_steps = max_time_steps
        self.pitch_min = pitch_min
        self.n_pitches = n_pitches

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.manifest.iloc[idx]
        arrays = load_note_arrays(self.note_array_dir / f"{row['sample_id']}.npz")
        pianoroll = build_pianoroll_from_note_arrays(
            arrays=arrays,
            frame_rate=self.frame_rate,
            max_time_steps=self.max_time_steps,
            pitch_min=self.pitch_min,
            n_pitches=self.n_pitches,
            random_crop=False,
        )
        return torch.from_numpy(pianoroll).float()


def load_concept_specs(
    definitions: Sequence[DictConfig] | ListConfig | None = None,
    manifest_paths: Sequence[str | Path] | ListConfig | None = None,
    manifest_dir: str | Path | None = None,
    start_id: int = 0,
) -> list[ConceptSpec]:
    specs: list[ConceptSpec] = []

    if definitions:
        for offset, entry in enumerate(definitions):
            container = OmegaConf.to_container(entry, resolve=True)
            if not isinstance(container, dict):
                raise TypeError(f"Concept definition must be a mapping, got: {type(entry)}")
            manifest_path = Path(str(container["manifest_path"]))
            name = str(container.get("name") or manifest_path.stem)
            concept_id = int(container.get("id", start_id + offset))
            specs.append(ConceptSpec(id=concept_id, name=name, manifest_path=manifest_path))

    if manifest_paths:
        next_id = start_id + len(specs)
        specs.extend(
            ConceptSpec(id=next_id + offset, name=Path(path).stem, manifest_path=Path(path))
            for offset, path in enumerate(manifest_paths)
        )

    if manifest_dir:
        next_id = start_id + len(specs)
        discovered = sorted(Path(manifest_dir).glob("*.csv"))
        specs.extend(
            ConceptSpec(id=next_id + offset, name=path.stem, manifest_path=path)
            for offset, path in enumerate(discovered)
        )

    return _deduplicate_specs(specs)


def _deduplicate_specs(specs: Iterable[ConceptSpec]) -> list[ConceptSpec]:
    seen_paths: set[Path] = set()
    unique_specs: list[ConceptSpec] = []
    for spec in specs:
        resolved = spec.manifest_path.expanduser()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_specs.append(
            ConceptSpec(
                id=spec.id,
                name=spec.name,
                manifest_path=resolved,
            )
        )
    return unique_specs


def write_random_control_manifests(
    metadata: pd.DataFrame,
    output_dir: Path,
    n_controls: int,
    samples_per_control: int,
    seed: int,
    candidate_indices: Sequence[int] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_metadata = metadata
    if candidate_indices is not None:
        candidate_metadata = metadata.iloc[list(candidate_indices)]

    if samples_per_control > len(candidate_metadata):
        raise ValueError(
            "samples_per_control cannot be larger than the available candidate metadata rows."
        )

    output_paths = []
    for control_idx in range(n_controls):
        random_state = seed + control_idx
        sample = candidate_metadata.sample(
            n=samples_per_control,
            replace=False,
            random_state=random_state,
        )
        keep_columns = [column for column in ("sample_id", "genre", "emotion") if column in sample]
        output_path = output_dir / f"random_control_{control_idx:02d}.csv"
        sample[keep_columns].to_csv(output_path, index=False)
        output_paths.append(output_path)

    return output_paths


def write_feature_concept_manifests(
    metadata: pd.DataFrame,
    note_array_dir: Path,
    output_dir: Path,
    concept_definitions: Sequence[DictConfig] | ListConfig,
    samples_per_concept: int,
    seed: int,
    candidate_indices: Sequence[int] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_metadata = metadata
    if candidate_indices is not None:
        candidate_metadata = metadata.iloc[list(candidate_indices)]

    feature_metadata = _build_feature_metadata(candidate_metadata, note_array_dir)
    output_paths = []
    for definition in concept_definitions:
        container = OmegaConf.to_container(definition, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"Concept definition must be a mapping, got: {type(definition)}")

        feature = str(container["feature"])
        if feature not in feature_metadata:
            raise ValueError(f"Unknown concept feature '{feature}'.")

        name = str(container.get("name") or _default_concept_name(container))
        tail = str(container.get("tail", "high"))
        n_samples = int(container.get("samples", samples_per_concept))
        ascending = _tail_to_sort_order(tail)

        sample = (
            feature_metadata.sample(frac=1.0, random_state=seed)
            .sort_values(feature, ascending=ascending)
            .head(n_samples)
            .copy()
        )
        sample["concept_name"] = name
        sample["feature"] = feature
        sample["feature_value"] = sample[feature]
        sample["tail"] = tail
        sample["rank"] = range(1, len(sample) + 1)

        keep_columns = [
            column
            for column in (
                "sample_id",
                "genre",
                "emotion",
                "concept_name",
                "feature",
                "feature_value",
                "tail",
                "rank",
            )
            if column in sample
        ]
        output_path = output_dir / f"{_safe_manifest_name(name)}.csv"
        sample[keep_columns].to_csv(output_path, index=False)
        output_paths.append(output_path)

    return output_paths


def _build_feature_metadata(metadata: pd.DataFrame, note_array_dir: Path) -> pd.DataFrame:
    rows = []
    for row in metadata.itertuples(index=False):
        sample_id = getattr(row, "sample_id")
        arrays = load_note_arrays(note_array_dir / f"{sample_id}.npz")
        rows.append(
            {
                **row._asdict(),
                **extract_note_features(arrays),
            }
        )
    return pd.DataFrame(rows)


def _tail_to_sort_order(tail: str) -> bool:
    if tail == "low":
        return True
    if tail == "high":
        return False
    raise ValueError(f"Unknown concept tail '{tail}'. Use 'high' or 'low'.")


def _default_concept_name(definition: dict) -> str:
    return f"{definition.get('tail', 'high')}_{definition['feature']}"


def _safe_manifest_name(name: str) -> str:
    return (
        name.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
