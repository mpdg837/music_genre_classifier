from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_label_mapping(metadata: pd.DataFrame) -> Dict[str, int]:
    genres = sorted(metadata["genre"].unique())
    return {genre: idx for idx, genre in enumerate(genres)}


def load_note_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


class MidiPianoRollDataset(Dataset):
    def __init__(self, pianoroll_dir: Path, metadata_csv: Path):
        self.metadata = pd.read_csv(metadata_csv)
        self.pianoroll_dir = pianoroll_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        pr = np.load(self.pianoroll_dir / f"{row['sample_id']}.npz")
        x = torch.from_numpy(pr["pianoroll"]).float()
        y = torch.tensor([row["genre"]], dtype=torch.long)

        return x, y


class MidiNoteMatrixDataset(Dataset):
    def __init__(
        self,
        matrix_dir: Path,
        metadata_csv: Path,
        label_to_idx: Dict[str, int] | None = None,
        max_notes: int = 2048,
    ):
        self.metadata = pd.read_csv(metadata_csv)
        self.matrix_dir = matrix_dir
        self.label_to_idx = label_to_idx
        self.max_notes = max_notes

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        genre = row["genre"]

        arrays = load_note_arrays(self.matrix_dir / f"{sample_id}.npz")

        note_matrix = np.stack(
            [
                arrays["pitch"].astype(np.float32),
                arrays["onset_sec"].astype(np.float32),
                arrays["duration_sec"].astype(np.float32),
                arrays["velocity"].astype(np.float32),
                arrays["track"].astype(np.float32),
                arrays["channel"].astype(np.float32),
            ],
            axis=1,
        )

        n_notes = min(len(note_matrix), self.max_notes)

        padded = np.zeros((self.max_notes, note_matrix.shape[1]), dtype=np.float32)
        mask = np.zeros((self.max_notes,), dtype=np.bool_)

        if n_notes > 0:
            padded[:n_notes] = note_matrix[:n_notes]
            mask[:n_notes] = True

        x = torch.from_numpy(padded)
        mask = torch.from_numpy(mask)
        y = torch.tensor(self.label_to_idx[genre], dtype=torch.long)

        return {
            "x": x,
            "mask": mask,
            "y": y,
        }


def extract_note_features(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    pitch = arrays["pitch"]
    onset = arrays["onset_sec"]
    duration = arrays["duration_sec"]
    velocity = arrays["velocity"]

    features: Dict[str, float] = {}

    features["n_notes"] = float(len(pitch))
    features["pitch_mean"] = float(np.mean(pitch)) if len(pitch) else 0.0
    features["pitch_std"] = float(np.std(pitch)) if len(pitch) else 0.0
    features["pitch_min"] = float(np.min(pitch)) if len(pitch) else 0.0
    features["pitch_max"] = float(np.max(pitch)) if len(pitch) else 0.0

    features["duration_mean"] = float(np.mean(duration)) if len(duration) else 0.0
    features["duration_std"] = float(np.std(duration)) if len(duration) else 0.0
    features["duration_sum"] = float(np.sum(duration)) if len(duration) else 0.0

    features["velocity_mean"] = float(np.mean(velocity)) if len(velocity) else 0.0
    features["velocity_std"] = float(np.std(velocity)) if len(velocity) else 0.0

    if len(onset) > 1:
        onset_sorted = np.sort(onset)
        ioi = np.diff(onset_sorted)
        features["ioi_mean"] = float(np.mean(ioi))
        features["ioi_std"] = float(np.std(ioi))
    else:
        features["ioi_mean"] = 0.0
        features["ioi_std"] = 0.0

    pitch_class_hist = np.bincount(pitch % 12, minlength=12).astype(np.float32)
    if pitch_class_hist.sum() > 0:
        pitch_class_hist /= pitch_class_hist.sum()

    for i, value in enumerate(pitch_class_hist):
        features[f"pitch_class_{i}"] = float(value)

    return features


def build_sklearn_dataset(
    note_array_dir: Path,
    metadata_csv: Path,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
    metadata = pd.read_csv(metadata_csv)
    label_to_idx = build_label_mapping(metadata)

    rows = []
    y = []

    for row in metadata.itertuples(index=False):
        sample_id = row.sample_id
        genre = row.genre

        arrays = load_note_arrays(Path(note_array_dir) / f"{sample_id}.npz")
        features = extract_note_features(arrays)
        rows.append(features)
        y.append(label_to_idx[genre])

    X = pd.DataFrame(rows)
    y = np.asarray(y, dtype=np.int64)

    return X, y, label_to_idx
