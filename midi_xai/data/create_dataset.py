from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def build_label_mapping(metadata: pd.DataFrame) -> Dict[str, int]:
    genres = sorted(metadata["genre"].unique())
    return {genre: idx for idx, genre in enumerate(genres)}


def load_note_arrays(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


class MidiPianoRollDataset(Dataset):
    def __init__(
        self,
        note_array_dir: Path,
        metadata_csv: Path,
        label_to_idx: Dict[str, int] | None = None,
        frame_rate: float = 20.0,
        max_time_steps: int = 1024,
        pitch_min: int = 0,
        n_pitches: int = 128,
        random_crop: bool = False,
    ):
        self.metadata = pd.read_csv(metadata_csv)
        self.note_array_dir = note_array_dir
        self.label_to_idx = label_to_idx or build_label_mapping(self.metadata)
        self.frame_rate = frame_rate
        self.max_time_steps = max_time_steps
        self.pitch_min = pitch_min
        self.n_pitches = n_pitches
        self.random_crop = random_crop

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        arrays = load_note_arrays(self.note_array_dir / f"{row['sample_id']}.npz")

        pianoroll = build_pianoroll_from_note_arrays(
            arrays=arrays,
            frame_rate=self.frame_rate,
            max_time_steps=self.max_time_steps,
            pitch_min=self.pitch_min,
            n_pitches=self.n_pitches,
            random_crop=self.random_crop,
        )
        x = torch.from_numpy(pianoroll).float()
        y = torch.tensor(self.label_to_idx[row["genre"]], dtype=torch.long)

        return {
            "x": x,
            "y": y,
        }


def build_pianoroll_from_note_arrays(
    arrays: Dict[str, np.ndarray],
    frame_rate: float,
    max_time_steps: int,
    pitch_min: int = 0,
    n_pitches: int = 128,
    random_crop: bool = False,
) -> np.ndarray:
    pitch = arrays["pitch"].astype(np.int16)
    onset = arrays["onset_sec"].astype(np.float32)
    duration = arrays["duration_sec"].astype(np.float32)
    velocity = arrays["velocity"].astype(np.float32)

    pianoroll = np.zeros((n_pitches, max_time_steps), dtype=np.float32)
    if len(pitch) == 0:
        return pianoroll

    note_start_sec = onset - np.min(onset)
    note_end_sec = note_start_sec + np.maximum(duration, 0.0)
    total_steps = max(int(np.ceil(np.max(note_end_sec) * frame_rate)), 1)

    crop_start = 0
    if random_crop and total_steps > max_time_steps:
        crop_start = np.random.randint(0, total_steps - max_time_steps + 1)

    starts = np.floor(note_start_sec * frame_rate).astype(np.int64) - crop_start
    ends = np.ceil(note_end_sec * frame_rate).astype(np.int64) - crop_start
    ends = np.maximum(ends, starts + 1)

    for note_pitch, start, end, note_velocity in zip(pitch, starts, ends, velocity):
        pitch_idx = int(note_pitch) - pitch_min
        if pitch_idx < 0 or pitch_idx >= n_pitches:
            continue

        start = max(int(start), 0)
        end = min(int(end), max_time_steps)
        if start >= end:
            continue

        pianoroll[pitch_idx, start:end] = np.maximum(
            pianoroll[pitch_idx, start:end],
            note_velocity / 127.0,
        )

    return pianoroll


class MidiNoteMatrixDataset(Dataset):
    def __init__(
        self,
        matrix_dir: Path,
        metadata_csv: Path,
        label_to_idx: Dict[str, int] | None = None,
        max_notes: int = 2048,
        normalize: bool = True,
    ):
        self.metadata = pd.read_csv(metadata_csv)
        self.matrix_dir = matrix_dir
        self.label_to_idx = label_to_idx or build_label_mapping(self.metadata)
        self.max_notes = max_notes
        self.normalize = normalize

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
            sequence = note_matrix[:n_notes]
            if self.normalize:
                sequence = normalize_note_matrix(sequence)
            padded[:n_notes] = sequence
            mask[:n_notes] = True

        x = torch.from_numpy(padded)
        mask = torch.from_numpy(mask)
        y = torch.tensor(self.label_to_idx[genre], dtype=torch.long)

        return {
            "x": x,
            "mask": mask,
            "y": y,
        }


def normalize_note_matrix(note_matrix: np.ndarray) -> np.ndarray:
    normalized = note_matrix.copy().astype(np.float32)

    normalized[:, 0] = normalized[:, 0] / 127.0

    onset = normalized[:, 1]
    onset = onset - np.min(onset)
    max_onset = np.max(onset)
    normalized[:, 1] = onset / max_onset if max_onset > 0 else onset

    duration = normalized[:, 2]
    max_duration = np.max(duration)
    normalized[:, 2] = duration / max_duration if max_duration > 0 else duration

    normalized[:, 3] = normalized[:, 3] / 127.0
    normalized[:, 4] = normalized[:, 4] / max(float(np.max(normalized[:, 4])), 1.0)
    normalized[:, 5] = normalized[:, 5] / 15.0

    return normalized


def compute_piece_duration(onset: np.ndarray, duration: np.ndarray) -> float:
    if len(onset) == 0:
        return 0.0
    note_end = onset + duration
    return float(np.max(note_end) - np.min(onset))


def compute_polyphony_features(onset: np.ndarray, duration: np.ndarray) -> Tuple[float, float]:
    if len(onset) == 0:
        return 0.0, 0.0

    note_end = onset + duration
    events = []

    for start, end in zip(onset, note_end):
        events.append((float(start), 1))
        events.append((float(end), -1))

    events.sort(key=lambda x: (x[0], x[1]))

    active = 0
    max_polyphony = 0
    weighted_polyphony = 0.0
    prev_time = events[0][0]

    for time, delta in events:
        dt = time - prev_time
        if dt > 0:
            weighted_polyphony += active * dt
        active += delta
        max_polyphony = max(max_polyphony, active)
        prev_time = time

    piece_duration = compute_piece_duration(onset, duration)
    avg_polyphony = weighted_polyphony / piece_duration if piece_duration > 0 else 0.0

    return float(avg_polyphony), float(max_polyphony)


def extract_note_features(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    pitch = arrays["pitch"]
    onset = arrays["onset_sec"]
    duration = arrays["duration_sec"]
    velocity = arrays["velocity"]

    features: Dict[str, float] = {}

    n_notes = float(len(pitch))
    piece_duration_sec = compute_piece_duration(onset, duration)
    note_density = n_notes / piece_duration_sec if piece_duration_sec > 0 else 0.0
    avg_polyphony, max_polyphony = compute_polyphony_features(onset, duration)

    features["n_notes"] = n_notes
    features["piece_duration_sec"] = piece_duration_sec
    features["note_density"] = note_density
    features["avg_polyphony"] = avg_polyphony
    features["max_polyphony"] = max_polyphony

    features["pitch_mean"] = float(np.mean(pitch)) if len(pitch) else 0.0
    features["pitch_std"] = float(np.std(pitch)) if len(pitch) else 0.0
    features["pitch_min"] = float(np.min(pitch)) if len(pitch) else 0.0
    features["pitch_max"] = float(np.max(pitch)) if len(pitch) else 0.0
    features["pitch_range"] = float(np.max(pitch) - np.min(pitch)) if len(pitch) else 0.0

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

    for row in tqdm(
        metadata.itertuples(index=False),
        desc="Building sklearn dataset...",
        total=len(metadata),
        file=sys.stdout,
    ):
        sample_id = row.sample_id
        genre = row.genre

        arrays = load_note_arrays(Path(note_array_dir) / f"{sample_id}.npz")
        features = extract_note_features(arrays)

        rows.append(features)
        y.append(label_to_idx[genre])

    X = pd.DataFrame(rows)
    y = np.asarray(y, dtype=np.int64)

    return X, y, label_to_idx
