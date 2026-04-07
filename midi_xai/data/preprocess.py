from pathlib import Path
import re
from typing import Dict

import numpy as np
import pandas as pd
from partitura import load_performance_midi
from tqdm import tqdm

pattern = re.compile(
    r"^XMIDI_(?P<emotion>[a-z]+)_(?P<genre>[a-z]+)_(?P<sample_id>[A-Z0-9]+)\.midi$"
)


def preprocess_dataset(datasetpath: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    midi_files = list(datasetpath.rglob("*.midi")) + list(datasetpath.rglob("*.mid"))

    for midi_path in tqdm(midi_files, desc="Preprocessing MIDI metadata"):
        file = midi_path.name
        match = pattern.match(file)
        if not match:
            continue

        emotion = match.group("emotion")
        genre = match.group("genre")
        sample_id = match.group("sample_id")

        rows.append(
            {
                "emotion": emotion,
                "genre": genre,
                "sample_id": sample_id,
                "filename": file,
                "filepath": str(midi_path),
            }
        )

        data = parse_midi(midi_path)
        output_path = output_dir / f"{sample_id}.npz"
        np.savez_compressed(output_path, **data)

    return pd.DataFrame(rows)


def parse_midi(midi_path: Path) -> Dict[str, np.ndarray]:
    ppart = load_performance_midi(str(midi_path))
    na = ppart.note_array()

    return {
        "pitch": na["pitch"].astype(np.int16),
        "onset_sec": na["onset_sec"].astype(np.float32),
        "duration_sec": na["duration_sec"].astype(np.float32),
        "velocity": na["velocity"].astype(np.int16),
        "track": na["track"].astype(np.int16),
        "channel": na["channel"].astype(np.int16),
    }
