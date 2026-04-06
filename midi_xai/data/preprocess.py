import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from partitura import load_score
from tqdm import tqdm

pattern = re.compile(
    r"^XMIDI_(?P<emotion>[a-z]+)_(?P<genre>[a-z]+)_(?P<sample_id>[A-Z0-9]+)\.midi$"
)


def preprocess_dataset(datasetpath: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(columns=["emotion", "genre", "sample_id"])
    for file in tqdm(os.listdir(datasetpath), desc="Preprocessing MIDI metadata"):
        match = pattern.match(file)
        if match:
            emotion = match.group("emotion")
            genre = match.group("genre")
            sample_id = match.group("sample_id")
            df = pd.concat(
                [df, pd.DataFrame([{"emotion": emotion, "genre": genre, "sample_id": sample_id}])],
                ignore_index=True,
            )

            midi_path = datasetpath / file
            data = parse_midi(midi_path)
            output_path = output_dir / (sample_id + ".npz")
            np.savez_compressed(output_path, **data)

    return df


def parse_midi(midi_path: Path):
    ppart = load_score(midi_path)
    na = ppart.note_array()
    return {
        "pitch": na["pitch"].astype(np.int16),
        "onset_sec": na["onset_sec"].astype(np.float32),
        "duration_sec": na["duration_sec"].astype(np.float32),
        "velocity": na["velocity"].astype(np.int16),
        "track": na["track"].astype(np.int16),
        "channel": na["channel"].astype(np.int16),
    }
