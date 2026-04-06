from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
    def __init__(self, matrix_dir: Path, metadata_csv: Path):
        self.metadata = pd.read_csv(metadata_csv)
        self.matrix_dir = matrix_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx): ...
