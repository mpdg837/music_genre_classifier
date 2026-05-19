from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from miditok import REMI, MusicTokenizer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from midi_xai.data.create_dataset import build_label_mapping


def load_remi_tokenizer(tokenizer_name_or_path: str) -> MusicTokenizer:
    tokenizer_path = Path(tokenizer_name_or_path).expanduser()
    if tokenizer_path.exists():
        tokenizer_file = (
            tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
        )
        return REMI(params=tokenizer_file)

    try:
        return MusicTokenizer.from_pretrained(tokenizer_name_or_path)
    except TypeError as exc:
        message = str(exc)
        if "proxies" not in message or "resume_download" not in message:
            raise

        tokenizer_file = hf_hub_download(
            repo_id=tokenizer_name_or_path,
            filename="tokenizer.json",
        )
        return REMI(params=Path(tokenizer_file))


def get_token_id(
    tokenizer: MusicTokenizer, token_name: str, fallback: int | None = None
) -> int | None:
    attr_name = token_name.lower().replace("_none", "") + "_token_id"
    token_id = getattr(tokenizer, attr_name, None)
    if token_id is not None:
        return int(token_id)

    try:
        return int(tokenizer[token_name])
    except (KeyError, TypeError, AttributeError):
        return fallback


def flatten_token_ids(tokenized: Any) -> list[int]:
    if hasattr(tokenized, "ids"):
        ids = tokenized.ids
        if ids and isinstance(ids[0], list):
            return [int(token_id) for stream in ids for token_id in stream]
        return [int(token_id) for token_id in ids]

    if isinstance(tokenized, (list, tuple)):
        ids = []
        for item in tokenized:
            ids.extend(flatten_token_ids(item))
        return ids

    raise TypeError(f"Unexpected MidiTok output type: {type(tokenized)!r}")


def crop_or_pad_token_ids(
    token_ids: list[int],
    max_length: int,
    pad_token_id: int,
    random_crop: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) > max_length:
        if random_crop:
            start = np.random.randint(0, len(token_ids) - max_length + 1)
        else:
            start = 0
        token_ids = token_ids[start : start + max_length]

    attention_mask = [1] * len(token_ids)
    pad_length = max_length - len(token_ids)
    if pad_length > 0:
        token_ids = token_ids + [pad_token_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length

    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def build_token_windows(
    token_ids: list[int],
    max_length: int,
    stride: int,
    max_windows: int | None = None,
) -> list[list[int]]:
    if len(token_ids) <= max_length:
        return [token_ids]

    stride = max(1, stride)
    starts = list(range(0, len(token_ids) - max_length + 1, stride))
    last_start = len(token_ids) - max_length
    if starts[-1] != last_start:
        starts.append(last_start)

    if max_windows is not None and len(starts) > max_windows:
        selected = np.linspace(0, len(starts) - 1, num=max_windows, dtype=np.int64)
        starts = [starts[int(index)] for index in selected]

    return [token_ids[start : start + max_length] for start in starts]


class MidiMusicBertDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Path,
        tokenizer_name_or_path: str,
        label_to_idx: dict[str, int] | None = None,
        midi_root_dir: Path | None = None,
        max_length: int = 1024,
        random_crop: bool = False,
        add_bos_eos: bool = True,
        cache_token_ids: bool = True,
    ):
        self.metadata = pd.read_csv(metadata_csv)
        self.label_to_idx = label_to_idx or build_label_mapping(self.metadata)
        self.midi_root_dir = Path(midi_root_dir) if midi_root_dir is not None else None
        self.max_length = max_length
        self.random_crop = random_crop
        self.add_bos_eos = add_bos_eos
        self.cache_token_ids = cache_token_ids
        self._token_cache: dict[str, list[int]] = {}

        self.tokenizer = load_remi_tokenizer(tokenizer_name_or_path)
        self.pad_token_id = get_token_id(self.tokenizer, "PAD_None", fallback=0)
        self.bos_token_id = get_token_id(self.tokenizer, "BOS_None")
        self.eos_token_id = get_token_id(self.tokenizer, "EOS_None")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        sample_id = str(row["sample_id"])
        genre = row["genre"]

        token_ids = self._get_token_ids(row=row, sample_id=sample_id)
        input_ids, attention_mask = crop_or_pad_token_ids(
            token_ids=token_ids,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            random_crop=self.random_crop,
        )
        y = torch.tensor(self.label_to_idx[genre], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "y": y,
        }

    def get_windowed_item(
        self,
        idx: int,
        stride: int,
        max_windows: int | None = None,
    ) -> dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        sample_id = str(row["sample_id"])
        genre = row["genre"]
        token_ids = self._get_token_ids(row=row, sample_id=sample_id)
        windows = build_token_windows(
            token_ids=token_ids,
            max_length=self.max_length,
            stride=stride,
            max_windows=max_windows,
        )
        tensors = [
            crop_or_pad_token_ids(
                token_ids=window,
                max_length=self.max_length,
                pad_token_id=self.pad_token_id,
                random_crop=False,
            )
            for window in windows
        ]
        input_ids = torch.stack([item[0] for item in tensors])
        attention_mask = torch.stack([item[1] for item in tensors])
        y = torch.tensor(self.label_to_idx[genre], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "y": y,
        }

    def _get_token_ids(self, row: pd.Series, sample_id: str) -> list[int]:
        if self.cache_token_ids and sample_id in self._token_cache:
            return self._token_cache[sample_id]

        midi_path = self._resolve_midi_path(row)
        token_ids = flatten_token_ids(self.tokenizer(midi_path))

        if self.add_bos_eos:
            if self.bos_token_id is not None:
                token_ids = [self.bos_token_id] + token_ids
            if self.eos_token_id is not None:
                token_ids = token_ids + [self.eos_token_id]

        if not token_ids:
            token_ids = [self.pad_token_id]

        if self.cache_token_ids:
            self._token_cache[sample_id] = token_ids

        return token_ids

    def _resolve_midi_path(self, row: pd.Series) -> Path:
        if "filepath" in row and isinstance(row["filepath"], str):
            path = Path(row["filepath"])
            if path.exists():
                return path

        filename = row.get("filename")
        if self.midi_root_dir is not None and isinstance(filename, str):
            direct_path = self.midi_root_dir / filename
            if direct_path.exists():
                return direct_path

            matches = list(self.midi_root_dir.rglob(filename))
            if matches:
                return matches[0]

        sample_id = row.get("sample_id", "<unknown>")
        raise FileNotFoundError(f"Could not resolve MIDI path for sample_id={sample_id}")
