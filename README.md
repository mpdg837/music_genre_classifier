# Music Genre XAI Classifier

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Genre classification on symbolic MIDI (XMIDI): classic scikit-learn baselines, **MuSeReNet** (multi-resolution CNN on a piano roll), and a **Transformer** encoder over a padded note sequence. Configuration uses [Hydra](https://hydra.cc/); metrics can be logged with [Weights & Biases](https://wandb.ai/).

## Requirements

- Python **3.11** (see `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) (or your own venv plus `pip install -e .` for the `midi_xai` package)

## Setup

From the repository root:

```bash
make requirements
# or: uv sync
```

Activate the virtual environment (after `make create_environment` / `uv venv`):

```bash
source .venv/bin/activate
```

## Data (XMIDI)

1. Paths and download settings live in [`configs/data/xmidi.yaml`](configs/data/xmidi.yaml), e.g. `data/raw/xmidi`, `data/processed/xmidi`, `data/interim/xmidi_labels.csv`.
2. Prepare raw MIDI into per-piece `.npz` files plus a label CSV:

```bash
uv run python scripts/prepare_data.py
```

**Neural models do not need a separate “export neural dataset to disk” step.** The same `processed_dir` with `*.npz` files is used for classic models (hand-crafted features in code) and for:

- **MuSeReNet** — piano roll built on the fly in [`MidiPianoRollDataset`](midi_xai/data/create_dataset.py),
- **Transformer** — note matrix + padding mask in [`MidiNoteMatrixDataset`](midi_xai/data/create_dataset.py).

## Training

### Classic baseline

Default config: [`configs/config.yaml`](configs/config.yaml) (e.g. `linear_svc`). Pick another model from `configs/model/classic/`:

```bash
uv run python scripts/train_classic.py
uv run python scripts/train_classic.py model/classic=svc
```

### Neural models

Entry point: [`scripts/train_neural.py`](scripts/train_neural.py); top-level neural config: [`configs/neural_config.yaml`](configs/neural_config.yaml).

**MuSeReNet** (default):

```bash
uv run python scripts/train_neural.py
```

**Transformer:**

```bash
uv run python scripts/train_neural.py model=transformer
```

Example Hydra overrides (epochs, batch size):

```bash
uv run python scripts/train_neural.py model=transformer model.training.epochs=50 model.dataset.batch_size=16
```

Checkpoints are written under `save_weights_path` (default `checkpoints/`); the filename comes from the model’s `name` field in its YAML (e.g. `muserenet.pt`, `midi_transformer.pt`).

### GPU / cluster

Training uses `cuda` when PyTorch detects a GPU; otherwise CPU. On a cluster, load a CUDA-capable module / image that matches your PyTorch build and verify with `nvidia-smi`.

### Weights & Biases

Both training scripts call `wandb.init`. For non-interactive or air-gapped runs:

```bash
export WANDB_MODE=offline
# or after: wandb login
```

## Tests

```bash
make test
# or: uv run pytest
```

Tests cover dependency imports, Hydra config loading, and a smoke forward pass for **MuSeReNet** and **Transformer** (instantiated from configs).

## Repository layout (short)

| Path | Role |
|------|------|
| `midi_xai/data/` | Fetch, preprocess, `create_dataset.py` (PyTorch `Dataset`s + sklearn feature extraction) |
| `midi_xai/models/classic_model.py` | Classic model wrapper |
| `midi_xai/models/neural/muserenet.py` | MuSeReNet |
| `midi_xai/models/neural/transformer.py` | Transformer encoder + classifier head |
| `configs/` | Hydra: `config.yaml`, `neural_config.yaml`, `data/`, `model/` |
| `scripts/` | `prepare_data.py`, `train_classic.py`, `train_neural.py`, `test_dependencies.py` |

## Project organization

Layout of the repository as used in this codebase (CCDS-inspired; paths may be empty until you run pipelines).

```
├── Makefile                 # uv sync, lint (ruff), format, tests
├── pyproject.toml           # package `midi_xai`, dependencies, ruff
├── uv.lock
├── README.md
│
├── configs/                 # Hydra
│   ├── config.yaml          # classic training defaults
│   ├── neural_config.yaml   # neural training defaults (MuSeReNet)
│   ├── data/
│   │   ├── default.yaml
│   │   └── xmidi.yaml       # XMIDI paths + download id
│   └── model/
│       ├── muserenet.yaml
│       ├── transformer.yaml
│       └── classic/         # linear_svc, svc, mlp, knn, …
│
├── scripts/
│   ├── prepare_data.py      # download + preprocess → npz + labels CSV
│   ├── train_classic.py     # sklearn baselines
│   ├── train_neural.py      # MuSeReNet / Transformer
│   └── test_dependencies.py # pytest: imports, Hydra, model smoke tests
│
├── midi_xai/                # installable package (import as `midi_xai`)
│   ├── __init__.py
│   ├── data/
│   │   ├── fetch_dataset.py
│   │   ├── preprocess.py
│   │   └── create_dataset.py  # MidiPianoRollDataset, MidiNoteMatrixDataset, sklearn features
│   └── models/
│       ├── classic_model.py
│       └── neural/
│           ├── muserenet.py
│           ├── transformer.py
│           └── __init__.py
│
├── data/                    # created at runtime (see configs/data/xmidi.yaml)
│   ├── raw/xmidi/           # extracted MIDI corpus
│   ├── processed/xmidi/     # per-piece *.npz note arrays
│   └── interim/
│       └── xmidi_labels.csv # sample_id → genre (and metadata columns)
│
├── checkpoints/             # saved .pt (neural) / .joblib (classic); gitignored if large
├── notebooks/               # exploratory notebooks
├── docs/                    # mkdocs / design docs
├── reports/                 # figures / write-ups (optional)
├── outputs/                 # Hydra multirun logs (local runs)
└── wandb/                   # local W&B run data (if not using cloud-only mode)
```

--------