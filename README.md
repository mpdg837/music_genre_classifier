# Music Genre XAI Classifier

Brief project README for MIDI genre classification and concept-based interpretability.

The project trains genre classifiers on symbolic MIDI data from XMIDI and analyzes learned genre cues with TCAV. It includes classic scikit-learn baselines, MuSeReNet, a MIDI Transformer, Hugging Face MusicBERT fine-tuning, and TCAV workflows for interpretability.

## Current Scope

- Dataset: XMIDI, currently 52,421 processed MIDI samples.
- Task: 6-way genre classification: `classical`, `country`, `jazz`, `pop`, `rock`, `traditional`.
- Representations:
  - tabular symbolic features for classic models,
  - piano-roll for MuSeReNet,
  - padded note sequences for MIDI Transformer,
  - REMI token windows for MusicBERT.
- Interpretability: TCAV workflows for model-level concept analysis.

## Setup

Requirements:

- Python 3.11
- `uv`

Install dependencies:

```bash
make requirements
```

Create a venv manually if needed:

```bash
make create_environment
source .venv/bin/activate
```

## Data

Prepare XMIDI into `.npz` note arrays and a label CSV:

```bash
make data
# or
uv run python scripts/prepare_data.py
```

Configured paths live in:

- `configs/data/xmidi.yaml`

## Training

Classic baselines:

```bash
uv run python scripts/train_classic.py
uv run python scripts/train_classic.py model/classic=svc
uv run python scripts/train_classic.py model/classic=random_forest
```

Neural models:

```bash
make train_muserenet
make train_transformer
make train_musicbert
make train_musicbert_frozen_head
```

Equivalent direct commands:

```bash
uv run python scripts/train_neural.py model=muserenet
uv run python scripts/train_neural.py model=transformer
uv run python scripts/train_neural.py model=musicbert
uv run python scripts/train_neural.py model=musicbert_frozen_head
```

Training is configured with Hydra in:

- `configs/neural_config.yaml`
- `configs/model/*.yaml`
- `configs/model/classic/*.yaml`

Weights & Biases logging is supported. Use offline mode when needed:

```bash
export WANDB_MODE=offline
```

## TCAV

Prepare concept and random-control manifests:

```bash
make prepare_tcav_concepts
make prepare_tcav_controls
```

Run MuSeReNet TCAV:

```bash
make tcav_muserenet
```

Run MusicBERT TCAV directly:

```bash
uv run python scripts/run_tcav_musicbert.py model=musicbert tcav=musicbert_2614025
```

Aggregate TCAV summaries:

```bash
uv run python scripts/aggregate_tcav_results.py \
  --summary-csv /path/to/tcav_summary.csv \
  --output-dir checkpoints/tcav_scores/aggregated \
  --figures-dir reports/tcav
```

Generated TCAV figures and summaries are stored under:

- `reports/tcav/`
- `reports/tcav_muserenet_summary.md`

## Results Snapshot

Best validation macro-F1 observed in W&B:

| Model | Best val macro-F1 |
|---|---:|
| MusicBERT full fine-tuning | 0.588 |
| Random Forest | 0.562 |
| MuSeReNet, larger run | 0.507 |
| MuSeReNet baseline | 0.485 |
| MusicBERT frozen head | 0.456 |
| SVC | 0.450 |
| MLP | 0.436 |
| KNN | 0.431 |
| Logistic Regression | 0.395 |
| Linear SVC | 0.389 |
| MIDI Transformer | 0.387 |

MuSeReNet TCAV summary:

- 132 tests,
- 20 statistically significant results,
- significant concepts appeared in `classifier.1`,
- examples: `pop` with high note density, `rock` with strong velocity / short notes, `classical` with long notes / high pitch register.

## Reports

Useful project write-ups:

- `reports/raport.md` - updated final report.
- `reports/muserenet_transformer_classification_comparison.md` - detailed MuSeReNet vs Transformer comparison.
- `reports/tcav_muserenet_summary.md` - TCAV summary for MuSeReNet.

## Repository Layout

| Path | Purpose |
|---|---|
| `configs/` | Hydra configs for data, models, training and TCAV |
| `midi_xai/data/` | MIDI preprocessing, datasets and feature extraction |
| `midi_xai/models/` | Classic wrapper and neural model definitions |
| `midi_xai/interpretability/tcav/` | TCAV helpers and reporting logic |
| `scripts/` | Data prep, training, TCAV and aggregation entry points |
| `slurm/` | Cluster scripts |
| `reports/` | Figures, summaries and presentation notes |
| `notebooks/` | Exploratory analysis |

## Tests and Formatting

```bash
make test
make lint
make format
```
