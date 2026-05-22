import importlib

import pytest

MODULES = [
    "torch",
    "pandas",
    "sklearn",
    "matplotlib",
    "hydra",
    "wandb",
    "captum",
    "partitura",
    "omegaconf",
    "transformers",
    "miditok",
    "scipy",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_dependencies_import(module_name: str) -> None:
    """Ensure core dependencies can be imported."""
    assert importlib.import_module(module_name) is not None


def test_torch_basic() -> None:
    """Check that torch runs a simple tensor operation."""
    import torch

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    z = x + y

    assert torch.allclose(z, torch.tensor([4.0, 6.0]))


def test_pandas_basic() -> None:
    """Check that pandas can build and query a DataFrame."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "genre": ["rock", "jazz", "rock"],
            "value": [1, 2, 3],
        }
    )

    assert list(df["genre"].unique()) == ["rock", "jazz"]
    assert int(df["value"].sum()) == 6


def test_sklearn_basic() -> None:
    """Check that sklearn metrics work."""
    from sklearn.metrics import accuracy_score

    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    assert accuracy_score(y_true, y_pred) == 0.75


def test_matplotlib_basic() -> None:
    """Check that matplotlib can create and close a figure."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    assert fig is not None
    plt.close(fig)


def test_hydra_config_load() -> None:
    """Ensure Hydra configs load correctly."""
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    assert cfg is not None
    assert OmegaConf.is_config(cfg)


def test_tcav_config_load() -> None:
    """Ensure the MuseResNet TCAV config composes."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="tcav_config")

    assert cfg.tcav.name == "muserenet_baseline"
    assert "classifier.1" in cfg.tcav.layers


def test_neural_models_compose_and_forward() -> None:
    """Hydra neural_config + MuSeReNet and Transformer forward shapes."""
    from pathlib import Path

    import hydra
    from hydra import compose, initialize_config_dir
    import torch

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        for model_name in ("muserenet", "transformer"):
            cfg = compose(config_name="neural_config", overrides=[f"model={model_name}"])
            model = hydra.utils.instantiate(cfg.model.instance, num_classes=3).eval()
            if model_name == "muserenet":
                x = torch.randn(2, 128, 64)
                logits = model(x)
            else:
                x = torch.randn(2, 32, 6)
                mask = torch.ones(2, 32, dtype=torch.bool)
                logits = model(x, mask)
            assert logits.shape == (2, 3)


def test_musicbert_classifier_forward_tiny_config() -> None:
    """MusicBERT classifier head works with a tiny offline BERT config."""
    import torch

    from midi_xai.models.neural.musicbert import MusicBertGenreClassifier

    model = MusicBertGenreClassifier(
        pretrained_model_name_or_path=None,
        encoder_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
            "type_vocab_size": 1,
        },
        num_classes=3,
        classifier_hidden_dims=[8, 4],
        freeze_encoder=True,
    ).eval()
    input_ids = torch.ones(2, 8, dtype=torch.long)
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)

    assert logits.shape == (2, 3)


def test_tcav_cav_classifier_smoke() -> None:
    """Captum-compatible CAV classifier returns normalized weights."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from midi_xai.interpretability.tcav.core import SklearnLinearSVCClassifier

    inputs = torch.cat([torch.randn(8, 5) - 1.0, torch.randn(8, 5) + 1.0])
    labels = torch.tensor([0] * 8 + [1] * 8)
    classifier = SklearnLinearSVCClassifier()

    classifier.train_and_eval(
        DataLoader(TensorDataset(inputs, labels), batch_size=4),
        test_split_ratio=0.25,
    )
    weights = classifier.weights()

    assert weights.shape == (2, 5)
    assert torch.allclose(torch.linalg.vector_norm(weights, dim=1), torch.ones(2))


def test_omegaconf_basic() -> None:
    """Check that OmegaConf can create a config object."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"model": {"name": "cnn", "dropout": 0.3}})
    assert cfg.model.name == "cnn"
    assert cfg.model.dropout == 0.3


def test_wandb_offline_init() -> None:
    """Check that wandb can initialize in disabled mode without network usage."""
    import wandb

    run = wandb.init(
        project="env-check",
        mode="disabled",
        reinit=True,
    )
    assert run is not None

    wandb.log({"sanity_metric": 1.0})
    run.finish()


def test_captum_basic() -> None:
    """Check that Captum can compute a simple attribution."""
    from captum.attr import IntegratedGradients
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(2, 1))
    model.eval()

    inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
    baseline = torch.zeros_like(inputs)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(inputs, baselines=baseline)

    assert attributions.shape == inputs.shape


def test_partitura_basic_note_array() -> None:
    """Check that partitura can create a score and extract a note array."""
    import partitura as pt
    from partitura.score import Note, Part

    part = Part("P1")
    part.add(Note(id="n1", step="C", octave=4, voice=1), start=0, end=1)
    part.add(Note(id="n2", step="E", octave=4, voice=1), start=1, end=2)

    note_array = pt.utils.music.note_array_from_part(part)

    assert len(note_array) == 2
    assert "pitch" in note_array.dtype.names
