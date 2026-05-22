from dataclasses import dataclass
from pathlib import Path
from typing import Any

from captum.concept import TCAV, Concept
from captum.concept._utils.classifier import Classifier
from captum.concept._utils.common import concepts_to_str
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import torch
from torch import nn
from torch.utils.data import DataLoader


class SklearnLinearSVCClassifier(Classifier):
    def __init__(self) -> None:
        self.classifier: LinearSVC | None = None
        self._classes: list[int] = []

    def train_and_eval(
        self,
        dataloader: DataLoader,
        test_split_ratio: float = 0.2,
        c: float = 1.0,
        max_iter: int = 10000,
        seed: int = 42,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        inputs = []
        labels = []
        for batch_inputs, batch_labels in dataloader:
            inputs.append(batch_inputs.detach().cpu())
            labels.append(batch_labels.detach().cpu())

        if not inputs:
            raise ValueError("Cannot train a CAV classifier on an empty dataloader.")

        x = torch.cat(inputs, dim=0).numpy()
        y = torch.cat(labels, dim=0).numpy()
        self.classifier = LinearSVC(
            C=c,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=seed,
        )
        accuracy = _fit_and_score_classifier(self.classifier, x, y, test_split_ratio, seed)
        self._classes = [int(value) for value in self.classifier.classes_]
        return {"accs": torch.tensor(accuracy, dtype=torch.float32)}

    def weights(self) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("The CAV classifier has not been trained yet.")

        weights = torch.from_numpy(self.classifier.coef_.astype(np.float32))
        if weights.shape[0] == 1:
            weights = torch.stack([-weights[0], weights[0]], dim=0)

        norms = torch.linalg.vector_norm(weights, dim=1, keepdim=True).clamp_min(1e-12)
        return weights / norms

    def classes(self) -> list[int]:
        if not self._classes:
            raise RuntimeError("The CAV classifier has not been trained yet.")
        return self._classes


@dataclass(frozen=True)
class TcavScore:
    sign_count: float
    magnitude: float
    n_examples: int


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    label_to_idx: dict[str, int],
    device: torch.device,
    strict: bool = True,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_label_to_idx = checkpoint.get("label_to_idx")
        if checkpoint_label_to_idx is not None and dict(checkpoint_label_to_idx) != label_to_idx:
            raise ValueError("Checkpoint label mapping does not match the current dataset labels.")
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        checkpoint = {}

    load_result = model.load_state_dict(state_dict, strict=strict)
    if strict and load_result.missing_keys:
        raise RuntimeError(f"Missing checkpoint keys: {load_result.missing_keys}")
    if strict and load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")

    return checkpoint


def compute_captum_tcav_score(
    tcav: TCAV,
    dataloader: DataLoader,
    experimental_set: list[Concept],
    layer_name: str,
    target_class: int,
    device: torch.device,
    concept_index: int = 0,
) -> TcavScore:
    weighted_sign_count = 0.0
    weighted_magnitude = 0.0
    n_examples = 0
    concepts_key = concepts_to_str(experimental_set)

    for batch in dataloader:
        inputs = batch["x"].to(device)
        batch_size = int(inputs.shape[0])
        scores = tcav.interpret(
            inputs=inputs,
            experimental_sets=[experimental_set],
            target=target_class,
        )
        layer_scores = scores[concepts_key][layer_name]
        sign_count = float(layer_scores["sign_count"][concept_index].detach().cpu().item())
        magnitude = float(layer_scores["magnitude"][concept_index].detach().cpu().item())

        weighted_sign_count += sign_count * batch_size
        weighted_magnitude += magnitude * batch_size
        n_examples += batch_size

    if n_examples == 0:
        return TcavScore(sign_count=0.0, magnitude=0.0, n_examples=0)

    return TcavScore(
        sign_count=weighted_sign_count / n_examples,
        magnitude=weighted_magnitude / n_examples,
        n_examples=n_examples,
    )


def _fit_and_score_classifier(
    classifier: LinearSVC,
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> float:
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2 or np.any(counts < 2) or test_size <= 0:
        classifier.fit(x, y)
        return float(accuracy_score(y, classifier.predict(x)))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    classifier.fit(x_train, y_train)
    return float(accuracy_score(y_test, classifier.predict(x_test)))
