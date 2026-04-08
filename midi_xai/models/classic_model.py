from typing import Any, Dict, Self

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import FunctionTransformer
import wandb


class ClassicModel:
    def __init__(
        self,
        model: BaseEstimator,
        imputer: BaseEstimator | None = None,
        scaler: BaseEstimator | None = None,
        is_transform: bool = False,
    ):
        self.model = model
        self.imputer = imputer
        self.scaler = scaler
        self.transformer = (
            FunctionTransformer(np.log1p, validate=False) if is_transform else None
        )

    def _preprocess_fit(self, x: pd.DataFrame) -> np.ndarray:
        x_out = x
        if self.imputer is not None:
            x_out = self.imputer.fit_transform(x_out)
        if self.transformer is not None:
            x_out = self.transformer.fit_transform(x_out)
        if self.scaler is not None:
            x_out = self.scaler.fit_transform(x_out)
        return x_out

    def _preprocess_transform(self, x: pd.DataFrame) -> np.ndarray:
        x_out = x
        if self.imputer is not None:
            x_out = self.imputer.transform(x_out)
        if self.transformer is not None:
            x_out = self.transformer.transform(x_out)
        if self.scaler is not None:
            x_out = self.scaler.transform(x_out)
        return x_out

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Self:
        x_processed = self._preprocess_fit(x)
        self.model.fit(x_processed, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_processed = self._preprocess_transform(x)
        return self.model.predict(x_processed)

    def evaluate(self, x: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predict(x)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "report": classification_report(y, y_pred),
        }


def log_metrics(
    model: ClassicModel, X: pd.DataFrame, y: np.ndarray, prefix: str
) -> None:
    metrics = model.evaluate(X, y)
    wandb.log(
        {
            f"{prefix}_accuracy": metrics["accuracy"],
            f"{prefix}_f1_macro": metrics["f1_macro"],
        }
    )
    logger.info("{} classification report:\n{}", prefix, metrics["report"])
