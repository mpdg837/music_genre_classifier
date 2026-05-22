from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


@dataclass(frozen=True)
class SignificanceResult:
    mean_score: float
    std_score: float
    p_value: float
    corrected_alpha: float
    significant: bool
    direction: str
    n_trials: int


def sanitize_name(name: str) -> str:
    return (
        name.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_score_tables(
    rows: list[dict[str, Any]],
    output_dir: Path,
    alpha: float,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "tcav_scores.csv"
    summary_path = output_dir / "tcav_summary.csv"

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)

    summary_rows = []
    if not raw.empty:
        correction_count = (
            raw[["concept", "layer", "target_class"]].drop_duplicates().shape[0]
        )
        for keys, group in raw.groupby(["concept", "layer", "target_class"], sort=True):
            concept, layer, target_class = keys
            result = summarize_tcav_trials(
                group["sign_count"].astype(float).tolist(),
                alpha=alpha,
                correction_count=correction_count,
            )
            summary_rows.append(
                {
                    "concept": concept,
                    "layer": layer,
                    "target_class": target_class,
                    "mean_sign_count": result.mean_score,
                    "std_sign_count": result.std_score,
                    "p_value": result.p_value,
                    "corrected_alpha": result.corrected_alpha,
                    "significant": result.significant,
                    "direction": result.direction,
                    "n_trials": result.n_trials,
                }
            )

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return raw_path, summary_path


def summarize_tcav_trials(
    scores: list[float],
    alpha: float = 0.05,
    correction_count: int = 1,
) -> SignificanceResult:
    if not scores:
        return SignificanceResult(
            mean_score=0.0,
            std_score=0.0,
            p_value=float("nan"),
            corrected_alpha=alpha / max(1, correction_count),
            significant=False,
            direction="insufficient_data",
            n_trials=0,
        )

    values = np.asarray(scores, dtype=np.float64)
    corrected_alpha = alpha / max(1, correction_count)
    p_value = float("nan")
    if len(values) >= 2 and not np.allclose(values, values[0]):
        p_value = float(ttest_1samp(values, popmean=0.5).pvalue)

    mean_score = float(values.mean())
    significant = bool(np.isfinite(p_value) and p_value < corrected_alpha)
    direction = "positive" if mean_score > 0.5 else "negative"
    if not significant:
        direction = "not_significant"

    return SignificanceResult(
        mean_score=mean_score,
        std_score=float(values.std(ddof=1)) if len(values) >= 2 else 0.0,
        p_value=p_value,
        corrected_alpha=corrected_alpha,
        significant=significant,
        direction=direction,
        n_trials=int(len(values)),
    )
