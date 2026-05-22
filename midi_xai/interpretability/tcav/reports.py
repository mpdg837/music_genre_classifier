from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


REQUIRED_SUMMARY_COLUMNS = {
    "concept",
    "layer",
    "target_class",
    "mean_sign_count",
    "significant",
    "direction",
}


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


def aggregate_tcav_summary(
    summary: pd.DataFrame,
    include_non_significant: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_columns = REQUIRED_SUMMARY_COLUMNS.difference(summary.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"TCAV summary is missing required columns: {missing}")

    annotated = summary.copy()
    annotated["mean_sign_count"] = annotated["mean_sign_count"].astype(float)
    annotated["effect"] = annotated["mean_sign_count"] - 0.5
    annotated["abs_effect"] = annotated["effect"].abs()
    annotated["signed_effect_label"] = np.where(
        annotated["effect"] >= 0.0,
        "supports_class",
        "opposes_class",
    )
    annotated.loc[~annotated["significant"].astype(bool), "signed_effect_label"] = (
        "not_significant"
    )

    ranking_source = annotated
    if not include_non_significant:
        ranking_source = annotated[annotated["significant"].astype(bool)]

    ranked = ranking_source.sort_values(
        ["target_class", "significant", "abs_effect", "mean_sign_count"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    ranked["rank_within_class"] = (
        ranked.groupby(["target_class", "layer"], sort=False).cumcount() + 1
    )

    return annotated, ranked


def write_tcav_aggregates(
    summary_path: Path,
    output_dir: Path,
    figures_dir: Path | None = None,
    top_k: int = 8,
    include_non_significant: bool = True,
) -> dict[str, list[Path] | Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    annotated, ranked = aggregate_tcav_summary(
        summary,
        include_non_significant=include_non_significant,
    )

    annotated_path = output_dir / "tcav_summary_with_effects.csv"
    ranked_path = output_dir / "tcav_ranked_by_class.csv"
    class_layer_path = output_dir / "tcav_class_layer_concept_matrix.csv"
    class_path = output_dir / "tcav_class_concept_matrix.csv"

    annotated.to_csv(annotated_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    write_tcav_matrices(annotated, class_layer_path=class_layer_path, class_path=class_path)

    figure_paths: list[Path] = []
    if figures_dir is not None:
        figure_paths.extend(write_tcav_heatmaps(annotated, figures_dir))
        figure_paths.extend(write_tcav_class_bars(annotated, figures_dir, top_k=top_k))

    return {
        "annotated": annotated_path,
        "ranked": ranked_path,
        "class_layer_matrix": class_layer_path,
        "class_matrix": class_path,
        "figures": figure_paths,
    }


def write_tcav_matrices(
    annotated: pd.DataFrame,
    class_layer_path: Path,
    class_path: Path,
) -> None:
    class_layer = annotated.pivot_table(
        index=["layer", "concept"],
        columns="target_class",
        values="effect",
        aggfunc="mean",
    )
    class_layer.to_csv(class_layer_path)

    class_matrix = annotated.pivot_table(
        index="concept",
        columns="target_class",
        values="effect",
        aggfunc="mean",
    )
    class_matrix.to_csv(class_path)


def write_tcav_heatmaps(annotated: pd.DataFrame, figures_dir: Path) -> list[Path]:
    paths = []
    for layer, layer_df in annotated.groupby("layer", sort=True):
        matrix = layer_df.pivot_table(
            index="concept",
            columns="target_class",
            values="effect",
            aggfunc="mean",
        )
        significant = layer_df.pivot_table(
            index="concept",
            columns="target_class",
            values="significant",
            aggfunc="max",
        ).reindex_like(matrix).fillna(False)
        path = figures_dir / f"tcav_heatmap_{sanitize_name(str(layer))}.png"
        _plot_tcav_heatmap(matrix, significant, title=f"TCAV effect - {layer}", output_path=path)
        paths.append(path)

    aggregate_matrix = annotated.pivot_table(
        index="concept",
        columns="target_class",
        values="effect",
        aggfunc="mean",
    )
    aggregate_significant = annotated.pivot_table(
        index="concept",
        columns="target_class",
        values="significant",
        aggfunc="max",
    ).reindex_like(aggregate_matrix).fillna(False)
    aggregate_path = figures_dir / "tcav_heatmap_all_layers_mean.png"
    _plot_tcav_heatmap(
        aggregate_matrix,
        aggregate_significant,
        title="TCAV effect - mean across layers",
        output_path=aggregate_path,
    )
    paths.append(aggregate_path)
    return paths


def write_tcav_class_bars(
    annotated: pd.DataFrame,
    figures_dir: Path,
    top_k: int = 8,
) -> list[Path]:
    paths = []
    for layer, layer_df in annotated.groupby("layer", sort=True):
        for target_class, class_df in layer_df.groupby("target_class", sort=True):
            selected = (
                class_df.assign(abs_effect=class_df["effect"].abs())
                .sort_values(["significant", "abs_effect"], ascending=[False, False])
                .head(top_k)
                .sort_values("effect")
            )
            path = (
                figures_dir
                / f"tcav_top_concepts_{sanitize_name(str(layer))}_{sanitize_name(str(target_class))}.png"
            )
            _plot_tcav_class_bar(
                selected,
                title=f"{target_class} - {layer}",
                output_path=path,
            )
            paths.append(path)
    return paths


def _plot_tcav_heatmap(
    matrix: pd.DataFrame,
    significant: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if matrix.empty:
        return

    vmax = max(0.5, float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))))
    fig_width = max(7.0, 1.0 + 1.2 * len(matrix.columns))
    fig_height = max(4.5, 1.0 + 0.45 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Concept")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for y_idx, concept in enumerate(matrix.index):
        for x_idx, target_class in enumerate(matrix.columns):
            value = matrix.loc[concept, target_class]
            if pd.isna(value):
                continue
            marker = "*" if bool(significant.loc[concept, target_class]) else ""
            ax.text(
                x_idx,
                y_idx,
                f"{value:+.2f}{marker}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("mean TCAV sign count - 0.5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_tcav_class_bar(
    class_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if class_df.empty:
        return

    colors = np.where(class_df["significant"].astype(bool), "#2f7ed8", "#b8b8b8")
    fig_height = max(3.5, 0.45 * len(class_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_height), constrained_layout=True)
    ax.barh(class_df["concept"], class_df["effect"], color=colors)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("TCAV effect (mean sign count - 0.5)")
    ax.set_ylabel("Concept")

    for y_idx, (_, row) in enumerate(class_df.iterrows()):
        marker = "*" if bool(row["significant"]) else ""
        ax.text(
            float(row["effect"]),
            y_idx,
            f" {float(row['mean_sign_count']):.2f}{marker}",
            va="center",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
