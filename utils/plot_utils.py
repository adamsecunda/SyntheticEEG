import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import numpy as np
from utils.classifier_utils import CLASS_NAMES

# Constants for consistent styling across the project
PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,
}

PALETTE = {
    "balanced": {"facecolor": "#4477AA", "edgecolor": "#2B4F73"},
    "imbalanced": {"facecolor": "#EE6677", "edgecolor": "#A63D4A"},
    "augmented": {"facecolor": "#228833", "edgecolor": "#15531F"},
    "baseline_line": "#444444",
}

FULL_CLASS_NAMES = [
    f"{name} Hand" if name in ("Left", "Right") else name for name in CLASS_NAMES
]


def _save(fig: plt.Figure, save_dir: Union[str, Path], name: str) -> None:
    """Saves the figure as a high-resolution PDF and closes the plot."""
    path = Path(save_dir) / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", format="pdf", dpi=600)
    plt.close(fig)


def _style_ax(ax: plt.Axes) -> None:
    """Applies clean, consistent axis styling with percentage y-axis labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)

    # Increased top margin to 1.15 to accommodate bar text labels
    ax.set_ylim(0, 1.15)
    ticks = np.linspace(0, 1, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.0%}" for v in ticks])
    ax.tick_params(length=3.5, width=0.85)


def _legend(ax: plt.Axes, **kwargs: Any) -> None:
    """Adds a clean legend with consistent frame and font appearance."""
    ax.legend(
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        fancybox=False,
        fontsize=8,
        **kwargs,
    )


def _bar(
    ax: plt.Axes,
    x: np.ndarray,
    heights: List[float],
    width: float,
    palette_key: str,
    label: Optional[str] = None,
    zorder: int = 3,
) -> BarContainer:
    """Draws a themed bar chart using the project's color palette."""
    return ax.bar(
        x,
        heights,
        width,
        label=label,
        color=PALETTE[palette_key]["facecolor"],
        edgecolor=PALETTE[palette_key]["edgecolor"],
        linewidth=0.85,
        zorder=zorder,
    )


def _plot_baseline(results: Dict[str, Any], save_dir: Path) -> None:
    """Generates the baseline performance bar chart."""
    balanced = results["balanced"]
    # Increased height slightly to accommodate the lower legend
    fig, ax = plt.subplots(figsize=(4.1, 3.1), constrained_layout=True)

    x = np.arange(len(FULL_CLASS_NAMES))
    _bar(ax, x, balanced["per_class"], 0.62, "balanced", "Per-class Accuracy")

    ax.axhline(
        y=balanced["overall"],
        color=PALETTE["baseline_line"],
        linestyle="--",
        linewidth=1.1,
        label=f"Overall Accuracy: {balanced['overall']:.1%}",
    )

    ax.set_title("Baseline Performance (Balanced Dataset)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(FULL_CLASS_NAMES, rotation=30, ha="right")

    _style_ax(ax)
    # Lowered legend to -0.28 for better spacing
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    for bar in ax.patches:
        if isinstance(bar, plt.Rectangle) and bar.get_label() != "_nolegend_":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _save(fig, save_dir, "baseline_performance")


def _plot_impact_matrix(
    results: Dict[str, Any], save_dir: Path, removal_pct: float = 0.5
) -> None:
    """Plots cross-class accuracy impact heatmap."""
    balanced_accs = results["balanced"]["per_class"]
    tested_classes = sorted(results["imbalanced"].keys())
    num_classes = len(FULL_CLASS_NAMES)

    change_matrix = np.zeros((num_classes, num_classes))
    for removed_idx in tested_classes:
        accs = results["imbalanced"][removed_idx][removal_pct]["per_class"]
        for affected_idx in range(num_classes):
            change_matrix[affected_idx, removed_idx] = (
                accs[affected_idx] - balanced_accs[affected_idx]
            ) * 100

    v_limit = max(np.max(np.abs(change_matrix)), 5.0)

    fig, ax = plt.subplots(figsize=(4.3, 3.7), constrained_layout=True)
    im = ax.imshow(
        change_matrix, cmap="RdBu", aspect="auto", vmin=-v_limit, vmax=v_limit
    )

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(FULL_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(FULL_CLASS_NAMES)

    title_pct = f"{int(removal_pct * 100)}% Data Removal"
    ax.set_title(f"Cross-Class Accuracy Impact\n({title_pct})")

    plt.colorbar(im, ax=ax).set_label("Accuracy Change (pp)", rotation=270, labelpad=14)

    for i in range(num_classes):
        for j in range(num_classes):
            val = change_matrix[i, j]
            color = "white" if abs(val) > v_limit * 0.6 else "#333333"
            ax.text(
                j, i, f"{val:+.1f}", ha="center", va="center", color=color, fontsize=7.5
            )

    suffix = f"_{int(removal_pct * 100)}pct"
    _save(fig, save_dir, f"cross_class_change_matrix{suffix}")


def _plot_augmentation_summary(
    results: Dict[str, Any], aug_results: Dict[str, Any], save_dir: Path
) -> None:
    """Compares baseline, imbalanced, and augmented performance in one bar chart."""
    tested_classes = sorted(results["imbalanced"].keys())
    pct = 0.5
    x = np.arange(len(tested_classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)

    baseline_vals = [results["balanced"]["per_class"][c] for c in tested_classes]
    imb_vals = [results["imbalanced"][c][pct]["per_class"][c] for c in tested_classes]
    aug_vals = [aug_results[c][pct]["per_class"][c] for c in tested_classes]

    _bar(ax, x - width, baseline_vals, width, "balanced", "Balanced Baseline")
    _bar(ax, x, imb_vals, width, "imbalanced", "Imbalanced (50% Removal)")
    _bar(ax, x + width, aug_vals, width, "augmented", "Augmented (Synthetic)")

    ax.set_title("Augmentation Recovery at 50% Data Removal")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [FULL_CLASS_NAMES[c] for c in tested_classes], rotation=30, ha="right"
    )

    _style_ax(ax)
    # Lowered legend to -0.28 for consistency with baseline plot
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    _save(fig, save_dir, "augmentation_recovery_summary")


def _plot_recovery_matrix(
    results: Dict[str, Any],
    aug_results: Dict[str, Any],
    save_dir: Path,
    pct: float = 0.5,
) -> None:
    """Plots the accuracy improvement matrix gained from synthetic augmentation."""
    tested_classes = sorted(results["imbalanced"].keys())
    num_classes = len(FULL_CLASS_NAMES)

    change_matrix = np.zeros((num_classes, num_classes))
    for c_idx in tested_classes:
        imb = results["imbalanced"][c_idx][pct]["per_class"]
        aug = aug_results[c_idx][pct]["per_class"]
        for affected in range(num_classes):
            change_matrix[affected, c_idx] = (aug[affected] - imb[affected]) * 100

    v_limit = max(np.max(np.abs(change_matrix)), 5.0)

    fig, ax = plt.subplots(figsize=(4.4, 3.9), constrained_layout=True)
    im = ax.imshow(
        change_matrix, cmap="RdBu", aspect="auto", vmin=-v_limit, vmax=v_limit
    )

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(FULL_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(FULL_CLASS_NAMES)

    ax.set_title(
        f"Accuracy Gain from Synthetic Augmentation\n({int(pct * 100)}% Removal)"
    )

    plt.colorbar(im, ax=ax).set_label("Accuracy Change (pp)", rotation=270, labelpad=14)

    for i in range(num_classes):
        for j in range(num_classes):
            val = change_matrix[i, j]
            color = "white" if abs(val) > v_limit * 0.6 else "#333333"
            ax.text(
                j, i, f"{val:+.1f}", ha="center", va="center", color=color, fontsize=7.5
            )

    suffix = f"_{int(pct * 100)}pct"
    _save(fig, save_dir, f"augmentation_gain_matrix{suffix}")


def plot_results(
    results: Dict[str, Any],
    aug_results: Optional[Dict[str, Any]] = None,
    save_dir: str = "plots",
) -> None:
    """Generates all publication-ready plots with consistent styling."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Apply project-wide Matplotlib configuration
    plt.rcParams.update(PLOT_STYLE)

    try:
        _plot_baseline(results, save_path)
        _plot_impact_matrix(results, save_path, removal_pct=0.5)
        _plot_impact_matrix(results, save_path, removal_pct=1.0)

        if aug_results:
            _plot_augmentation_summary(results, aug_results, save_path)
            _plot_recovery_matrix(results, aug_results, save_path, pct=0.5)
            _plot_recovery_matrix(results, aug_results, save_path, pct=1.0)

        logging.info("Plots successfully saved to: %s", save_path.resolve())

    except KeyError as e:
        logging.error("Result dictionary missing required key: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred during plotting: %s", e)
        raise