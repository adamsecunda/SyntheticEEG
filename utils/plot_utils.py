import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.classifier_utils import CLASS_NAMES

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,
}

# Paul Tol muted palette
PALETTE = {
    "balanced": {"face": "#4477AA", "edge": "#2B4F73"},
    "removal_50": {"face": "#EE6677", "edge": "#A63D4A"},
    "removal_100": {"face": "#CCBB44", "edge": "#8A7D2E"},
    "baseline_line": "#444444",
}

FULL_CLASS_NAMES = [
    f"{name} Hand" if name in ("Left", "Right") else name for name in CLASS_NAMES
]


def _save(fig, save_dir, name):
    path = Path(save_dir) / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", format="pdf")
    print(f"Saved: {path}")


def _style_ax(ax):
    """Standard axis styling applied consistently across all plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.tick_params(length=3, width=0.8)


def _legend(ax, **kwargs):
    """Consistent legend styling across all plots."""
    ax.legend(
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        fancybox=False,
        fontsize=8,
        **kwargs,
    )


def _bar(ax, x, heights, width, palette_key, label, zorder=3):
    """Draw a bar with matched face and edge colour."""
    return ax.bar(
        x,
        heights,
        width,
        label=label,
        color=PALETTE[palette_key]["face"],
        edgecolor=PALETTE[palette_key]["edge"],
        linewidth=0.8,
        zorder=zorder,
    )


def _removal_palette_key(idx):
    keys = ["removal_50", "removal_100"]
    return keys[min(idx, len(keys) - 1)]


def _plot_baseline(results, save_dir):
    balanced_acc = results["balanced"]["overall"]
    balanced_class_accs = results["balanced"]["per_class"]

    fig, ax = plt.subplots(figsize=(4.0, 2.8), constrained_layout=True)

    x = np.arange(len(FULL_CLASS_NAMES))
    bars = _bar(ax, x, balanced_class_accs, 0.6, "balanced", label=None)

    ax.axhline(
        y=balanced_acc,
        color=PALETTE["baseline_line"],
        linestyle="--",
        linewidth=1.0,
        label=f"Overall: {balanced_acc:.1%}",
    )

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(FULL_CLASS_NAMES)
    _style_ax(ax)
    _legend(ax, loc="lower right")

    for bar, acc in zip(bars, balanced_class_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{acc:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    _save(fig, save_dir, "baseline_performance")
    plt.close()


def _plot_class_performance(results, save_dir):
    balanced_class_accs = results["balanced"]["per_class"]
    tested_classes = sorted(results["imbalanced"].keys())

    if not tested_classes:
        return

    removal_percentages = sorted(results["imbalanced"][tested_classes[0]].keys())

    n = len(tested_classes)
    ncols = min(n, 2)
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows), constrained_layout=True
    )

    # Normalise axes to always be iterable
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, class_idx in enumerate(tested_classes):
        ax = axes[idx]

        x_vals = [pct * 100 for pct in removal_percentages]
        class_accs = [
            results["imbalanced"][class_idx][pct]["per_class"][class_idx]
            for pct in removal_percentages
        ]

        ax.plot(
            x_vals,
            class_accs,
            marker="o",
            linewidth=1.2,
            markersize=5,
            color=PALETTE["balanced"]["face"],
            markeredgecolor=PALETTE["balanced"]["edge"],
            markeredgewidth=0.8,
            zorder=3,
        )

        ax.axhline(
            y=balanced_class_accs[class_idx],
            color=PALETTE["baseline_line"],
            linestyle="--",
            linewidth=1.0,
            zorder=2,
            label="Balanced baseline",
        )

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{int(v)}%" for v in x_vals])
        ax.set_xlabel("Training Data Removed")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            FULL_CLASS_NAMES[class_idx], fontsize=9, fontweight="normal", pad=8
        )
        _style_ax(ax)
        _legend(ax, loc="lower left")

    # Hide any unused subplots
    for idx in range(len(tested_classes), len(axes)):
        axes[idx].set_visible(False)

    _save(fig, save_dir, "individual_class_performance")
    plt.close()


def _plot_impact_matrix(results, save_dir):
    balanced_class_accs = results["balanced"]["per_class"]
    tested_classes = sorted(results["imbalanced"].keys())

    removal_pct = sorted(results["imbalanced"][tested_classes[0]].keys())[0]

    # Rows = affected class, columns = removed class
    impact_matrix = np.zeros((4, 4))
    for removed_class in tested_classes:
        class_accs = results["imbalanced"][removed_class][removal_pct]["per_class"]
        for affected_class in range(4):
            drop = (
                balanced_class_accs[affected_class] - class_accs[affected_class]
            ) * 100
            impact_matrix[affected_class, removed_class] = drop

    fig, ax = plt.subplots(figsize=(3.8, 3.2), constrained_layout=True)

    im = ax.imshow(
        impact_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=np.max(impact_matrix)
    )

    ax.set_xticks(range(4))
    ax.set_xticklabels(FULL_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(FULL_CLASS_NAMES)
    ax.set_xlabel(f"Class Reduced ({int(removal_pct * 100)}%)")
    ax.set_ylabel("Affected Class")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy Drop (pp)", rotation=270, labelpad=14)
    cbar.outline.set_linewidth(0.6)

    for i in range(4):
        for j in range(4):
            value = impact_matrix[i, j]
            color = "white" if value > np.max(impact_matrix) / 2 else "#333333"
            ax.text(
                j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=7
            )

    _save(fig, save_dir, "cross_class_impact_matrix")
    plt.close()


def _plot_summary(results, save_dir):
    balanced_class_accs = results["balanced"]["per_class"]
    tested_classes = sorted(results["imbalanced"].keys())

    if len(tested_classes) < 2:
        return

    removal_percentages = sorted(results["imbalanced"][tested_classes[0]].keys())

    if not removal_percentages:
        return

    fig, ax = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)

    n_conditions = len(removal_percentages) + 1
    bar_width = 0.7 / n_conditions
    x = np.arange(len(tested_classes))

    # Centre the group of bars around each tick
    offsets = np.linspace(
        -(n_conditions - 1) / 2 * bar_width,
        (n_conditions - 1) / 2 * bar_width,
        n_conditions,
    )

    baseline_accs = [balanced_class_accs[c] for c in tested_classes]
    _bar(ax, x + offsets[0], baseline_accs, bar_width, "balanced", "Balanced")

    for idx, pct in enumerate(removal_percentages):
        imbalanced_accs = [
            results["imbalanced"][c][pct]["per_class"][c] for c in tested_classes
        ]
        key = _removal_palette_key(idx)
        _bar(
            ax,
            x + offsets[idx + 1],
            imbalanced_accs,
            bar_width,
            key,
            f"{int(pct * 100)}% removed",
        )

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([FULL_CLASS_NAMES[c] for c in tested_classes])
    _style_ax(ax)
    _legend(ax, loc="upper right")

    _save(fig, save_dir, "summary_comparison")
    plt.close()


def plot_results(results, save_dir="plots"):
    """
    Generate and save all experiment plots as PDFs.

    Individual plots are skipped gracefully if the results dict does
    not contain sufficient data to render them meaningfully.

    Args:
        results (dict): Results dict from run_experiments
        save_dir (str): Directory to save plots. Default: "plots"
    """
    Path(save_dir).mkdir(exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)

    _plot_baseline(results, save_dir)
    _plot_class_performance(results, save_dir)
    _plot_impact_matrix(results, save_dir)
    _plot_summary(results, save_dir)
