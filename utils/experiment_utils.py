import json
import numpy as np
import torch
from pathlib import Path
from utils.data_utils import create_imbalanced_dataset
from utils.classifier_utils import train_model, CLASS_NAMES


def _results_path(save_dir, tag):
    return Path(save_dir) / f"{tag}.json"


def _save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _load(path):
    with open(path) as f:
        return json.load(f)


def run_experiments(X, y, removal_percentages=[0.5, 1.0], target_classes=[0, 1, 2, 3], save_dir="results"):
    """
    Run the full imbalance experiment pipeline.

    Trains the classifier on the balanced dataset to establish a baseline,
    then retrains from scratch for each combination of target class and
    removal percentage. Results are saved after each configuration so
    the experiment can be interrupted and resumed.

    Args:
        X (np.ndarray): EEG epochs of shape (n_epochs, 22, 1001)
        y (np.ndarray): Integer class labels of shape (n_epochs,)
        removal_percentages (list[float]): Proportions of target class to remove
        target_classes (list[int]): Class indices to undersample (0-3)
        save_dir (str): Directory to save results. Default: "results"

    Returns:
        results (dict): Nested dict containing overall and per-class accuracy
                        for the balanced baseline and each imbalanced configuration
    """
    Path(save_dir).mkdir(exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    results = {'balanced': {}, 'imbalanced': {}}

    # Balanced baseline
    baseline_path = _results_path(save_dir, "balanced")
    if baseline_path.exists():
        print("\nBalanced baseline - loading saved results")
        results['balanced'] = _load(baseline_path)
    else:
        print("\nBalanced baseline")
        balanced_acc, balanced_class_accs = train_model(X, y)
        results['balanced'] = {'overall': balanced_acc, 'per_class': balanced_class_accs}
        _save(results['balanced'], baseline_path)

    # Imbalanced configurations
    for target_class in target_classes:
        results['imbalanced'][target_class] = {}

        for removal_pct in removal_percentages:
            tag = f"imbalanced_{target_class}_{int(removal_pct * 100)}"
            config_path = _results_path(save_dir, tag)

            if config_path.exists():
                print(f"\n{CLASS_NAMES[target_class]} - {int(removal_pct * 100)}% removed - loading saved results")
                results['imbalanced'][target_class][removal_pct] = _load(config_path)
                continue

            print(f"\n{CLASS_NAMES[target_class]} - {int(removal_pct * 100)}% removed")
            X_imb, y_imb = create_imbalanced_dataset(X, y, target_class, removal_pct)
            imb_acc, imb_class_accs = train_model(X_imb, y_imb)

            config_results = {'overall': imb_acc, 'per_class': imb_class_accs}
            results['imbalanced'][target_class][removal_pct] = config_results
            _save(config_results, config_path)

    return results