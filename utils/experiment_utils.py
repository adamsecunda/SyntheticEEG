import json
import numpy as np
import torch
from pathlib import Path
from utils.data_utils import create_imbalanced_dataset
from utils.classifier_utils import train_model, CLASS_NAMES
from utils.generative_utils import generate_samples


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


def run_tstr(generator, y, n_per_class=None):
    if n_per_class is None:
        n_per_class = len(y) // 4

    X_syn_list, y_syn_list = [], []
    for class_label in range(4):
        samples = generate_samples(generator, class_label, n_per_class)
        X_syn_list.append(samples)
        y_syn_list.append(np.full(n_per_class, class_label))

    X_syn = np.concatenate(X_syn_list, axis=0).astype('float32')
    y_syn = np.concatenate(y_syn_list, axis=0)

    print('TSTR — training classifier on synthetic data...')
    best_acc, class_accs = train_model(X_syn, y_syn, verbose=False)

    print(f'\nTSTR Results')
    print(f'Overall accuracy - {best_acc:.3f}')
    for i, name in enumerate(CLASS_NAMES):
        print(f'{name} - {class_accs[i]:.3f}')

    return best_acc, class_accs


def run_augmentation_experiments(X, y, results, generator, save_dir='results'):
    aug_results = {}
    balanced_class_counts = {c: int((y == c).sum()) for c in range(4)}

    for target_class in sorted(results['imbalanced'].keys()):
        aug_results[target_class] = {}

        for removal_pct in sorted(results['imbalanced'][target_class].keys()):
            tag         = f'augmented_{target_class}_{int(float(removal_pct) * 100)}'
            config_path = Path(save_dir) / f'{tag}.json'

            if config_path.exists():
                print(f'\n{CLASS_NAMES[target_class]} - {int(float(removal_pct) * 100)}% removed - loading saved results')
                with open(config_path) as f:
                    aug_results[target_class][removal_pct] = json.load(f)
                continue

            print(f'\n{CLASS_NAMES[target_class]} - {int(float(removal_pct) * 100)}% removed - augmenting...')

            X_imb, y_imb = create_imbalanced_dataset(X, y, target_class, float(removal_pct))

            n_removed = balanced_class_counts[target_class] - int(balanced_class_counts[target_class] * (1 - float(removal_pct)))
            X_syn     = generate_samples(generator, target_class, n_removed)
            y_syn     = np.full(n_removed, target_class)

            X_aug = np.concatenate([X_imb, X_syn], axis=0).astype('float32')
            y_aug = np.concatenate([y_imb, y_syn], axis=0)

            acc, class_accs = train_model(X_aug, y_aug, verbose=False)

            config_result = {'overall': acc, 'per_class': class_accs}
            aug_results[target_class][removal_pct] = config_result

            with open(config_path, 'w') as f:
                json.dump(config_result, f, indent=2)

    return aug_results
def print_imbalance_results(results, aug_results=None, class_names=None):
    if class_names is None:
        class_names = ["Left", "Right", "Feet", "Tongue"]

    header = f"{'Configuration':<25} {'Overall':>8} {'Left':>7} {'Right':>7} {'Feet':>7} {'Tongue':>7} {'Delta':>9}"
    print(header)
    print("-" * len(header))

    # Balanced Baseline (A_base)
    b = results["balanced"]
    print(f"{'Balanced Baseline':<25} {b['overall']:>8.3f} "
          f"{b['per_class'][0]:>7.3f} {b['per_class'][1]:>7.3f} "
          f"{b['per_class'][2]:>7.3f} {b['per_class'][3]:>7.3f} {'-':>9}")

    for cls_idx in sorted(results["imbalanced"].keys()):
        for pct in sorted(results["imbalanced"][cls_idx].keys()):
            
            # Imbalance Impact: Change after removing data (A_imb - A_base)
            imb = results["imbalanced"][cls_idx][pct]
            d_imb = (imb['per_class'][cls_idx] - b['per_class'][cls_idx]) * 100
            
            tag_imb = f"{class_names[cls_idx]} {int(float(pct)*100)}% Imbalance"
            print(f"{tag_imb:<25} {imb['overall']:>8.3f} "
                  f"{imb['per_class'][0]:>7.3f} {imb['per_class'][1]:>7.3f} "
                  f"{imb['per_class'][2]:>7.3f} {imb['per_class'][3]:>7.3f} {d_imb:>8.1f}pp")

            if aug_results and cls_idx in aug_results and pct in aug_results[cls_idx]:
                # Augmented State (A_aug)
                aug = aug_results[cls_idx][pct]
                
                # Net Change: Final difference vs original baseline (A_aug - A_base)
                d_net = (aug['per_class'][cls_idx] - b['per_class'][cls_idx]) * 100
                # Augmentation Gain: Change after adding synthetic data (A_aug - A_imb)
                d_util = (aug['per_class'][cls_idx] - imb['per_class'][cls_idx]) * 100
                
                tag_aug = f"{class_names[cls_idx]} {int(float(pct)*100)}% Augmented"
                print(f"{tag_aug:<25} {aug['overall']:>8.3f} "
                      f"{aug['per_class'][0]:>7.3f} {aug['per_class'][1]:>7.3f} "
                      f"{aug['per_class'][2]:>7.3f} {aug['per_class'][3]:>7.3f} {d_net:>8.1f}pp")
                
                # Report the recovery amount from the GAN
                print(f"{'Augmentation Gain':<25} {'':>8} {'':>7} {'':>7} {'':>7} {'':>7} {d_util:>+8.1f}pp")
            
            print("-" * len(header))