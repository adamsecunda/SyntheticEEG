from pathlib import Path
import json
from utils.plot_utils import plot_results

def load_results(save_dir="results"):
    results = {"balanced": {}, "imbalanced": {}}

    balanced_path = Path(save_dir) / "balanced.json"
    if balanced_path.exists():
        with open(balanced_path) as f:
            results["balanced"] = json.load(f)

    for path in Path(save_dir).glob("imbalanced_*.json"):
        parts = path.stem.split("_")
        target_class = int(parts[1])
        removal_pct = int(parts[2]) / 100

        if target_class not in results["imbalanced"]:
            results["imbalanced"][target_class] = {}

        with open(path) as f:
            results["imbalanced"][target_class][removal_pct] = json.load(f)

    return results

results = load_results()
plot_results(results)