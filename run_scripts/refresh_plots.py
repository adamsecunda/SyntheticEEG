import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.plot_utils import plot_results

def main():
    res_dir = PROJECT_ROOT / "results"
    results = {"balanced": None, "imbalanced": {}}
    aug_results = {}

    with open(res_dir / "balanced.json") as f:
        results["balanced"] = json.load(f)

    # Dynamic scan for imbalanced and augmented files
    for p in res_dir.glob("*.json"):
        parts = p.stem.split("_")
        if len(parts) != 3: continue
        
        etype, c_idx, pct = parts[0], int(parts[1]), float(parts[2])/100.0
        
        with open(p) as f:
            data = json.load(f)
            
        if etype == "imbalanced":
            results["imbalanced"].setdefault(c_idx, {})[pct] = data
        elif etype == "augmented":
            aug_results.setdefault(c_idx, {})[pct] = data

    plot_results(results, aug_results, save_dir=str(PROJECT_ROOT / "plots"))

if __name__ == "__main__":
    main()