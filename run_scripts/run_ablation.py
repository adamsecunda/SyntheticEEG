import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.data_utils import load_data
from utils.experiment_utils import run_experiments, print_imbalance_results

def main():
    X, y = load_data(PROJECT_ROOT / "data")
    results = run_experiments(X, y, save_dir=str(PROJECT_ROOT / "results"))
    print_imbalance_results(results)

if __name__ == "__main__":
    main()