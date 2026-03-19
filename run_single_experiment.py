from pathlib import Path
from utils.data_utils import load_data
from utils.experiment_utils import run_experiments
from utils.plot_utils import plot_results

X, y = load_data(Path("data"))
results = run_experiments(X, y, target_classes=[0], removal_percentages=[0.5])
plot_results(results)