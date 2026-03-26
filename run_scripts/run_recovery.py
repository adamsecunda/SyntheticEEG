import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.data_utils import load_data
from utils.generative_utils import train_gan
from utils.experiment_utils import run_experiments, run_augmentation_experiments, print_imbalance_results

def main():
    X, y = load_data(PROJECT_ROOT / "data")
    res_path = str(PROJECT_ROOT / "results")
    
    # Load ablation results from disk
    results = run_experiments(X, y, save_dir=res_path)
    
    # Train GAN (70 epochs as per notebook)
    generator = train_gan(X, y, n_epochs=70)
    
    # Ensure models dir exists
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    torch.save(generator.state_dict(), PROJECT_ROOT / "models" / "generator.pt")
    
    # Run the recovery/augmentation
    aug_results = run_augmentation_experiments(X, y, results, generator=generator, save_dir=res_path)
    print_imbalance_results(results, aug_results)

if __name__ == "__main__":
    main()