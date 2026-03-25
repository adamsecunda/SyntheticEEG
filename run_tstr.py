from pathlib import Path
import numpy as np
from utils.data_utils import load_data
from utils.generative_utils import train_gan, generate_samples
from utils.classifier_utils import train_model, CLASS_NAMES
from utils.device import device

print(f"Using device: {device}")

X, y = load_data(Path("data"))

print("\nTraining GAN...")
generator = train_gan(X, y, n_epochs=70, verbose=True)

print("\nGenerating synthetic dataset...")
n_per_class = len(y) // 4
X_syn_list, y_syn_list = [], []
for class_label in range(4):
    samples = generate_samples(generator, class_label, n_per_class)
    X_syn_list.append(samples)
    y_syn_list.append(np.full(n_per_class, class_label))

X_syn = np.concatenate(X_syn_list, axis=0).astype("float32")
y_syn = np.concatenate(y_syn_list, axis=0)

print(f"Synthetic dataset shape: {X_syn.shape}")

print("\nTSTR evaluation — training classifier on synthetic data...")
best_acc, class_accs = train_model(X_syn, y_syn, verbose=True)

print(f"\nTSTR Results")
print(f"Overall accuracy - {best_acc:.3f}")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name} - {class_accs[i]:.3f}")