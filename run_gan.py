from pathlib import Path
from utils.data_utils import load_data
from utils.generative_utils import train_gan, generate_samples
from utils.device import device

print(f"Using device: {device}")

X, y = load_data(Path("data"))
generator = train_gan(X, y, n_epochs=30, verbose=True)

samples = generate_samples(generator, class_label=0, n_samples=4)
print(f"\nGenerated samples shape: {samples.shape}")
print(f"Value range: [{samples.min():.3f}, {samples.max():.3f}]")
print(f"Mean: {samples.mean():.3f}")
print(f"Std: {samples.std():.3f}")