from pathlib import Path
from utils.data_utils import load_data
from utils.classifier_utils import train_model

X, y = load_data(Path("data"))
best_acc, class_accs = train_model(X, y, verbose=True)