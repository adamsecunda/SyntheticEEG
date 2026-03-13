import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utils.data_utils import EEGDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Left", "Right", "Feet", "Tongue"]


class DeepConvNet(nn.Module):
    """
    Deep ConvNet for EEG motor imagery classification.

    Block 1 applies a temporal then spatial convolution, collapsing the channel
    dimension. Blocks 2-4 apply temporal convolutions only with increasing
    filter counts (50, 100, 200).

    Args:
        n_channels (int): Number of EEG channels. Default: 22
        n_timepoints (int): Number of timepoints per epoch. Default: 1001
        n_classes (int): Number of motor imagery classes. Default: 4
    """

    def __init__(self, n_channels=22, n_timepoints=1001, n_classes=4):
        super().__init__()

        # Block 1: temporal + spatial convolution
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 10))
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(n_channels, 1))
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Block 2
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 10))
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Block 3
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 10))
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Block 4
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 10))
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Infer flattened size with a dummy forward pass
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.pool1(F.elu(self.bn1(self.conv2(self.conv1(x)))))
            x = self.pool2(F.elu(self.bn2(self.conv3(x))))
            x = self.pool3(F.elu(self.bn3(self.conv4(x))))
            x = self.pool4(F.elu(self.bn4(self.conv5(x))))
            flatten_size = x.numel()

        self.fc = nn.Linear(flatten_size, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Block 4
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = F.dropout(x, p=0.5, training=self.training)

        return self.fc(x.flatten(1))


def train_model(X_train, y_train, n_epochs=150, lr=0.0005, verbose=True):
    """
    Train a DeepConvNet on the provided EEG data.

    Args:
        X_train (np.ndarray): EEG epochs of shape (n_epochs, 22, 1001)
        y_train (np.ndarray): Integer class labels of shape (n_epochs,)
        n_epochs (int): Maximum number of training epochs. Default: 150
        lr (float): Initial learning rate. Default: 0.0005
        verbose (bool): Print training progress. Default: True

    Returns:
        best_acc (float): Best validation accuracy achieved during training
        class_accs (list[float]): Per-class accuracy at the best epoch
    """
    dataset = EEGDataset(X_train, y_train)

    n_train = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [n_train, len(dataset) - n_train])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    model = DeepConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):

        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        if epoch % 5 != 0:  # evaluate every 5 epochs
            continue

        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 4
        class_total = [0] * 4

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(1)
                correct += (preds == y).sum().item()
                total += len(y)
                for c in range(4):
                    mask = y == c
                    if mask.sum() > 0:
                        class_correct[c] += (preds[mask] == y[mask]).sum().item()
                        class_total[c] += mask.sum().item()

        acc = correct / total
        class_accs = [
            class_correct[c] / class_total[c] if class_total[c] > 0 else 0
            for c in range(4)
        ]
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            per_class = "  ".join(
                f"{name} {class_accs[i]:.3f}" for i, name in enumerate(CLASS_NAMES)
            )
            suffix = "  (best)" if acc == best_acc else ""
            print(f"\n[Epoch {epoch}]  Accuracy {acc:.3f}{suffix}  {per_class}")

        if patience_counter >= 7:  # early stopping after 35 epochs without improvement
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
            break

    if verbose:
        print("\nTraining complete")
        print(f"Best accuracy - {best_acc:.3f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name} - {class_accs[i]:.3f}")

    return best_acc, class_accs
