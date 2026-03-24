import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import EEGDataset
from utils.device import device

NOISE_DIM    = 100
EMBED_DIM    = 32
N_CLASSES    = 4
N_CHANNELS   = 22
N_TIMES      = 1001

LAMBDA_GP    = 10
LAMBDA_STATS = 10
LAMBDA_PSD = 0.01
N_CRITIC     = 5


class Generator(nn.Module):
    """
    Conditional generator for EEG motor imagery epochs.

    Takes a noise vector and class label and produces a synthetic
    22x1001 EEG epoch. The noise and class embedding are projected
    to an initial feature map of 256x1x7 and progressively upsampled
    through four transposed convolutional blocks to 22x1001.

    Args:
        noise_dim (int): Dimension of the input noise vector. Default: 100
        embed_dim (int): Dimension of the class embedding. Default: 32
        n_classes (int): Number of motor imagery classes. Default: 4
    """

    def __init__(self, noise_dim=NOISE_DIM, embed_dim=EMBED_DIM, n_classes=N_CLASSES):
        super().__init__()

        self.embed   = nn.Embedding(n_classes, embed_dim)
        self.project = nn.Linear(noise_dim + embed_dim, 256 * 7)
        self.drop = nn.Dropout(p=0.3)

        # Block 1: 256 x 7 -> 128 x 32
        self.conv1 = nn.ConvTranspose1d(256, 128, kernel_size=8,  stride=4, padding=0)
        self.bn1   = nn.BatchNorm1d(128)

        # Block 2: 128 x 32 -> 64 x 107
        self.conv2 = nn.ConvTranspose1d(128, 64,  kernel_size=14, stride=3, padding=0)
        self.bn2   = nn.BatchNorm1d(64)

        # Block 3: 64 x 107 -> 32 x 330
        self.conv3 = nn.ConvTranspose1d(64,  32,  kernel_size=12, stride=3, padding=0)
        self.bn3   = nn.BatchNorm1d(32)

        # Block 4: 32 x 330 -> 22 x 1001
        self.conv4 = nn.ConvTranspose1d(32, N_CHANNELS, kernel_size=14, stride=3, padding=0)

    def forward(self, z, labels):
        e = self.embed(labels)
        
        if self.training:
            e = e + torch.randn_like(e) * 0.1
        
        x = torch.relu(self.project(torch.cat([z, e], dim=1)))
        x = x.view(-1, 256, 7)

        x = self.drop(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop(torch.relu(self.bn3(self.conv3(x))))
        x = torch.tanh(self.conv4(x))

        return x[:, :, :N_TIMES]


class Critic(nn.Module):
    """
    Conditional critic for EEG motor imagery epochs.

    Uses dilated convolutions to capture multi-scale temporal structure,
    deliberately distinct from the classifier architecture to avoid the
    generator learning to fool a classifier-like discriminator.

    Args:
        embed_dim (int): Dimension of the class embedding. Default: 32
        n_classes (int): Number of motor imagery classes. Default: 4
        n_channels (int): Number of EEG channels. Default: 22
    """

    def __init__(self, embed_dim=EMBED_DIM, n_classes=N_CLASSES, n_channels=N_CHANNELS):
        super().__init__()

        self.embed      = nn.Embedding(n_classes, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, N_TIMES)

        # Dilated convolutions capture different temporal scales
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels + 1, 32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, dilation=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, dilation=4, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, dilation=8, padding=8),
            nn.LeakyReLU(0.2),
        )

        with torch.no_grad():
            x            = torch.zeros(1, n_channels + 1, N_TIMES)
            flatten_size = self.conv(x).numel()

        self.fc = nn.Linear(flatten_size, 1)

    def forward(self, x, labels):
        e     = self.embed(labels)
        e_map = self.embed_proj(e).unsqueeze(1)
        x     = torch.cat([x, e_map], dim=1)
        x     = self.conv(x)
        return self.fc(x.flatten(1))


def _gradient_penalty(critic, real_x, fake_x, labels):
    """
    Compute the gradient penalty for WGAN-GP.

    Interpolates between real and fake samples and penalises the critic
    if the gradient norm deviates from 1, enforcing the Lipschitz constraint.

    Args:
        critic (Critic): The critic model
        real_x (Tensor): Real EEG epochs of shape (batch, 22, 1001)
        fake_x (Tensor): Fake EEG epochs of shape (batch, 22, 1001)
        labels (Tensor): Class labels of shape (batch,)

    Returns:
        penalty (Tensor): Scalar gradient penalty
    """
    batch_size   = real_x.size(0)
    alpha        = torch.rand(batch_size, 1, 1).to(device)
    interpolated = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    score        = critic(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    penalty = ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def _stats_loss(fake_x, real_x):
    """
    Penalise the generator if its outputs do not match the per-channel
    mean and standard deviation of the real data.

    Args:
        fake_x (Tensor): Fake EEG epochs of shape (batch, 22, 1001)
        real_x (Tensor): Real EEG epochs of shape (batch, 22, 1001)

    Returns:
        loss (Tensor): Scalar statistics matching loss
    """
    fake_mean = fake_x.mean(dim=[0, 2])
    real_mean = real_x.mean(dim=[0, 2])
    fake_std  = fake_x.std(dim=[0, 2])
    real_std  = real_x.std(dim=[0, 2])
    return F.mse_loss(fake_mean, real_mean) + F.mse_loss(fake_std, real_std)


def _psd_loss(fake_x, real_x):
    """
    Penalise the generator if its outputs do not match the power spectral
    density of the real data. This encourages the generator to produce
    signals with realistic frequency content in the mu and beta bands.

    Args:
        fake_x (Tensor): Fake EEG epochs of shape (batch, 22, 1001)
        real_x (Tensor): Real EEG epochs of shape (batch, 22, 1001)

    Returns:
        loss (Tensor): Scalar PSD matching loss
    """
    fake_psd = torch.fft.rfft(fake_x, dim=2).abs().mean(dim=0)
    real_psd = torch.fft.rfft(real_x, dim=2).abs().mean(dim=0)
    return F.mse_loss(fake_psd, real_psd)


def train_gan(X, y, n_epochs=200, lr_g=0.00001, lr_d=0.0001, verbose=True):
    """
    Train a conditional WGAN-GP on EEG epochs.

    The critic is updated N_CRITIC times per generator update. A gradient
    penalty enforces the Lipschitz constraint on the critic. Statistics
    and PSD matching losses encourage the generator to produce outputs
    with realistic statistical and frequency properties.

    Args:
        X (np.ndarray): EEG epochs of shape (n_epochs, 22, 1001)
        y (np.ndarray): Integer class labels of shape (n_epochs,)
        n_epochs (int): Number of training epochs. Default: 200
        lr_g (float): Generator learning rate. Default: 1e-5
        lr_d (float): Critic learning rate. Default: 1e-5
        verbose (bool): Print training progress. Default: True

    Returns:
        generator (Generator): Trained generator model
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == "mps":
        torch.mps.manual_seed(42)
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif device.type == "cuda":
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    dataset = EEGDataset(X, y)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True,
                         generator=torch.Generator().manual_seed(42))

    generator = Generator().to(device)
    critic    = Critic().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    c_optimizer = torch.optim.Adam(critic.parameters(),    lr=lr_d, betas=(0.0, 0.9))

    for epoch in range(1, n_epochs + 1):
        g_loss_total     = 0
        c_loss_total     = 0
        stats_loss_total = 0
        psd_loss_total   = 0

        for real_x, real_y in loader:
            real_x     = real_x.to(device)
            real_y     = real_y.to(device)
            batch_size = real_x.size(0)

            # Train critic N_CRITIC times per generator update
            for _ in range(N_CRITIC):
                z      = torch.randn(batch_size, NOISE_DIM).to(device)
                fake_x = generator(z, real_y).detach()

                gp     = _gradient_penalty(critic, real_x, fake_x, real_y)
                c_loss = critic(fake_x, real_y).mean() - critic(real_x, real_y).mean() + LAMBDA_GP * gp

                c_optimizer.zero_grad()
                c_loss.backward()
                c_optimizer.step()

            # Train generator with adversarial + stats + PSD loss
            z          = torch.randn(batch_size, NOISE_DIM).to(device)
            fake_x     = generator(z, real_y)
            adv_loss   = -critic(fake_x, real_y).mean()
            stats_loss = _stats_loss(fake_x, real_x)
            psd_loss   = _psd_loss(fake_x, real_x)
            g_loss     = adv_loss + LAMBDA_STATS * stats_loss + LAMBDA_PSD * psd_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_loss_total     += adv_loss.item()
            c_loss_total     += c_loss.item()
            stats_loss_total += stats_loss.item()
            psd_loss_total   += psd_loss.item()

        if verbose and epoch % 10 == 0:
            n_batches = len(loader)
            print(f"\n[Epoch {epoch}]")
            print(f"Generator loss - {g_loss_total / n_batches:.4f}")
            print(f"Critic loss    - {c_loss_total / n_batches:.4f}")
            print(f"Stats loss     - {stats_loss_total / n_batches:.4f}")
            print(f"PSD loss       - {psd_loss_total / n_batches:.4f}")

    if verbose:
        print("\nTraining complete")

    return generator


def generate_samples(generator, class_label, n_samples):
    """
    Generate synthetic EEG epochs for a given motor imagery class.

    Args:
        generator (Generator): Trained generator model
        class_label (int): Class index to generate (0-3)
        n_samples (int): Number of epochs to generate

    Returns:
        samples (np.ndarray): Synthetic epochs of shape (n_samples, 22, 1001)
    """
    generator.eval()
    with torch.no_grad():
        z       = torch.randn(n_samples, NOISE_DIM).to(device)
        labels  = torch.full((n_samples,), class_label, dtype=torch.long).to(device)
        samples = generator(z, labels)
    return samples.cpu().numpy()