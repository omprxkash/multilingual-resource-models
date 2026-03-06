"""Denoising Autoencoder and Variational Autoencoder for embedding augmentation.

Both architectures operate in AfriBERTa's 768-dimensional embedding space,
compressing embeddings to a latent space and reconstructing them.  Used by
the LiDA augmentation pipeline from Paper 2 (MAGE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    """DAE: 768 → 32 (latent) → 768.

    LeakyReLU activations, BatchNorm1d, Dropout(0.2).
    Trained to reconstruct clean embeddings from noisy input — the noise
    is the LiDA perturbation from the augmentation module.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class VariationalAutoencoder(nn.Module):
    """VAE: 768 → 512 → 256 (mu, logvar) → 512 → 768.

    Uses the reparameterization trick for backpropagation through the
    stochastic latent space.  BatchNorm, ReLU, Dropout(0.2) throughout.
    Enables generation of diverse augmented embeddings around each original.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """ELBO loss: MSE reconstruction + beta-weighted KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_autoencoder(
    model,
    embeddings: torch.Tensor,
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    is_vae: bool = False,
) -> list:
    """Train DAE or VAE on a fixed embedding tensor. Returns per-epoch losses."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(embeddings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            if is_vae:
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar)
            else:
                recon = model(x)
                loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
    return losses
