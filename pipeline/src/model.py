import torch
import torch.nn as nn

from src.config import IN_CHANNELS, LATENT_DIM


class GeoAutoencoder(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.SiLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.encoder(x), p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z
