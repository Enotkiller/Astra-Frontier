from pathlib import Path

import numpy as np
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

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decoder(z), z


def load_autoencoder(weights_path: Path) -> GeoAutoencoder:
    model = GeoAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def decode_latent_to_3d(model: GeoAutoencoder, vec_128: np.ndarray) -> dict:
    z = torch.from_numpy(vec_128.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        layers = model.decoder(z).squeeze(0).numpy()

    slope_deg = np.clip(layers[0] * 45.0, 0.0, 65.0)
    roughness = np.maximum(0.0, np.expm1(np.clip(layers[1] * 5.0, 0.0, 6.0)))
    rel_clip = np.clip(layers[2], -0.98, 0.98)
    z_relief = np.arctanh(rel_clip) * 150.0
    thermal = np.clip(layers[3], 0.0, 1.0)
    soil = np.clip(layers[4], 0.0, 1.0)

    return {
        "z_relief": np.round(z_relief, 2).tolist(),
        "slope_deg": np.round(slope_deg, 2).tolist(),
        "roughness": np.round(roughness, 2).tolist(),
        "thermal": np.round(thermal, 3).tolist(),
        "soil": np.round(soil, 3).tolist(),
        "source": "GeoAutoencoder Decoder (128D)",
    }
