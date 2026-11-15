import os
from pathlib import Path

os.environ.setdefault("PROJ_IGNORE_CELESTIAL_BODY", "YES")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ASSETS_DIR / "data"
MOON_DIR = ASSETS_DIR / "moon"
MARS_DIR = ASSETS_DIR / "mars"
FRONTEND_DIR = ROOT_DIR / "frontend"

PATCH_PX = 64
TILE_SIZE_M = 5000.0
PIXEL_RES_M = TILE_SIZE_M / PATCH_PX
IN_CHANNELS = 5
LATENT_DIM = 128
