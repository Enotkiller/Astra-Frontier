import os
from pathlib import Path

os.environ.setdefault("PROJ_IGNORE_CELESTIAL_BODY", "YES")
os.environ.setdefault("GDAL_CACHEMAX", "256")

ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUT_DIR = ASSETS_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_THERMAL_TIF = ASSETS_DIR / "earth_static" / "wc2.1_2.5m_bio_2.tif"
EARTH_ARIDITY_TIF = ASSETS_DIR / "earth_static" / "wc2.1_2.5m_bio_12.tif"

MOON_ALBEDO_XML = ASSETS_DIR / "moon" / "ldam_50s_1000m_float.xml"
MOON_DIVINER_NIGHT_LBL = ASSETS_DIR / "moon" / "pcp_avg_tbol_pols_sum_ltim01_240.lbl"
MOON_DIVINER_NOON_LBL = ASSETS_DIR / "moon" / "pcp_avg_tbol_pols_sum_ltim49_240.lbl"
MOON_DEM_SOURCES = [
    (ASSETS_DIR / "moon" / "ldem_80s_80m_float.xml", 0.0, 295_000.0, 5000.0),
    (ASSETS_DIR / "moon" / "ldem_60s_120m_float.xml", 295_000.0, 880_000.0, 10000.0),
]

MARS_TES_NIGHT_LBL = (
    ASSETS_DIR / "mars" / "mgst_9001" / "data" / "global_ti_night_2007.lbl"
)
MARS_TES_DAY_LBL = ASSETS_DIR / "mars" / "mgst_9001" / "data" / "global_ti_day_2007.lbl"
MARS_DEM_FILES = [
    ASSETS_DIR / "mars" / "megt90n000gb.lbl",
    ASSETS_DIR / "mars" / "megt90n180gb.lbl",
    ASSETS_DIR / "mars" / "megt00n000gb.lbl",
    ASSETS_DIR / "mars" / "megt00n180gb.lbl",
]

TILE_SIZE_M = 5000.0
PATCH_PX = 64
PIXEL_RES_M = TILE_SIZE_M / PATCH_PX
IN_CHANNELS = 5
LATENT_DIM = 128

EARTH_GRID_STEP_M = 15000.0
BATCH_SIZE = 128
TRAIN_STEPS = 2500
LR = 1e-3
STREAM_WORKERS = 16
QUEUE_MAX_SIZE = 1024

MARS_RADIUS_M = 3396190.0
PC_SUBSCRIPTION_KEY = os.getenv("PC_SUBSCRIPTION_KEY", "")
