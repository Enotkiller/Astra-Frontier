import math
import urllib.request

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.io import MemoryFile
from scipy.ndimage import uniform_filter, zoom

from src.config import MARS_DIR, MOON_DIR, PATCH_PX, PIXEL_RES_M, TILE_SIZE_M


def dem_to_3d_payload(
    dem: np.ndarray,
    thermal_val: float,
    soil_val: float,
    source_label: str,
) -> dict:
    dy, dx = np.gradient(dem, PIXEL_RES_M)
    slope_deg = np.degrees(np.arctan(np.hypot(dx, dy)))
    mean_h = uniform_filter(dem, size=3, mode="nearest")
    mean_h2 = uniform_filter(dem * dem, size=3, mode="nearest")
    roughness = np.sqrt(np.maximum(0.0, mean_h2 - mean_h * mean_h))
    rel_height = dem - float(np.mean(dem))

    return {
        "z_relief": np.round(rel_height, 2).tolist(),
        "slope_deg": np.round(slope_deg, 2).tolist(),
        "roughness": np.round(roughness, 2).tolist(),
        "thermal": np.round(
            np.full((PATCH_PX, PATCH_PX), thermal_val, dtype=np.float32), 3
        ).tolist(),
        "soil": np.round(
            np.full((PATCH_PX, PATCH_PX), soil_val, dtype=np.float32), 3
        ).tolist(),
        "source": source_label,
    }


def slice_local_dem(row: pd.Series) -> np.ndarray | None:
    planet = str(row["planet"])
    try:
        if planet == "moon":
            x_m, y_m = float(row["x_m"]), float(row["y_m"])
            r = math.hypot(x_m, y_m)
            path = MOON_DIR / (
                "ldem_80s_80m_float.xml" if r < 295_000.0 else "ldem_60s_120m_float.xml"
            )
            if not path.exists():
                return None

            with rasterio.open(path) as src:
                win = rasterio.windows.from_bounds(
                    x_m - TILE_SIZE_M / 2.0,
                    y_m - TILE_SIZE_M / 2.0,
                    x_m + TILE_SIZE_M / 2.0,
                    y_m + TILE_SIZE_M / 2.0,
                    transform=src.transform,
                )
                patch = src.read(
                    1,
                    window=win,
                    out_shape=(PATCH_PX, PATCH_PX),
                    boundless=True,
                    fill_value=np.nan,
                ).astype(np.float32)
                if src.nodata is not None:
                    patch = np.where(patch == src.nodata, np.nan, patch)
                if np.isnan(patch).any():
                    return None
                if np.percentile(np.abs(patch), 90) < 50.0:
                    patch *= 1000.0
                return patch

        elif planet == "mars":
            lat, lon = float(row["lat"]), float(row["lon"])
            mars_files = [
                MARS_DIR / "megt90n000gb.lbl",
                MARS_DIR / "megt90n180gb.lbl",
                MARS_DIR / "megt00n000gb.lbl",
                MARS_DIR / "megt00n180gb.lbl",
            ]
            for path in mars_files:
                if not path.exists():
                    continue
                with rasterio.open(path) as src:
                    left, bottom, right, top = src.bounds
                    is_proj = (src.crs is not None and not src.crs.is_geographic) or (
                        abs(right - left) > 400.0
                    )
                    if is_proj and src.crs is not None:
                        tf = pyproj.Transformer.from_crs(
                            src.crs.geodetic_crs, src.crs, always_xy=True
                        )
                        cx, cy = tf.transform(lon, lat)
                        hx = hy = TILE_SIZE_M / 2.0
                    else:
                        cx = (lon % 360.0) if right > 180.5 else lon
                        cy = lat
                        hy = (TILE_SIZE_M / 2.0) / 59274.0
                        hx = hy / max(math.cos(math.radians(lat)), 0.15)

                    if left + hx <= cx <= right - hx and bottom + hy <= cy <= top - hy:
                        win = rasterio.windows.from_bounds(
                            cx - hx, cy - hy, cx + hx, cy + hy, transform=src.transform
                        )
                        patch = src.read(
                            1, window=win, out_shape=(PATCH_PX, PATCH_PX)
                        ).astype(np.float32)
                        if src.nodata is not None and np.any(patch == src.nodata):
                            continue
                        return patch
    except Exception:
        return None
    return None


def fetch_earth_dem(lat: float, lon: float) -> np.ndarray | None:
    try:
        z = 11
        n = 2**z
        lat_rad = math.radians(max(-85.0, min(85.0, lat)))
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{xtile}/{ytile}.png"

        req = urllib.request.Request(
            url, headers={"User-Agent": "AstraFrontierApp/1.0"}
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            png_bytes = resp.read()

        with MemoryFile(png_bytes) as memfile:
            with memfile.open() as src:
                r = src.read(1).astype(np.float32)
                g = src.read(2).astype(np.float32)
                b = src.read(3).astype(np.float32)
                elev = (r * 256.0 + g + b / 256.0) - 32768.0

        res_m = (40075016.686 * math.cos(lat_rad)) / (n * 256.0)
        half_px = max(8, min(120, int(round((TILE_SIZE_M / 2.0) / max(res_m, 1.0)))))
        c = 128
        sub = elev[c - half_px : c + half_px, c - half_px : c + half_px]
        return zoom(
            sub, (PATCH_PX / sub.shape[0], PATCH_PX / sub.shape[1]), order=1
        ).astype(np.float32)
    except Exception:
        return None


def fetch_earth_satellite_jpeg(lat: float, lon: float) -> bytes:
    z = 12
    n = 2**z
    lat_rad = math.radians(max(-85.0, min(85.0, lat)))
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ytile}/{xtile}"

    req = urllib.request.Request(url, headers={"User-Agent": "AstraFrontierApp/1.0"})
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        return resp.read()
