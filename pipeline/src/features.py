import numpy as np
from scipy.ndimage import uniform_filter

from src.config import PATCH_PX, PIXEL_RES_M


def dem_to_feature_tensor(
    dem_64x64: np.ndarray,
    thermal_swing: float,
    soil_surface: float,
) -> tuple[np.ndarray, dict] | None:
    if dem_64x64.shape != (PATCH_PX, PATCH_PX):
        return None
    if np.isnan(dem_64x64).any() or np.isinf(dem_64x64).any():
        return None
    if np.std(dem_64x64) < 0.05:
        return None

    dem = dem_64x64.astype(np.float32)

    dy, dx = np.gradient(dem, PIXEL_RES_M)
    slope_deg = np.degrees(np.arctan(np.hypot(dx, dy)))

    mean_h = uniform_filter(dem, size=3, mode="nearest")
    mean_h2 = uniform_filter(dem * dem, size=3, mode="nearest")
    roughness = np.sqrt(np.maximum(0.0, mean_h2 - mean_h * mean_h))

    rel_elevation = dem - float(np.mean(dem))

    ch_slope = np.clip(slope_deg / 45.0, 0.0, 1.5)
    ch_rough = np.log1p(roughness) / 5.0
    ch_relief = np.tanh(rel_elevation / 150.0)
    ch_thermal = np.full((PATCH_PX, PATCH_PX), float(thermal_swing), dtype=np.float32)
    ch_soil = np.full((PATCH_PX, PATCH_PX), float(soil_surface), dtype=np.float32)

    tensor = np.stack(
        [ch_slope, ch_rough, ch_relief, ch_thermal, ch_soil], axis=0
    ).astype(np.float32)

    stats = {
        "mean_slope": float(np.mean(slope_deg)),
        "max_slope": float(np.max(slope_deg)),
        "mean_roughness": float(np.mean(roughness)),
        "relief_std": float(np.std(rel_elevation)),
        "thermal_swing_idx": float(thermal_swing),
        "surface_soil_idx": float(soil_surface),
    }
    return tensor, stats
