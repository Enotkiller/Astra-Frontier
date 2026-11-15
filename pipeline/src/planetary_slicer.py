import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from scipy.ndimage import zoom
from tqdm import tqdm

from src.config import (
    MARS_DEM_FILES,
    MARS_RADIUS_M,
    MOON_DEM_SOURCES,
    PATCH_PX,
    TILE_SIZE_M,
)
from src.features import dem_to_feature_tensor
from src.lookups import LunarPolarLookup, MarsTesLookup
from src.utils import log_step


def load_lunar_tiles(
    lunar_lookup: LunarPolarLookup, max_tiles: int = 20000
) -> tuple[np.ndarray, pd.DataFrame]:
    log_step("[1/5] Slicing Moon maps in RAM (LOLA DEM + Albedo + Diviner Tbol)...")
    rng = np.random.default_rng(42)

    candidates = []
    for path, r_min, r_max, step in MOON_DEM_SOURCES:
        if not path.exists():
            log_step(f"  -> Moon DEM file not found: {path}")
            continue
        coords = np.arange(-r_max, r_max, step)
        for x in coords:
            for y in coords:
                r = math.hypot(x, y)
                if r_min <= r < r_max:
                    candidates.append((path, float(x), float(y)))

    log_step(f"  -> Total lunar candidate patches: {len(candidates)}")
    target_pool = int(max_tiles * 1.15)
    if len(candidates) > target_pool:
        chosen = rng.choice(len(candidates), size=target_pool, replace=False)
        candidates = [candidates[i] for i in chosen]

    by_file: dict[Path, list[tuple[float, float]]] = {}
    for path, x, y in candidates:
        by_file.setdefault(path, []).append((x, y))

    tensors, meta_rows = [], []
    for path, pts in by_file.items():
        log_step(f"  -> Reading {path.name} into RAM ({len(pts)} sample patches)...")
        with rasterio.open(path) as src:
            full_dem = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                full_dem[full_dem == nodata] = np.nan

            valid_sample = full_dem[::32, ::32]
            valid_sample = valid_sample[np.isfinite(valid_sample)]
            height_scale = (
                1000.0
                if (
                    len(valid_sample) > 0
                    and np.percentile(np.abs(valid_sample), 90) < 50.0
                )
                else 1.0
            )
            full_dem *= height_scale

            inv_tf = ~src.transform
            rx = max(2, int(round((TILE_SIZE_M / 2.0) / abs(src.transform.a))))
            ry = max(2, int(round((TILE_SIZE_M / 2.0) / abs(src.transform.e))))
            h, w = full_dem.shape
            transformer = pyproj.Transformer.from_crs(
                src.crs, src.crs.geodetic_crs, always_xy=True
            )

            for x, y in tqdm(pts, desc=f"Slice {path.stem}", leave=False):
                col_c, row_c = inv_tf * (x, y)
                r0, r1 = int(round(row_c)) - ry, int(round(row_c)) + ry
                c0, c1 = int(round(col_c)) - rx, int(round(col_c)) + rx
                if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
                    continue

                sub = full_dem[r0:r1, c0:c1]
                if np.isnan(sub).any():
                    continue

                patch = zoom(
                    sub, (PATCH_PX / sub.shape[0], PATCH_PX / sub.shape[1]), order=1
                )
                th_val, alb_val = lunar_lookup.get_xy(x, y)
                res = dem_to_feature_tensor(patch, th_val, alb_val)
                if res is not None:
                    tensor, stats = res
                    lon, lat = transformer.transform(x, y)
                    tensors.append(tensor)
                    meta_rows.append(
                        {
                            "planet": "moon",
                            "lat": float(lat),
                            "lon": float(lon),
                            "x_m": x,
                            "y_m": y,
                            **stats,
                        }
                    )

        del full_dem
        gc.collect()
        log_step(
            f"  -> Finished {path.name}, accumulated lunar patches: {len(tensors)}"
        )

    if len(tensors) > max_tiles:
        idx = rng.choice(len(tensors), size=max_tiles, replace=False)
        tensors = [tensors[i] for i in idx]
        meta_rows = [meta_rows[i] for i in idx]

    out_arr = np.array(tensors, dtype=np.float32)
    del tensors
    gc.collect()
    log_step(f"  -> Total Moon patches: {len(out_arr)} ({out_arr.nbytes / 1e6:.1f} MB)")
    return out_arr, pd.DataFrame(meta_rows)


def load_mars_tiles(
    mars_lookup: MarsTesLookup, max_tiles: int = 25000
) -> tuple[np.ndarray, pd.DataFrame]:
    log_step("[2/5] Slicing Mars maps in RAM (MOLA DEM + TES Thermal Inertia)...")
    MARS_M_PER_DEG = 59274.0
    STEP_M = 35000.0
    rng = np.random.default_rng(42)

    candidates = []
    for path in MARS_DEM_FILES:
        if not path.exists():
            log_step(f"  -> Mars DEM file not found: {path}")
            continue
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            is_projected = (src.crs is not None and not src.crs.is_geographic) or (
                abs(right - left) > 400.0
            )
            log_step(
                f"  -> Analyzing grid {path.name}: bounds=({left:.1f}, {bottom:.1f}, {right:.1f}, {top:.1f}), projected={is_projected}"
            )

            if is_projected:
                half_m = TILE_SIZE_M / 2.0
                xs = np.arange(left + half_m, right - half_m, STEP_M)
                ys = np.arange(bottom + half_m, top - half_m, STEP_M)
                for y in ys:
                    lat_approx = math.degrees(y / MARS_RADIUS_M)
                    if abs(lat_approx) > 82.0:
                        continue
                    for x in xs:
                        candidates.append(
                            (path, True, float(x), float(y), half_m, half_m)
                        )
            else:
                lat_min = max(bottom + 0.5, -82.0)
                lat_max = min(top - 0.5, 82.0)
                dlat_step = STEP_M / MARS_M_PER_DEG
                for lat in np.arange(lat_min, lat_max, dlat_step):
                    cos_lat = max(math.cos(math.radians(lat)), 0.15)
                    dlon_step = dlat_step / cos_lat
                    half_dlat = (TILE_SIZE_M / 2.0) / MARS_M_PER_DEG
                    half_dlon = half_dlat / cos_lat
                    for lon in np.arange(
                        left + half_dlon, right - half_dlon, dlon_step
                    ):
                        candidates.append(
                            (path, False, float(lon), float(lat), half_dlon, half_dlat)
                        )

    log_step(f"  -> Total Martian candidate patches: {len(candidates)}")
    target_pool = int(max_tiles * 1.15)
    if len(candidates) > target_pool:
        chosen = rng.choice(len(candidates), size=target_pool, replace=False)
        candidates = [candidates[i] for i in chosen]
    log_step(f"  -> Selected for slicing: {len(candidates)} points")

    by_file: dict[Path, list] = {}
    for item in candidates:
        by_file.setdefault(item[0], []).append(item[1:])
    del candidates
    gc.collect()

    tensors, meta_rows = [], []
    for path, pts in by_file.items():
        log_step(f"  -> Reading {path.name} into RAM ({len(pts)} patches)...")
        with rasterio.open(path) as src:
            full_dem = src.read(1)
            nodata = src.nodata
            inv_tf = ~src.transform
            px_per_unit_x = 1.0 / abs(src.transform.a)
            px_per_unit_y = 1.0 / abs(src.transform.e)
            h, w = full_dem.shape

            to_geodetic = None
            if pts[0][0] and src.crs is not None:
                to_geodetic = pyproj.Transformer.from_crs(
                    src.crs, src.crs.geodetic_crs, always_xy=True
                )

            for is_proj, cx, cy, hx, hy in tqdm(
                pts, desc=f"Slice {path.stem}", leave=False
            ):
                col_c, row_c = inv_tf * (cx, cy)
                rx = max(2, int(round(hx * px_per_unit_x)))
                ry = max(2, int(round(hy * px_per_unit_y)))

                r0, r1 = int(round(row_c)) - ry, int(round(row_c)) + ry
                c0, c1 = int(round(col_c)) - rx, int(round(col_c)) + rx
                if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
                    continue

                sub = full_dem[r0:r1, c0:c1]
                if nodata is not None and np.any(sub == nodata):
                    continue

                patch = zoom(
                    sub.astype(np.float32),
                    (PATCH_PX / sub.shape[0], PATCH_PX / sub.shape[1]),
                    order=1,
                )

                if is_proj:
                    if to_geodetic is not None:
                        lon, lat = to_geodetic.transform(cx, cy)
                    else:
                        lon = math.degrees(cx / MARS_RADIUS_M)
                        lat = math.degrees(cy / MARS_RADIUS_M)
                else:
                    lon, lat = cx, cy

                norm_lon = float(((lon + 180.0) % 360.0) - 180.0)
                lat = float(lat)
                if abs(lat) > 82.0:
                    continue

                th_val, rock_val = mars_lookup.get(norm_lon, lat)
                res = dem_to_feature_tensor(patch, th_val, rock_val)
                if res is not None:
                    tensor, stats = res
                    tensors.append(tensor)
                    meta_rows.append(
                        {
                            "planet": "mars",
                            "lat": lat,
                            "lon": norm_lon,
                            "x_m": cx if is_proj else 0.0,
                            "y_m": cy if is_proj else 0.0,
                            **stats,
                        }
                    )

        del full_dem
        gc.collect()
        log_step(
            f"  -> Finished quadrant {path.name}, accumulated Martian patches: {len(tensors)}"
        )

    if len(tensors) > max_tiles:
        idx = rng.choice(len(tensors), size=max_tiles, replace=False)
        tensors = [tensors[i] for i in idx]
        meta_rows = [meta_rows[i] for i in idx]

    out_arr = np.array(tensors, dtype=np.float32)
    del tensors
    gc.collect()
    log_step(f"  -> Total Mars patches: {len(out_arr)} ({out_arr.nbytes / 1e6:.1f} MB)")
    return out_arr, pd.DataFrame(meta_rows)
