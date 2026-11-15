import gc
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio

from src.config import MARS_RADIUS_M
from src.utils import log_step


class GeoTiffLookup:
    def __init__(self, tif_path: Path, invert: bool = False):
        log_step(f"Loading Earth raster layer: {tif_path}...")
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            self.transform = src.transform
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[~np.isfinite(arr)] = np.nan
            p05, p95 = np.nanpercentile(arr, [5, 95])
            arr = np.clip((arr - p05) / (p95 - p05 + 1e-6), 0.0, 1.0)
            if invert:
                arr = 1.0 - arr
            self.data = arr
        log_step(f"  -> Layer {tif_path.name} ready, shape: {self.data.shape}")

    def get(self, lon: float, lat: float) -> float:
        row, col = rasterio.transform.rowcol(self.transform, lon, lat)
        r = int(np.clip(row, 0, self.data.shape[0] - 1))
        c = int(np.clip(col, 0, self.data.shape[1] - 1))
        val = float(self.data[r, c])
        return 0.5 if np.isnan(val) else val


class MarsTesLookup:
    def __init__(self, night_lbl: Path, day_lbl: Path):
        log_step("Loading Mars thermal inertia (MGS TES)...")
        grids = []
        self.transform = None
        self.is_projected = False
        self.lon_360 = True
        self.to_map = None

        for path in [night_lbl, day_lbl]:
            if not path.exists():
                log_step(f"  -> Skipping missing file: {path}")
                continue
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                if self.transform is None:
                    self.transform = src.transform
                    left, bottom, right, top = src.bounds
                    self.is_projected = (
                        src.crs is not None and not src.crs.is_geographic
                    ) or (abs(right - left) > 400.0)
                    self.lon_360 = right > 180.5
                    if self.is_projected and src.crs is not None:
                        self.to_map = pyproj.Transformer.from_crs(
                            src.crs.geodetic_crs, src.crs, always_xy=True
                        )
                    log_step(
                        f"  -> TES bounds={src.bounds}, projected={self.is_projected}, shape={arr.shape}"
                    )

                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                arr[(arr <= 0.0) | (arr > 3000.0) | (~np.isfinite(arr))] = np.nan
                grids.append(arr)

        if not grids:
            raise FileNotFoundError(
                f"MGS TES Thermal Inertia files not found in {night_lbl.parent}"
            )

        merged = np.nanmean(np.stack(grids, axis=0), axis=0)
        del grids
        gc.collect()

        p05, p95 = np.nanpercentile(merged, [5, 95])
        self.ti_norm = np.clip((merged - p05) / (p95 - p05 + 1e-6), 0.0, 1.0)
        log_step(f"  -> Mars thermal inertia grid ready (p05={p05:.1f}, p95={p95:.1f})")

    def get(self, lon_180: float, lat: float) -> tuple[float, float]:
        if self.is_projected:
            if self.to_map is not None:
                x_q, y_q = self.to_map.transform(lon_180, lat)
            else:
                x_q = math.radians(lon_180) * MARS_RADIUS_M
                y_q = math.radians(lat) * MARS_RADIUS_M
        else:
            x_q = (lon_180 % 360.0) if self.lon_360 else lon_180
            y_q = lat

        row, col = rasterio.transform.rowcol(self.transform, x_q, y_q)
        r = int(np.clip(row, 0, self.ti_norm.shape[0] - 1))
        c = int(np.clip(col, 0, self.ti_norm.shape[1] - 1))
        ti = float(self.ti_norm[r, c])
        if np.isnan(ti):
            return 0.5, 0.5
        return 1.0 - ti, ti


class LunarPolarLookup:
    def __init__(
        self, albedo_xml: Path, diviner_night_lbl: Path, diviner_noon_lbl: Path
    ):
        log_step("Loading Lunar polar albedo (LOLA LDAM)...")
        with rasterio.open(albedo_xml) as src:
            alb = src.read(1).astype(np.float32)
            self.alb_transform = src.transform
            self.lunar_crs = src.crs
            if src.nodata is not None:
                alb[alb == src.nodata] = np.nan
            alb[~np.isfinite(alb)] = np.nan
            p05, p95 = np.nanpercentile(alb, [5, 95])
            self.albedo_grid = np.clip((alb - p05) / (p95 - p05 + 1e-6), 0.0, 1.0)
        log_step(f"  -> Lunar albedo loaded, shape: {self.albedo_grid.shape}")

        self.grid_step = 1000.0
        self.grid_extent = 900_000.0
        self.grid_size = int((2 * self.grid_extent) / self.grid_step)

        log_step("Parsing Lunar nighttime temperatures (Diviner ltim01)...")
        t_night = self._load_diviner_grid(diviner_night_lbl)
        gc.collect()

        log_step("Parsing Lunar daytime temperatures (Diviner ltim49)...")
        t_noon = self._load_diviner_grid(diviner_noon_lbl)
        gc.collect()

        delta_t = np.maximum(0.0, t_noon - t_night)
        del t_night, t_noon
        gc.collect()

        p05, p95 = np.nanpercentile(delta_t, [5, 95])
        self.thermal_swing_grid = np.clip(
            (delta_t - p05) / (p95 - p05 + 1e-6), 0.0, 1.0
        )
        log_step(
            f"  -> Lunar diurnal thermal swing grid ready (p05={p05:.1f}K, p95={p95:.1f}K)"
        )

    def _load_diviner_grid(self, lbl_path: Path) -> np.ndarray:
        tab_path = lbl_path.with_suffix(".tab")
        with open(tab_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()

        has_comma = "," in first_line
        raw_cols = [
            c.strip().lower()
            for c in (first_line.split(",") if has_comma else first_line.split())
        ]
        has_header = any(re.search(r"[a-z]", c) for c in raw_cols)

        if has_header:
            x_idx = next(
                (i for i, c in enumerate(raw_cols) if c in ("x", "x_km", "x_m")), 0
            )
            y_idx = next(
                (i for i, c in enumerate(raw_cols) if c in ("y", "y_km", "y_m")), 1
            )
            t_idx = next(
                (
                    i
                    for i, c in enumerate(raw_cols)
                    if ("tbol" in c or "temp" in c)
                    and "std" not in c
                    and "err" not in c
                ),
                None,
            )
            if t_idx is None:
                lat_idx = next((i for i, c in enumerate(raw_cols) if "lat" in c), 1)
                t_idx = min(lat_idx + 1, len(raw_cols) - 1)
        else:
            x_idx, y_idx = 0, 1
            t_idx = 4 if len(raw_cols) >= 5 else 2

        read_kwargs = {
            "usecols": [x_idx, y_idx, t_idx],
            "header": 0 if has_header else None,
            "dtype": np.float32,
            "on_bad_lines": "skip",
        }
        if has_comma:
            df = pd.read_csv(tab_path, sep=",", skipinitialspace=True, **read_kwargs)
        else:
            df = pd.read_csv(tab_path, sep=r"\s+", **read_kwargs)

        x = df.iloc[:, 0].to_numpy(dtype=np.float32)
        y = df.iloc[:, 1].to_numpy(dtype=np.float32)
        t = df.iloc[:, 2].to_numpy(dtype=np.float32)
        del df

        if np.nanmax(np.abs(x)) < 3000.0:
            x = x * 1000.0
            y = y * 1000.0

        valid = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & (t > 10.0) & (t < 450.0)
        )
        x, y, t = x[valid], y[valid], t[valid]

        col_idx = np.floor((x + self.grid_extent) / self.grid_step).astype(np.int32)
        row_idx = np.floor((self.grid_extent - y) / self.grid_step).astype(np.int32)
        in_bounds = (
            (col_idx >= 0)
            & (col_idx < self.grid_size)
            & (row_idx >= 0)
            & (row_idx < self.grid_size)
        )

        col_idx, row_idx, t = col_idx[in_bounds], row_idx[in_bounds], t[in_bounds]
        sum_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        cnt_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        np.add.at(sum_grid, (row_idx, col_idx), t)
        np.add.at(cnt_grid, (row_idx, col_idx), 1)

        with np.errstate(invalid="ignore"):
            grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan).astype(
                np.float32
            )
        return grid

    def get_xy(self, x_m: float, y_m: float) -> tuple[float, float]:
        r_a, c_a = rasterio.transform.rowcol(self.alb_transform, x_m, y_m)
        r_a = int(np.clip(r_a, 0, self.albedo_grid.shape[0] - 1))
        c_a = int(np.clip(c_a, 0, self.albedo_grid.shape[1] - 1))
        alb_val = float(self.albedo_grid[r_a, c_a])
        if np.isnan(alb_val):
            alb_val = 0.5

        c_t = int(
            np.clip(
                math.floor((x_m + self.grid_extent) / self.grid_step),
                0,
                self.grid_size - 1,
            )
        )
        r_t = int(
            np.clip(
                math.floor((self.grid_extent - y_m) / self.grid_step),
                0,
                self.grid_size - 1,
            )
        )
        th_val = float(self.thermal_swing_grid[r_t, c_t])
        if np.isnan(th_val):
            th_val = 0.5

        return th_val, alb_val
