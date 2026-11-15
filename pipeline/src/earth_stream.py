import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import torch
from global_land_mask import globe
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from torch.utils.data import IterableDataset

from src.config import (
    EARTH_GRID_STEP_M,
    PATCH_PX,
    PC_SUBSCRIPTION_KEY,
    QUEUE_MAX_SIZE,
    STREAM_WORKERS,
    TILE_SIZE_M,
)
from src.features import dem_to_feature_tensor
from src.lookups import GeoTiffLookup
from src.utils import log_step

if PC_SUBSCRIPTION_KEY:
    planetary_computer.set_subscription_key(PC_SUBSCRIPTION_KEY)


def build_earth_macro_cells(step_m: float = EARTH_GRID_STEP_M) -> list:
    log_step("[3/5] Generating global Earth land grid (filtering oceans)...")
    EARTH_M_PER_DEG = 111320.0
    dlat = step_m / EARTH_M_PER_DEG

    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    total_tiles = 0

    for lat in np.arange(-85.0, 83.0, dlat):
        cos_lat = max(math.cos(math.radians(lat)), 0.12)
        dlon = dlat / cos_lat
        lons = np.arange(-179.8, 179.8, dlon)
        lats_arr = np.full_like(lons, lat)

        valid_lons = lons[globe.is_land(lats_arr, lons)]
        lat_idx = int(math.floor(lat))
        for lon in valid_lons:
            cells.setdefault((lat_idx, int(math.floor(lon))), []).append(
                (float(lat), float(lon))
            )
            total_tiles += 1

    cell_list = [(k[0], k[1], v) for k, v in cells.items()]
    np.random.default_rng(42).shuffle(cell_list)
    log_step(
        f"  -> Generated {total_tiles} terrestrial patches across {len(cell_list)} macro-cells (1°x1°)."
    )
    return cell_list


class EarthCogStreamer:
    def __init__(
        self,
        macro_cells: list,
        thermal_lookup: GeoTiffLookup,
        aridity_lookup: GeoTiffLookup,
        workers: int = STREAM_WORKERS,
        max_queue: int = QUEUE_MAX_SIZE,
    ):
        self.macro_cells = macro_cells
        self.thermal_lookup = thermal_lookup
        self.aridity_lookup = aridity_lookup
        self.workers = workers
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stop_event = threading.Event()
        self._thread = None

    def _fetch_cell(self, cell: tuple[int, int, list[tuple[float, float]]]):
        if self.stop_event.is_set():
            return
        lat_deg, lon_deg, points = cell
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            search = catalog.search(
                collections=["cop-dem-glo-90"],
                bbox=[lon_deg + 0.05, lat_deg + 0.05, lon_deg + 0.95, lat_deg + 0.95],
            )
            items = list(search.items())
            if not items:
                return

            url = items[0].assets["data"].href
            EARTH_M_PER_DEG = 111320.0
            half_dlat = (TILE_SIZE_M / 2.0) / EARTH_M_PER_DEG

            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_HTTP_MAX_RETRY="3",
                GDAL_HTTP_RETRY_DELAY="1",
            ):
                with rasterio.open(url) as src:
                    for lat, lon in points:
                        if self.stop_event.is_set():
                            return
                        cos_lat = max(math.cos(math.radians(lat)), 0.12)
                        half_dlon = half_dlat / cos_lat
                        win = from_bounds(
                            lon - half_dlon,
                            lat - half_dlat,
                            lon + half_dlon,
                            lat + half_dlat,
                            transform=src.transform,
                        )
                        patch = src.read(
                            1,
                            window=win,
                            out_shape=(PATCH_PX, PATCH_PX),
                            resampling=Resampling.bilinear,
                            boundless=True,
                            fill_value=np.nan,
                        ).astype(np.float32)

                        if src.nodata is not None:
                            patch = np.where(patch == src.nodata, np.nan, patch)

                        th_val = self.thermal_lookup.get(lon, lat)
                        arid_val = self.aridity_lookup.get(lon, lat)
                        res = dem_to_feature_tensor(patch, th_val, arid_val)

                        if res is not None:
                            tensor, stats = res
                            meta = {"planet": "earth", "lat": lat, "lon": lon, **stats}
                            while not self.stop_event.is_set():
                                try:
                                    self.q.put((tensor, meta), timeout=0.5)
                                    break
                                except queue.Full:
                                    continue
        except Exception:
            return

    def _producer_loop(self):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._fetch_cell, cell) for cell in self.macro_cells]
            for f in futures:
                if self.stop_event.is_set():
                    break
                try:
                    f.result()
                except Exception:
                    pass
        self.q.put(None)

    def start(self):
        self._thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def __iter__(self):
        self.start()
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                yield item
        finally:
            self.stop()


class MixedPlanetaryStreamDataset(IterableDataset):
    def __init__(
        self,
        earth_cells: list,
        thermal_lookup: GeoTiffLookup,
        aridity_lookup: GeoTiffLookup,
        moon_tensors: np.ndarray,
        mars_tensors: np.ndarray,
    ):
        self.earth_cells = earth_cells
        self.thermal_lookup = thermal_lookup
        self.aridity_lookup = aridity_lookup
        self.moon_tensors = moon_tensors
        self.mars_tensors = mars_tensors

    def _augment(self, x: np.ndarray) -> np.ndarray:
        k = np.random.randint(0, 4)
        if k > 0:
            x = np.rot90(x, k=k, axes=(1, 2))
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2)
        return np.ascontiguousarray(x)

    def __iter__(self):
        streamer = EarthCogStreamer(
            self.earth_cells, self.thermal_lookup, self.aridity_lookup
        )
        rng = np.random.default_rng()
        n_moon = len(self.moon_tensors)
        n_mars = len(self.mars_tensors)
        n_total = n_moon + n_mars

        for earth_tensor, _ in streamer:
            yield torch.from_numpy(self._augment(earth_tensor))
            if n_total > 0 and rng.random() < 0.66:
                idx = int(rng.integers(0, n_total))
                sample = (
                    self.moon_tensors[idx]
                    if idx < n_moon
                    else self.mars_tensors[idx - n_moon]
                )
                yield torch.from_numpy(self._augment(sample))
