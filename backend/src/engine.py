import math

import faiss
import numpy as np
import pandas as pd

from src.config import DATA_DIR, LATENT_DIM
from src.model import decode_latent_to_3d, load_autoencoder
from src.terrain import (
    dem_to_3d_payload,
    fetch_earth_dem,
    fetch_earth_satellite_jpeg,
    slice_local_dem,
)


def classify_continents(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    cont = np.full(len(lats), "Asia", dtype=object)
    cont[lats < -60.0] = "Antarctica"
    cont[(lats >= -50.0) & (lats <= 0.0) & (lons >= 110.0) & (lons <= 180.0)] = (
        "Oceania"
    )
    cont[(lats >= -60.0) & (lats < 12.0) & (lons >= -90.0) & (lons <= -30.0)] = (
        "South America"
    )
    na_mask = ((lats >= 12.0) & (lons >= -170.0) & (lons <= -50.0)) | (
        (lats >= 58.0) & (lons >= -75.0) & (lons <= -10.0)
    )
    cont[na_mask] = "North America"
    eu_mask = (lats >= 35.0) & (lats <= 72.0) & (lons >= -25.0) & (lons < 60.0)
    cont[eu_mask] = "Europe"
    af_mask = (
        (lats >= -36.0)
        & (lats < 35.0)
        & (lons >= -20.0)
        & (lons <= 52.0)
        & ~((lats > 12.0) & (lons > 34.0))
    )
    cont[af_mask] = "Africa"
    return cont


class EarthToSpaceEngine:
    def __init__(self):
        print("[1/4] Loading the GeoAutoencoder neural network...")
        self.model = load_autoencoder(DATA_DIR / "geo_encoder.pth")

        print("[2/4] Loading Earth base and Moon/Mars targets...")
        self.earth_df = pd.read_parquet(DATA_DIR / "earth_tiles.parquet").reset_index(
            drop=True
        )
        self.earth_df["earth_id"] = self.earth_df.index
        self.earth_df["continent"] = classify_continents(
            self.earth_df["lat"].to_numpy(), self.earth_df["lon"].to_numpy()
        )

        earth_index = faiss.read_index(str(DATA_DIR / "earth_analogs.index"))
        try:
            self.earth_vecs = (
                faiss.vector_to_array(earth_index.codes)
                .view(np.float32)
                .reshape(earth_index.ntotal, LATENT_DIM)
            )
        except Exception:
            self.earth_vecs = earth_index.reconstruct_n(0, earth_index.ntotal)
        self.earth_vecs = np.ascontiguousarray(self.earth_vecs, dtype=np.float32)

        self.targets_df = pd.read_parquet(
            DATA_DIR / "planetary_targets.parquet"
        ).reset_index(drop=True)
        self.targets_df["target_id"] = self.targets_df.index
        emb_cols = [f"emb_{i}" for i in range(LATENT_DIM)]
        self.target_vecs = np.ascontiguousarray(
            self.targets_df[emb_cols].to_numpy(dtype=np.float32)
        )

        self.moon_ids = self.targets_df.index[
            self.targets_df["planet"] == "moon"
        ].to_numpy()
        self.mars_ids = self.targets_df.index[
            self.targets_df["planet"] == "mars"
        ].to_numpy()

        moon_index = faiss.IndexFlatIP(LATENT_DIM)
        moon_index.add(self.target_vecs[self.moon_ids])

        mars_index = faiss.IndexFlatIP(LATENT_DIM)
        mars_index.add(self.target_vecs[self.mars_ids])

        print(
            "[3/4] Calculating the best space-based analogs for all of Earth’s landmasses..."
        )
        m_sim, m_local_idx = moon_index.search(self.earth_vecs, k=1)
        r_sim, r_local_idx = mars_index.search(self.earth_vecs, k=1)

        self.earth_df["sim_moon"] = (m_sim[:, 0] * 100.0).astype(np.float32)
        self.earth_df["best_moon_id"] = self.moon_ids[m_local_idx[:, 0]]
        self.earth_df["sim_mars"] = (r_sim[:, 0] * 100.0).astype(np.float32)
        self.earth_df["best_mars_id"] = self.mars_ids[r_local_idx[:, 0]]

        self.terrain_cache: dict[tuple[int, int], dict] = {}
        self.sat_cache: dict[tuple[float, float], bytes] = {}
        print(
            f"[4/4] Ready! Earth: {len(self.earth_df):,} | "
            f"Moon: {len(self.moon_ids):,} | Mars: {len(self.mars_ids):,}"
        )

    def query_earth_locations(
        self,
        continent: str = "ALL",
        planet: str = "ALL",
        match_type: str = "overall",
        limit: int = 120,
    ) -> dict:
        df = self.earth_df
        mask = np.ones(len(df), dtype=bool)
        if continent != "ALL":
            mask &= df["continent"].to_numpy() == continent

        sub_idx = np.where(mask)[0]
        if len(sub_idx) == 0:
            return {"map_points": [], "list_items": [], "total_matched": 0}

        sim_moon = df["sim_moon"].to_numpy()[sub_idx]
        sim_mars = df["sim_mars"].to_numpy()[sub_idx]

        if planet == "moon":
            base_score = sim_moon
            matched_planet = np.full(len(sub_idx), "moon", dtype=object)
        elif planet == "mars":
            base_score = sim_mars
            matched_planet = np.full(len(sub_idx), "mars", dtype=object)
        else:
            base_score = np.maximum(sim_moon, sim_mars)
            matched_planet = np.where(sim_moon >= sim_mars, "moon", "mars")

        rank_score = base_score.copy()
        if match_type == "slope":
            slope = df["mean_slope"].to_numpy()[sub_idx]
            rank_score = base_score * 0.85 + np.clip(slope / 20.0, 0, 1) * 15.0
        elif match_type == "roughness":
            rough = df["mean_roughness"].to_numpy()[sub_idx]
            rank_score = base_score * 0.85 + np.clip(rough / 20.0, 0, 1) * 15.0
        elif match_type == "thermal":
            soil = df["surface_soil_idx"].to_numpy()[sub_idx]
            non_stub = np.where(np.abs(soil - 0.5) > 0.02, 10.0, -15.0)
            rank_score = base_score * 0.90 + soil * 10.0 + non_stub

        k = min(limit, len(sub_idx))
        top_local = np.argpartition(-rank_score, k - 1)[:k]
        top_local = top_local[np.argsort(-rank_score[top_local])]

        list_items = []
        for li in top_local:
            eid = int(sub_idx[li])
            r = df.iloc[eid]
            pl = str(matched_planet[li])
            tid = int(r["best_moon_id"] if pl == "moon" else r["best_mars_id"])
            t_row = self.targets_df.iloc[tid]
            list_items.append(
                {
                    "earth_id": eid,
                    "target_id": tid,
                    "planet": pl,
                    "score": round(float(base_score[li]), 1),
                    "lat": round(float(r["lat"]), 3),
                    "lon": round(float(r["lon"]), 3),
                    "continent": str(r["continent"]),
                    "mean_slope": round(float(r["mean_slope"]), 1),
                    "target_lat": round(float(t_row["lat"]), 2),
                    "target_lon": round(float(t_row["lon"]), 2),
                }
            )

        high_local = np.where(base_score >= 88.0)[0]
        if len(high_local) > 4500:
            rng = np.random.default_rng(42)
            high_local = rng.choice(high_local, size=4500, replace=False)

        lats_sub = df["lat"].to_numpy()[sub_idx]
        lons_sub = df["lon"].to_numpy()[sub_idx]
        map_points = [
            [
                int(sub_idx[li]),
                round(float(lats_sub[li]), 2),
                round(float(lons_sub[li]), 2),
                round(float(base_score[li]), 1),
                1 if matched_planet[li] == "moon" else 2,
            ]
            for li in high_local
        ]

        return {
            "map_points": map_points,
            "list_items": list_items,
            "total_matched": int(np.sum(base_score >= 90.0)),
        }

    def find_nearest_earth(self, lat: float, lon: float, continent: str = "ALL") -> int:
        sub = (
            self.earth_df
            if continent == "ALL"
            else self.earth_df[self.earth_df["continent"] == continent]
        )
        if sub.empty:
            sub = self.earth_df
        dlat = sub["lat"].to_numpy() - lat
        dlon = ((sub["lon"].to_numpy() - lon + 180.0) % 360.0) - 180.0
        dist2 = (
            dlat * dlat
            + (dlon * math.cos(math.radians(max(-85.0, min(85.0, lat))))) ** 2
        )
        return int(sub.index[int(np.argmin(dist2))])

    def get_pair_comparison(
        self, earth_id: int, force_planet: str | None = None
    ) -> dict:
        earth_id = max(0, min(len(self.earth_df) - 1, earth_id))
        e_row = self.earth_df.iloc[earth_id]

        if force_planet == "moon":
            planet, target_id, score = (
                "moon",
                int(e_row["best_moon_id"]),
                float(e_row["sim_moon"]),
            )
        elif force_planet == "mars":
            planet, target_id, score = (
                "mars",
                int(e_row["best_mars_id"]),
                float(e_row["sim_mars"]),
            )
        elif float(e_row["sim_moon"]) >= float(e_row["sim_mars"]):
            planet, target_id, score = (
                "moon",
                int(e_row["best_moon_id"]),
                float(e_row["sim_moon"]),
            )
        else:
            planet, target_id, score = (
                "mars",
                int(e_row["best_mars_id"]),
                float(e_row["sim_mars"]),
            )

        t_row = self.targets_df.iloc[target_id]
        terrain = self._get_terrain_pair(earth_id, target_id)

        return {
            "score": round(score, 2),
            "sim_moon": round(float(e_row["sim_moon"]), 2),
            "sim_mars": round(float(e_row["sim_mars"]), 2),
            "earth": {
                "earth_id": int(earth_id),
                "lat": round(float(e_row["lat"]), 4),
                "lon": round(float(e_row["lon"]), 4),
                "continent": str(e_row["continent"]),
                "mean_slope": round(float(e_row["mean_slope"]), 2),
                "max_slope": round(float(e_row["max_slope"]), 2),
                "mean_roughness": round(float(e_row["mean_roughness"]), 2),
                "relief_std": round(float(e_row["relief_std"]), 2),
                "thermal_swing_idx": round(float(e_row["thermal_swing_idx"]), 3),
                "surface_soil_idx": round(float(e_row["surface_soil_idx"]), 3),
                "vec_64": np.round(self.earth_vecs[earth_id][:64], 3).tolist(),
                "terrain": terrain["earth_3d"],
            },
            "target": {
                "target_id": int(target_id),
                "planet": planet,
                "lat": round(float(t_row["lat"]), 4),
                "lon": round(float(t_row["lon"]), 4),
                "mean_slope": round(float(t_row["mean_slope"]), 2),
                "max_slope": round(float(t_row["max_slope"]), 2),
                "mean_roughness": round(float(t_row["mean_roughness"]), 2),
                "relief_std": round(float(t_row["relief_std"]), 2),
                "thermal_swing_idx": round(float(t_row["thermal_swing_idx"]), 3),
                "surface_soil_idx": round(float(t_row["surface_soil_idx"]), 3),
                "vec_64": np.round(self.target_vecs[target_id][:64], 3).tolist(),
                "terrain": terrain["target_3d"],
            },
        }

    def _get_terrain_pair(self, earth_id: int, target_id: int) -> dict:
        key = (earth_id, target_id)
        if key in self.terrain_cache:
            return self.terrain_cache[key]

        t_row = self.targets_df.iloc[target_id]
        e_row = self.earth_df.iloc[earth_id]

        real_target_dem = slice_local_dem(t_row)
        if real_target_dem is not None:
            src_name = (
                "LOLA DEM (Moon)" if t_row["planet"] == "moon" else "MOLA DEM (Mars)"
            )
            target_3d = dem_to_3d_payload(
                real_target_dem,
                float(t_row["thermal_swing_idx"]),
                float(t_row["surface_soil_idx"]),
                src_name,
            )
        else:
            target_3d = decode_latent_to_3d(self.model, self.target_vecs[target_id])

        real_earth_dem = fetch_earth_dem(float(e_row["lat"]), float(e_row["lon"]))
        if real_earth_dem is not None and np.std(real_earth_dem) > 0.1:
            earth_3d = dem_to_3d_payload(
                real_earth_dem,
                float(e_row["thermal_swing_idx"]),
                float(e_row["surface_soil_idx"]),
                "AWS Terrarium Global DEM",
            )
        else:
            earth_3d = decode_latent_to_3d(self.model, self.earth_vecs[earth_id])

        result = {"target_3d": target_3d, "earth_3d": earth_3d}
        if len(self.terrain_cache) > 120:
            self.terrain_cache.clear()
        self.terrain_cache[key] = result
        return result

    def get_satellite_tile(self, lat: float, lon: float) -> bytes:
        key = (round(lat, 3), round(lon, 3))
        if key not in self.sat_cache:
            if len(self.sat_cache) > 200:
                self.sat_cache.clear()
            self.sat_cache[key] = fetch_earth_satellite_jpeg(lat, lon)
        return self.sat_cache[key]
