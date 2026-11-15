import gc
import time

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    EARTH_ARIDITY_TIF,
    EARTH_THERMAL_TIF,
    IN_CHANNELS,
    LATENT_DIM,
    LR,
    MARS_TES_DAY_LBL,
    MARS_TES_NIGHT_LBL,
    MOON_ALBEDO_XML,
    MOON_DIVINER_NIGHT_LBL,
    MOON_DIVINER_NOON_LBL,
    OUTPUT_DIR,
    TRAIN_STEPS,
)
from src.earth_stream import (
    EarthCogStreamer,
    MixedPlanetaryStreamDataset,
    build_earth_macro_cells,
)
from src.lookups import GeoTiffLookup, LunarPolarLookup, MarsTesLookup
from src.model import GeoAutoencoder
from src.planetary_slicer import load_lunar_tiles, load_mars_tiles
from src.utils import log_step


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log_step(f"=== Pipeline Started | Device: {device} ({device_name}) ===")

    log_step("[0/5] Loading planetary thermal & surface lookup grids into RAM...")
    lunar_lookup = LunarPolarLookup(
        albedo_xml=MOON_ALBEDO_XML,
        diviner_night_lbl=MOON_DIVINER_NIGHT_LBL,
        diviner_noon_lbl=MOON_DIVINER_NOON_LBL,
    )
    mars_lookup = MarsTesLookup(
        night_lbl=MARS_TES_NIGHT_LBL,
        day_lbl=MARS_TES_DAY_LBL,
    )
    earth_thermal = GeoTiffLookup(EARTH_THERMAL_TIF, invert=False)
    earth_aridity = GeoTiffLookup(EARTH_ARIDITY_TIF, invert=True)

    moon_tensors, moon_df = load_lunar_tiles(lunar_lookup)
    mars_tensors, mars_df = load_mars_tiles(mars_lookup)
    earth_cells = build_earth_macro_cells()

    log_step(
        f"[4/5] Training 5-channel Convolutional Autoencoder ({TRAIN_STEPS} steps, batch {BATCH_SIZE})..."
    )
    model = GeoAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    train_dataset = MixedPlanetaryStreamDataset(
        earth_cells, earth_thermal, earth_aridity, moon_tensors, mars_tensors
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)

    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=TRAIN_STEPS, desc="Training")

    for step, batch in pbar:
        if step >= TRAIN_STEPS:
            break
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()

        running_loss = (
            0.95 * running_loss + 0.05 * loss.item() if step > 0 else loss.item()
        )
        if step % 25 == 0:
            pbar.set_postfix(loss=f"{running_loss:.5f}")
        if step > 0 and step % 500 == 0:
            log_step(
                f"  -> Training step {step}/{TRAIN_STEPS} | loss={running_loss:.5f}"
            )

    model_path = OUTPUT_DIR / "geo_encoder.pth"
    torch.save(model.state_dict(), model_path)
    log_step(f"  -> Saved model weights: {model_path}")

    log_step("[5/5] Indexing Earth, Moon, and Mars embeddings into FAISS...")
    model.eval()

    def encode_numpy_array(arr: np.ndarray) -> np.ndarray:
        embs = []
        with torch.no_grad():
            for i in range(0, len(arr), 256):
                chunk = torch.from_numpy(arr[i : i + 256]).to(device)
                embs.append(model.encode(chunk).cpu().numpy())
        return np.vstack(embs).astype(np.float32)

    planetary_dfs = []
    if len(moon_tensors) > 0:
        moon_embs = encode_numpy_array(moon_tensors)
        emb_cols = pd.DataFrame(
            moon_embs, columns=[f"emb_{i}" for i in range(LATENT_DIM)]
        )
        planetary_dfs.append(
            pd.concat([moon_df.reset_index(drop=True), emb_cols], axis=1)
        )
        del moon_tensors
        gc.collect()

    if len(mars_tensors) > 0:
        mars_embs = encode_numpy_array(mars_tensors)
        emb_cols = pd.DataFrame(
            mars_embs, columns=[f"emb_{i}" for i in range(LATENT_DIM)]
        )
        planetary_dfs.append(
            pd.concat([mars_df.reset_index(drop=True), emb_cols], axis=1)
        )
        del mars_tensors
        gc.collect()

    planetary_all_df = pd.concat(planetary_dfs, ignore_index=True)
    planetary_path = OUTPUT_DIR / "planetary_targets.parquet"
    planetary_all_df.to_parquet(planetary_path, index=False)
    log_step(
        f"  -> Saved planetary targets: {planetary_path} ({len(planetary_all_df)} sites)"
    )

    faiss_index = faiss.IndexFlatIP(LATENT_DIM)
    earth_meta_records = []

    streamer = EarthCogStreamer(earth_cells, earth_thermal, earth_aridity)
    batch_tensors, batch_metas = [], []

    with torch.no_grad():
        for tensor, meta in tqdm(streamer, desc="Indexing Earth"):
            batch_tensors.append(tensor)
            batch_metas.append(meta)
            if len(batch_tensors) >= 256:
                x = torch.from_numpy(np.array(batch_tensors, dtype=np.float32)).to(
                    device
                )
                faiss_index.add(model.encode(x).cpu().numpy().astype(np.float32))
                earth_meta_records.extend(batch_metas)
                batch_tensors.clear()
                batch_metas.clear()
                if faiss_index.ntotal % 50000 == 0:
                    log_step(f"  -> Indexed Earth patches: {faiss_index.ntotal}")

        if batch_tensors:
            x = torch.from_numpy(np.array(batch_tensors, dtype=np.float32)).to(device)
            faiss_index.add(model.encode(x).cpu().numpy().astype(np.float32))
            earth_meta_records.extend(batch_metas)

    index_path = OUTPUT_DIR / "earth_analogs.index"
    earth_meta_path = OUTPUT_DIR / "earth_tiles.parquet"
    faiss.write_index(faiss_index, str(index_path))
    earth_df = pd.DataFrame(earth_meta_records)
    earth_df.to_parquet(earth_meta_path, index=False)

    log_step(
        f"  -> Saved Earth FAISS index: {index_path} ({faiss_index.ntotal} vectors)"
    )
    log_step(f"  -> Saved Earth metadata: {earth_meta_path}")
    log_step(f"=== PIPELINE COMPLETED in {(time.time() - t0) / 60:.1f} min ===")

    print("\n=== SANITY CHECK: TERRESTRIAL ANALOG SEARCH ===")
    emb_columns = [f"emb_{i}" for i in range(LATENT_DIM)]
    test_targets = [
        ("Moon: Shackleton Crater (South Pole, Artemis Zone)", "moon", -89.5, 0.0),
        ("Mars: Jezero Crater (Perseverance Landing Site)", "mars", 18.38, 77.58),
    ]

    for title, planet_name, target_lat, target_lon in test_targets:
        sub = planetary_all_df[planetary_all_df["planet"] == planet_name]
        if sub.empty:
            continue
        dist = (sub["lat"] - target_lat) ** 2 + (sub["lon"] - target_lon) ** 2
        best_row = sub.loc[dist.idxmin()]
        query_vec = best_row[emb_columns].to_numpy(dtype=np.float32).reshape(1, -1)

        scores, indices = faiss_index.search(query_vec, k=5)
        print(
            f"\nTarget: {title}\n"
            f"  Target Metrics -> Slope: {best_row['mean_slope']:.1f}°, "
            f"Thermal Swing: {best_row['thermal_swing_idx']:.2f}, "
            f"Soil/Aridity: {best_row['surface_soil_idx']:.2f}"
        )
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            m = earth_df.iloc[idx]
            print(
                f"  #{rank} | Similarity: {score * 100:.1f}% | "
                f"Earth ({m['lat']:.3f}, {m['lon']:.3f}) | "
                f"Slope: {m['mean_slope']:.1f}°, Thermal: {m['thermal_swing_idx']:.2f}, Aridity: {m['surface_soil_idx']:.2f}"
            )


if __name__ == "__main__":
    main()
