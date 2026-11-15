# 🚀 Astra Frontier

**Astra Frontier** is an interactive geospatial platform for discovering, 3D-visualizing, and physically comparing terrestrial locations that serve as natural analogs for candidate permanent lunar base sites and selected regions of Mars. Developed for the **NASA Space Apps Challenge**.

It helps mission engineers and planetary scientists find Earth field-testing sites for rover mobility, instrument calibration, and astronaut training, using real orbital datasets.

## Table of Contents

- [Key Results](#-key-results)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Downloading Assets](#-downloading-assets)
- [Running the Web App](#%EF%B8%8F-running-the-web-app-backend)
- [Rebuilding Embeddings & Index](#-rebuilding-embeddings--faiss-index-pipeline)
- [Limitations](#%EF%B8%8F-limitations)
- [Data Sources & Tech Stack](#%EF%B8%8F-data-sources--tech-stack)
- [Acknowledgements & License](#-acknowledgements--license)

---

## 📊 Key Results

- **575,483 Earth sites** indexed: the whole land surface on a 15 km grid, each a $5 \times 5\text{ km}$ patch.
- **45,000 planetary target points**: 20,000 around the Lunar South Pole and 25,000 across Mars.
- **Search in under 5 ms** over the whole Earth database (FAISS).
- **Unsupervised validation:** the model never sees coordinates or place names, yet it independently links:
  - Shackleton crater (Moon) → Atacama Desert and Antarctica;
  - Jezero crater (Mars) → rocky deserts of Antarctica and northern Greenland, which are real NASA test sites.

---

## 🧠 How It Works

1. **Orbital Data Harmonization ($5 \times 5\text{ km}$ patches):**
   Terrain across Earth, the Lunar South Pole, and Mars is sliced into $5 \times 5\text{ km}$ sections ($64 \times 64$ pixels) and converted into **5 scale-invariant physical channels**:
   * **Slope Angle**: terrain steepness in degrees;
   * **Micro-Roughness**: local elevation variance within a $3 \times 3$ neighborhood (a proxy for rockiness);
   * **Relative Relief**: normalized elevation geometry, independent of planetary sea level;
   * **Diurnal Thermal Swing**: thermophysical inertia of the surface (regolith looseness). Earth uses WorldClim mean diurnal temperature range (`bio_2`); Mars uses MGS TES thermal inertia (inverted: low inertia = large swing); the Moon uses Diviner temperature tables. Every layer is normalized to [0, 1] by its 5th–95th percentile. Where a layer has no data at a point (NaN), a neutral baseline of `0.5` is used.
   * **Surface Aridity / Albedo**: regolith dryness and reflectance. Earth uses WorldClim annual precipitation (`bio_12`, inverted so that drier = higher); Mars uses TES thermal inertia as a soil/rock proxy; the Moon uses LOLA albedo. All are scaled to [0, 1] (5th–95th percentile clipping), but they are different physical quantities, so this channel is a proxy rather than an exact match.
2. **Neural Feature Encoder (`PyTorch`):**
   A convolutional autoencoder (`GeoAutoencoder`) compresses each $5 \times 64 \times 64$ tensor into an L2-normalized **128-dimensional latent embedding** (`geo_encoder.pth`). It is trained on a mix of Earth, Moon and Mars patches with random 90° rotations and flips (rotation robustness comes from augmentation, not from the architecture), using SmoothL1 reconstruction loss, AdamW, 2,500 steps, batch size 128. It never sees geographic coordinates or region names.
3. **Vector Similarity Search (`FAISS`):**
   A `faiss.IndexFlatIP` over the L2-normalized embeddings (inner product = cosine similarity) covers all Earth sites (`earth_analogs.index`) and gives instant analog retrieval.
4. **Interactive 3D & 2D Inspector:**
   * World map with filtering by continent, planetary target (Moon / Mars), and physical priority;
   * Top-5 Earth analogs for any selected Moon/Mars point, with a similarity heatmap;
   * Dual synchronized 3D terrain viewports (`Three.js`) with real satellite imagery and physical overlays;
   * Dual $64 \times 64$ 2D projections with pixel-level crosshair inspection, live elevation cross-sections, and latent vector alignment (`Chart.js`).

---

## 📂 Project Structure

```text
Astra-Frontier/
├── assets/                 # Not in git, see "Downloading Assets"
│   ├── data/               # FAISS index, PyTorch weights, Parquet tables
│   ├── earth_static/       # WorldClim bioclimatic rasters (bio_2, bio_12)
│   ├── mars/               # MGS MOLA (megt*.img/.lbl) and TES thermal inertia (mgst_9001/)
│   └── moon/               # LOLA DEM/albedo (ldem*/ldam*) and Diviner temperature tables (.tab)
├── backend/
│   ├── src/                # FastAPI modules: config.py, engine.py, model.py, terrain.py
│   ├── main.py             # FastAPI application entry point
│   ├── pyproject.toml      # Web backend dependencies
│   └── uv.lock
├── pipeline/
│   ├── src/                # config.py, lookups.py, features.py, planetary_slicer.py,
│   │                       # earth_stream.py, model.py, utils.py
│   ├── pipeline.py         # End-to-end training and FAISS indexing entry point
│   ├── pyproject.toml      # ML pipeline dependencies
│   └── uv.lock
├── frontend/
│   ├── index.html          # Main UI (Overview, Earth Map, 3D Inspector, About)
│   ├── style.css           # Styling and layout animations
│   ├── app.js              # Leaflet, Three.js, and Chart.js logic
│   └── earth.svg           # Interactive hero globe graphic
└── README.md
```

Files in `assets/data/`:

| File | Description |
|---|---|
| `earth_analogs.index` | FAISS index of 575,483 Earth patches |
| `earth_tiles.parquet` | Physical characteristics of every Earth patch |
| `planetary_targets.parquet` | 45,000 Moon/Mars points with vectors and metrics |
| `geo_encoder.pth` | Trained encoder weights; encodes any new coordinates |

---

## 📦 Downloading Assets

Precomputed embeddings, FAISS index files, and raw planetary rasters are distributed separately due to file size limits.

1. Download `assets.zip` from Google Drive (SHA-256: `c4dbd9f39c7be23f7c2befd5a7e6d56cc3352e22ce49a85fa78f255adcf422d4`):
   👉 **[Download assets.zip](https://drive.google.com/file/d/1KTSRo91Z0aWQLf_hnkL6FnzklbAn8Iug/view?usp=sharing)**
2. Extract the archive into the repository root so that `assets/` contains `data/`, `earth_static/`, `mars/`, and `moon/`.

---

## ⚙️ Running the Web App (`backend/`)

**Requirements:** Python 3.12 (tested), and an internet connection: the frontend loads Tailwind CSS, Leaflet, Three.js, Chart.js, AOS and Google Fonts from CDNs. The FastAPI server also serves `frontend/`.

All web server dependencies are defined in `backend/pyproject.toml`. CPU **PyTorch** is installed separately from the PyTorch wheel index.

### Option 1: `uv` (recommended)

1. Go to the backend directory:
   ```bash
   cd backend
   ```
2. Sync dependencies and install CPU PyTorch separately:
   ```bash
   uv sync
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
3. Start the server (`--no-sync` stops `uv` from overwriting the CPU PyTorch wheel):
   ```bash
   uv run --no-sync uvicorn main:app --reload --port 8000
   ```

### Option 2: `venv` + `pip`

1. Create a virtual environment:
   ```bash
   cd backend
   python -m venv .venv
   ```
2. Activate it:
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
3. Install CPU PyTorch and the dependencies:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install numpy pandas pyarrow faiss-cpu rasterio pyproj scipy fastapi uvicorn
   ```
4. Start the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

Open **http://127.0.0.1:8000** in your browser.

---

## 🔬 Rebuilding Embeddings & FAISS Index (`pipeline/`)

Use this to retrain the autoencoder from scratch or regenerate `assets/data/` from raw orbital rasters.

> The pipeline streams Copernicus DEM tiles from Microsoft Planetary Computer, so it needs internet access. An optional `PC_SUBSCRIPTION_KEY` environment variable can be set for higher Planetary Computer rate limits. A CUDA/ROCm GPU is used automatically if available (the `torch` CPU wheel installed above is CPU-only; install a GPU build of PyTorch for your hardware to use it), otherwise the CPU.
>
> Reference run: about **3 hours** on an AMD Radeon RX 6750 XT, with roughly **9 GB of RAM** in use. Tested with Python 3.12.

### With `uv`

```bash
cd pipeline
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync python pipeline.py
```

### With `venv` + `pip`

```bash
cd pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas pyarrow faiss-cpu rasterio pyproj scipy planetary-computer pystac-client global-land-mask tqdm
python pipeline.py
```

The pipeline reads raw planetary maps from `../assets/` and writes `geo_encoder.pth`, `planetary_targets.parquet`, `earth_tiles.parquet`, and `earth_analogs.index` to `../assets/data/`.

**Extending without retraining:** new bodies (Mercury, Venus, large asteroids) or a denser Earth grid (15 km → 5 km) only need to be passed through the saved `geo_encoder.pth`.

---

## ⚠️ Limitations

- **Resolution:** a 5 km patch at 64 px is about 78 m/pixel. Copernicus GLO-90 is 90 m, but Mars MOLA is far coarser (hundreds of meters), so Mars patches are heavily interpolated.
- **Thermal and aridity channels:** Earth values come from coarse (2.5 arc-minute) WorldClim climate rasters, while Moon/Mars values come from orbital instruments. They are normalized to the same range but are not the same physical quantity. Points with missing data fall back to `0.5`.
- **Analogy ≠ equivalence:** similarity covers terrain and surface properties only. Gravity, vacuum, radiation, dust electrostatics and illumination are not modeled.
- **Earth grid:** sites are sampled every 15 km, so smaller local analogs can be missed.

---

## 🛰️ Data Sources & Tech Stack

* **Datasets:** NASA LOLA, Lunar Diviner, Mars Global Surveyor (MOLA & TES), Copernicus DEM (`cop-dem-glo-90` via Microsoft Planetary Computer), WorldClim v2.1.
* **Backend & ML:** Python, FastAPI, Uvicorn, PyTorch, FAISS, NumPy, Pandas, PyArrow, Rasterio, PyProj, SciPy.
* **Frontend:** HTML5, CSS3, Tailwind CSS, JavaScript (ES6+), Three.js (`OrbitControls`), Leaflet (`Canvas` renderer), Chart.js, AOS.

---

## 🙏 Acknowledgements & License

Built for the [NASA Space Apps Challenge 2026](https://www.spaceappschallenge.org/2026/find-a-team/astra-frontier/) by team **Astra Frontier**.

[MIT Licence](/LICENSE)

Planetary data courtesy of NASA PDS (LOLA, Diviner, MGS MOLA/TES). Elevation data: Copernicus DEM (ESA / European Commission). Climate data: WorldClim v2.1 (Fick & Hijmans, 2017).
