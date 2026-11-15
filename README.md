
uv add numpy pandas pyarrow faiss-cpu rasterio pyproj scipy fastapi uvicorn
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

uv run --no-sync uvicorn main:app --reload --port 8000
