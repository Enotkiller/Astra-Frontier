from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles

from src.config import FRONTEND_DIR
from src.engine import EarthToSpaceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = EarthToSpaceEngine()
    yield


app = FastAPI(title="Astra Frontier 2026 API", lifespan=lifespan)


@app.get("/api/earth_locations")
def get_earth_locations(
    request: Request,
    continent: str = "ALL",
    planet: str = "ALL",
    match_type: str = "overall",
):
    engine: EarthToSpaceEngine = request.app.state.engine
    return engine.query_earth_locations(continent, planet, match_type)


@app.get("/api/compare")
def compare_locations(
    request: Request,
    earth_id: int | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    continent: str = "ALL",
    planet: str | None = None,
):
    engine: EarthToSpaceEngine = request.app.state.engine
    eid = (
        earth_id
        if earth_id is not None
        else engine.find_nearest_earth(lat, lon, continent)
    )
    force_planet = None if planet in ("ALL", "", None) else planet
    return engine.get_pair_comparison(eid, force_planet=force_planet)


@app.get("/api/earth_sat")
def get_earth_satellite(request: Request, lat: float, lon: float):
    engine: EarthToSpaceEngine = request.app.state.engine
    try:
        jpg = engine.get_satellite_tile(lat, lon)
        return Response(content=jpg, media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
