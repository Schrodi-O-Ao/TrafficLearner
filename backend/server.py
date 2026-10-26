"""Serve preset and uploaded scenarios over HTTP and stream matched traffic over WebSocket."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_ingest import load_from_csv
from backend.runner import load_policy, run_comparison, stream_comparison
from backend.scenario import Scenario
from scenario_profiles import PRESET_PROFILES


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
PRESET_DESCRIPTIONS = {
    "balanced": "Equal low demand on all four approaches.",
    "morning_peak": "High north-south demand before 1800 seconds, then lower demand.",
    "evening_peak": "High east-west demand before 1800 seconds, then lower demand.",
    "ns_heavy": "Persistent high north-south demand with lighter east-west traffic.",
    "spike": "Balanced traffic with a 300-second west-approach surge.",
    "t_junction": "Three-way T-intersection with North, South, and East approaches.",
}
MAX_SCENARIOS = 50
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=4)
scenarios: dict[str, tuple[Scenario, int]] = {}
preset_ids: dict[str, str] = {}


class PresetRequest(BaseModel):
    """Validate a preset name, repeatable schedule seed, and bounded replay duration."""

    name: str
    seed: Optional[int] = Field(default=42, ge=0, le=4_294_967_295)
    episode_length: int = Field(default=3600, ge=300, le=3600)


async def _preload_presets() -> None:
    """Create bounded preview scenarios once when the API process starts."""
    loop = asyncio.get_running_loop()
    for name in PRESET_DESCRIPTIONS:
        scenario = await loop.run_in_executor(executor, lambda preset=name: Scenario.from_preset(preset))
        scenario_id = str(uuid.uuid4())
        scenarios[scenario_id] = (scenario, scenario.duration)
        preset_ids[name] = scenario_id


@asynccontextmanager
async def lifespan(_app):
    """Load the public policy and preset data before accepting web requests."""
    await loopless_preload()
    load_policy()
    try:
        yield
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def loopless_preload() -> None:
    """Run startup scenario creation on the configured worker pool."""
    await _preload_presets()


app = FastAPI(title="TrafficAI signal comparison", lifespan=lifespan)
allowed_origins = {
    "https://schrodi-o-ao.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
}
render_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
if render_url:
    allowed_origins.add(render_url)
allowed_origins.update(origin.strip().rstrip("/") for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",") if origin.strip())
app.add_middleware(CORSMiddleware, allow_origins=sorted(allowed_origins), allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")


def _store_scenario(scenario: Scenario, episode_length: int) -> str:
    """Store a custom scenario after evicting the oldest non-preset entry at the limit."""
    if len(scenarios) >= MAX_SCENARIOS:
        preset_scenario_ids = set(preset_ids.values())
        oldest_custom_id = next((scenario_id for scenario_id in scenarios if scenario_id not in preset_scenario_ids), None)
        if oldest_custom_id is None:
            raise HTTPException(status_code=503, detail="Scenario storage is full; try again later.")
        scenarios.pop(oldest_custom_id)
    scenario_id = str(uuid.uuid4())
    scenarios[scenario_id] = (scenario, episode_length)
    return scenario_id


@app.get("/")
async def root():
    """Return the single-page comparison interface."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    """Return the small vector icon used by browsers and search previews."""
    return FileResponse(os.path.join(FRONTEND_DIR, "favicon.svg"), media_type="image/svg+xml")


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    """Return crawler rules for the hosted application."""
    return FileResponse(os.path.join(FRONTEND_DIR, "robots.txt"), media_type="text/plain")


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    """Return the public application sitemap."""
    return FileResponse(os.path.join(FRONTEND_DIR, "sitemap.xml"), media_type="application/xml")


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Report that the API process is accepting requests."""
    return {"status": "ok"}


@app.get("/api/scenarios")
async def get_scenarios():
    """List shared preset names and their input descriptions."""
    return [{"id": preset_ids[name], "name": name, "description": description}
            for name, description in PRESET_DESCRIPTIONS.items() if name in preset_ids]


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Validate an uploaded traffic CSV and store its bounded arrival profile."""
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV uploads must be 2 MB or smaller.")
    loop = asyncio.get_running_loop()
    try:
        dataset = await loop.run_in_executor(executor, lambda: load_from_csv(contents))
        scenario = await loop.run_in_executor(executor, lambda: Scenario.from_dataset(dataset, name="custom"))
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"scenario_id": _store_scenario(scenario, scenario.duration)}


@app.post("/api/preset")
async def create_preset(request: PresetRequest):
    """Create a preset with the requested seed and replay duration."""
    if request.name not in PRESET_PROFILES:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {request.name}")
    loop = asyncio.get_running_loop()
    scenario = await loop.run_in_executor(executor, lambda: Scenario.from_preset(request.name, request.seed))
    return {"scenario_id": _store_scenario(scenario, request.episode_length)}


@app.get("/api/scenario/{scenario_id}/info")
async def scenario_info(scenario_id: str):
    """Return the retained scenario duration, demand estimate, and one-lane road drawing."""
    entry = scenarios.get(scenario_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    scenario, episode_length = entry
    return {
        "name": scenario.name,
        "duration": scenario.duration,
        "episode_length": episode_length,
        "total_vehicles_estimate": scenario.dataset_summary.get("estimated_vehicles", 0),
        "dataset_summary": scenario.dataset_summary,
        "road_config": scenario.road_config,
    }


@app.get("/api/presets/{name}/metrics")
async def preset_metrics(name: str, seed: int = Query(default=42, ge=0, le=4_294_967_295),
                         episode_length: int = Query(default=3600, ge=300, le=3600)):
    """Calculate one paired preset result without replay pacing."""
    if name not in PRESET_PROFILES:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {name}")
    loop = asyncio.get_running_loop()
    scenario = await loop.run_in_executor(executor, lambda: Scenario.from_preset(name, seed))
    result = await loop.run_in_executor(executor, lambda: run_comparison(scenario, episode_length, seed))
    summary = result["comparison"]
    return {"baseline": summary["baseline"], "rl": summary["rl"], "rule_layer": summary["rule_layer"]}


@app.websocket("/ws/{scenario_id}")
async def websocket_endpoint(websocket: WebSocket, scenario_id: str,
                             speed: float = Query(default=10.0, ge=1.0, le=50.0),
                             episode_length: Optional[int] = Query(default=None, ge=300, le=3600)):
    """Stream both controller states from the same tick and pace each message for browser playback."""
    await websocket.accept()
    entry = scenarios.get(scenario_id)
    if entry is None:
        await websocket.send_json({"type": "error", "message": "Scenario not found."})
        await websocket.close()
        return
    scenario, stored_duration = entry
    run_duration = stored_duration if episode_length is None else episode_length

    try:
        await websocket.send_json({"type": "road_config", "road_config": scenario.road_config, "scenario_name": scenario.name})
        for message in stream_comparison(scenario, run_duration):
            await websocket.send_json(message)
            if message["type"] == "tick":
                await asyncio.sleep(1.0 / speed)
    except WebSocketDisconnect:
        return
    except (FileNotFoundError, ValueError) as error:
        await websocket.send_json({"type": "error", "message": str(error)})
        await websocket.close()
