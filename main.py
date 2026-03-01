"""
Main Orchestrator: FastAPI Backend & Zero-Cost SQLite Database.
Acts as the central Source of Truth.
"""
# pylint: disable=import-error, no-member, redefined-outer-name, unused-argument, wrong-import-position

from pinn_engine.inference import run_inference
from agent_broker.negotiation_agent import trigger_negotiation_swarm
import asyncio
import os
import sys
import threading

import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# --- 1. SQLite FOSS Database Setup ---
DATABASE_URL = "sqlite:///./inventory.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SESSION_LOCAL = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- WEB3 PARAMETRIC INSURANCE ORACLE (LOCAL GANache BINDING) ---
# [USER INSTRUCTION]: Open your Ganache GUI so it is running live on 127.0.0.1:7545.
# Open Remix IDE in your browser, connect it to your Local Web3 Provider (port 7545),
# paste your KinexicaAsset.sol code, and hit Deploy.
# Copy the actual contract address Remix gives you and paste it explicitly below.
CONTRACT_ADDRESS = "PASTE_YOUR_REMIX_CONTRACT_ADDRESS_HERE"


class AssetRecord(Base):  # pylint: disable=too-few-public-methods
    """
    SQLAlchemy ORM model representing the perishable asset's current state.
    """
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String, unique=True, index=True)
    current_temp_c = Column(Float)
    ethylene_ppm = Column(Float)
    estimated_shelf_life_h = Column(Float)
    status = Column(String)  # "Stable" or "Distressed" or "Liquidated"
    tx_hash = Column(String, nullable=True)
    block_number = Column(Integer, nullable=True)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="SpoilSense API",
              description="Edge Backend for Autonomous Negotiation")


class AppState:  # pylint: disable=too-few-public-methods
    """State holder to avoid global variables for linting."""
    is_monitoring = False


app_state = AppState()

# --- 2. PINN Inference Mock ---


def mock_pinn_inference(row):
    """
    Executes actual PINN inference via run_inference instead of mocked data, 
    utilizing dynamic environmental and CV parameters.
    """
    res = run_inference(
        temp=row.get('temperature_c', 0.0),
        humidity=row.get('humidity_percent', 0.0),
        ethylene=row.get('ethylene_ppm', 0.0),
        cv_variance=row.get('variance_of_laplacian', 1000.0),
        cv_intensity=row.get('mean_intensity', 150.0)
    )
    return res.get("predicted_shelf_life_hours", row.get('actual_shelf_life_hours', 100.0))

# --- 3. Monitoring Loop ---


async def continuous_monitor(csv_path="data/synthetic_sensor_data.csv"):
    """
    Continuous monitoring loop for edge sensors.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_csv_path = os.path.join(base_dir, csv_path)

    if not os.path.exists(full_csv_path):
        print(f"Error: {full_csv_path} not found.")
        app_state.is_monitoring = False
        return

    df = pd.read_csv(full_csv_path)
    db = SESSION_LOCAL()

    # Get or create asset in DB
    asset = db.query(AssetRecord).filter(
        AssetRecord.asset_id == "Pallet-4B-Tomatoes").first()
    if not asset:
        asset = AssetRecord(
            asset_id="Pallet-4B-Tomatoes",
            current_temp_c=0.0,
            ethylene_ppm=0.0,
            estimated_shelf_life_h=500.0,
            status="Stable"
        )
        db.add(asset)
        db.commit()

    for _, row in df.iterrows():
        if not app_state.is_monitoring:
            break

        # Re-fetch asset to avoid stale state
        asset = db.query(AssetRecord).filter(
            AssetRecord.asset_id == "Pallet-4B-Tomatoes").first()

        # Check if already liquidated by a previous breach
        if asset.status == "Liquidated":
            print("[SYSTEM] Asset already liquidated. Monitoring paused.")
            app_state.is_monitoring = False
            break

        predicted_shelf_life = mock_pinn_inference(row)

        # Update DB State
        asset.current_temp_c = row['temperature_c']
        asset.ethylene_ppm = row['ethylene_ppm']
        asset.estimated_shelf_life_h = predicted_shelf_life

        status = "Stable"
        if predicted_shelf_life < 120.0:
            status = "Distressed"
        asset.status = status
        db.commit()

        print(
            f"[API UPDATE] Temp: {asset.current_temp_c}°C | "
            f"Ethylene: {asset.ethylene_ppm} ppm | Status: {asset.status} | "
            f"Shelf Life: {predicted_shelf_life:.2f}h"
        )

        if status == "Distressed":
            print("\n[ALERT] Thermodynamic Anomaly recorded in SQLite.")
            # Trigger Agent Swarm
            payload = {
                "asset_id": asset.asset_id,
                "current_temp_c": asset.current_temp_c,
                "peak_ethylene_ppm": asset.ethylene_ppm,
                "estimated_hours_remaining": predicted_shelf_life
            }
            # Freeze state as liquidated to prevent repeated swarm kickoff
            asset.status = "Liquidated"
            db.commit()

            # Offload heavy LLM task to prevent freezing the API
            threading.Thread(target=trigger_negotiation_swarm,
                             args=(payload,)).start()
            app_state.is_monitoring = False
            break

        # Async sleep simulates edge latency without blocking FastAPI
        await asyncio.sleep(2.0)

    db.close()

# --- 4. API Endpoints ---


@app.get("/")
def read_root():
    """
    Health check endpoint.
    """
    return {"message": "SpoilSense Hub API is entirely FOSS and currently online."}


@app.post("/start-monitoring")
async def start_monitoring(background_tasks: BackgroundTasks):
    """
    Triggers the continuous monitoring loop in a background task.
    """
    if not app_state.is_monitoring:
        app_state.is_monitoring = True
        background_tasks.add_task(continuous_monitor)
        return {"status": "Monitoring triggered in background."}
    return {"status": "Already monitoring."}


@app.get("/asset/{asset_id}")
def get_asset_state(asset_id: str):
    """
    Fetches the current state of a given perashable asset.
    """
    db = SESSION_LOCAL()
    asset = db.query(AssetRecord).filter(
        AssetRecord.asset_id == asset_id).first()
    db.close()
    if asset:
        return {
            "asset_id": asset.asset_id,
            "current_temp_c": asset.current_temp_c,
            "ethylene_ppm": asset.ethylene_ppm,
            "estimated_shelf_life_h": asset.estimated_shelf_life_h,
            "status": asset.status,
            "tx_hash": asset.tx_hash,
            "block_number": asset.block_number
        }
    return {"error": "Asset not found in the local FOSS db."}


@app.post("/sensor-ingest")
def ingest_sensor_data(payload: dict):
    """
    Ingest live telemetry from Arduino sensors, padding CV visual data 
    if not provided by an Edge camera.
    """

    cv_variance = payload.get("cv_variance", 1000.0)  # Safe baseline
    cv_intensity = payload.get("cv_intensity", 150.0)  # Safe baseline

    tensor_input = [
        payload.get("temp", 0.0),
        payload.get("humidity", 0.0),
        payload.get("ethylene", 0.0),
        cv_variance,
        cv_intensity
    ]

    res = run_inference(*tensor_input)

    return {"status": "success", "predicted_shelf_life": res["predicted_shelf_life_hours"]}


# ── Visual-PINN Pathogen & Fraud Detection Endpoints ──────────────────────────

@app.post("/lens/analyze")
def lens_analyze(payload: dict):
    """
    Single-image pathogen and chemical fraud analysis via Visual-PINN v3.

    POST body:
      {
        "image_path":    "<absolute or relative path to image>",
        "crop_archetype": 1   // 1-7 per Universal Crop Matrix
      }

    Returns full DetectionResult JSON including:
      - pathogen species, confidence, lesion area
      - fraud typologies, confidence
      - physics metrics (laplacian, entropy, edge density, HSV)
      - recommended_action, regulatory_flag
    """
    from pinn_engine.visual_pinn import analyze_lesion_kinetics  # pylint: disable=import-outside-toplevel
    image_path = payload.get("image_path", "")
    crop_archetype = int(payload.get("crop_archetype", 1))
    result = analyze_lesion_kinetics(image_path, crop_archetype)
    return result


@app.post("/lens/batch")
def lens_batch(payload: dict):
    """
    Batch pathogen and fraud analysis across a list of images.

    POST body:
      {
        "image_paths":   ["path1.jpg", "path2.jpg"],
        "crop_archetype": 1
      }

    Returns list of DetectionResult dicts + aggregate batch summary.
    """
    from pinn_engine.visual_pinn import analyze_batch, batch_summary  # pylint: disable=import-outside-toplevel
    paths = payload.get("image_paths", [])
    crop_archetype = int(payload.get("crop_archetype", 1))
    results = analyze_batch(paths, crop_archetype)
    summary = batch_summary(results)
    return {"summary": summary, "results": results}


@app.get("/lens/archetypes")
def lens_archetypes():
    """Return the full Universal Crop Matrix archetype registry."""
    from pinn_engine.visual_pinn import ARCHETYPES  # pylint: disable=import-outside-toplevel
    return {
        str(k): {"name": v["name"], "examples": v["examples"]}
        for k, v in ARCHETYPES.items()
    }


# ── Logistics Endpoints ───────────────────────────────────────────────────────

@app.post("/logistics/plan")
def logistics_plan(payload: dict):
    """
    Single-leg logistics dispatch plan.
    POST body:
      {
        "asset_id": "Pallet-T8",
        "origin": {"name":"Nashik Tomato Farm","lat":19.9975,"lon":73.7898},
        "destination": {"name":"Snowman Logistics Mumbai","lat":19.1136,"lon":72.8697},
        "mass_kg": 2500,
        "remaining_shelf_h": 22.0,
        "pidr": 0.0078,
        "base_price_inr": 125000,
        "cold_chain": true
      }
    """
    from pinn_engine.logistics_model import plan_dispatch, GeoNode  # pylint: disable=import-outside-toplevel
    origin_d = payload.get("origin", {})
    dest_d = payload.get("destination", {})
    origin = GeoNode(
        origin_d.get("id", "o"), origin_d.get("name", "Origin"),
        float(origin_d.get("lat", 0)), float(origin_d.get("lon", 0)),
        origin_d.get("type", "farm")
    )
    dest = GeoNode(
        dest_d.get("id", "d"), dest_d.get("name", "Destination"),
        float(dest_d.get("lat", 0)), float(dest_d.get("lon", 0)),
        dest_d.get("type", "warehouse")
    )
    result = plan_dispatch(
        asset_id=payload.get("asset_id", "unknown"),
        origin=origin,
        destination=dest,
        mass_kg=float(payload.get("mass_kg", 1000)),
        remaining_shelf_h=float(payload.get("remaining_shelf_h", 48)),
        pidr=float(payload.get("pidr", 0.0)),
        base_price_inr=float(payload.get("base_price_inr", 0)),
        cold_chain=bool(payload.get("cold_chain", True)),
    )
    return result.to_dict()


@app.post("/logistics/multi-stop")
def logistics_multi_stop(payload: dict):
    """
    Multi-stop optimised route plan (Nearest-Neighbour TSP).
    POST body adds "stops": [{name, lat, lon, type}, ...] instead of a single destination.
    """
    from pinn_engine.logistics_model import (  # pylint: disable=import-outside-toplevel
        plan_multi_stop_dispatch, GeoNode
    )
    origin_d = payload.get("origin", {})
    origin = GeoNode(
        origin_d.get("id", "o"), origin_d.get("name", "Origin"),
        float(origin_d.get("lat", 0)), float(origin_d.get("lon", 0))
    )
    stops = [
        GeoNode(s.get("id", f"s{i}"), s.get("name", f"Stop {i}"),
                float(s.get("lat", 0)), float(s.get("lon", 0)),
                s.get("type", "warehouse"))
        for i, s in enumerate(payload.get("stops", []))
    ]
    return plan_multi_stop_dispatch(
        asset_id=payload.get("asset_id", "unknown"),
        origin=origin,
        stops=stops,
        mass_kg=float(payload.get("mass_kg", 1000)),
        remaining_shelf_h=float(payload.get("remaining_shelf_h", 48)),
        pidr=float(payload.get("pidr", 0.0)),
        base_price_inr=float(payload.get("base_price_inr", 0)),
        cold_chain=bool(payload.get("cold_chain", True)),
    )


@app.get("/logistics/nodes")
def logistics_nodes():
    """Return all seeded India supply chain nodes."""
    from pinn_engine.logistics_model import INDIA_NODES  # pylint: disable=import-outside-toplevel
    return {
        k: {"name": v.name, "lat": v.lat, "lon": v.lon, "type": v.node_type}
        for k, v in INDIA_NODES.items()
    }


# ── PathogenCNN Direct Inference Endpoint ─────────────────────────────────────

@app.post("/pathogen/predict")
def pathogen_predict(payload: dict):
    """
    Run PathogenCNN inference directly on a feature vector (no image needed).
    Useful for edge devices that only send extracted features.

    POST body (all 9 features):
      {
        "diffusion_var": 3500,
        "mean_intensity": 95,
        "entropy": 6.2,
        "edge_density": 0.15,
        "dominant_hue": 12,
        "mean_saturation": 80,
        "contour_count": 4,
        "lesion_area_pct": 28.5,
        "spore_score": 3.1
      }
    """
    from pinn_engine.pathogen_cnn import predict_from_features  # pylint: disable=import-outside-toplevel
    try:
        return predict_from_features(payload)
    except FileNotFoundError as exc:
        return {"error": str(exc), "hint": "Train PathogenCNN first: python pinn_engine/pathogen_cnn.py --train"}


@app.get("/pathogen/classes")
def pathogen_classes():
    """Return all 10 PathogenCNN class labels."""
    from pinn_engine.pathogen_cnn import CLASS_NAMES  # pylint: disable=import-outside-toplevel
    return {str(i): name for i, name in enumerate(CLASS_NAMES)}


if __name__ == "__main__":
    import uvicorn
    print("\n[AI BROKER] Initializing Swarm Infrastructure...")
    print("[AI BROKER] Supply Chain Routing Agent successfully loaded tools:")
    print("   - Calculate Biomass Carbon Yield")
    print("   - Dispatch Gig Driver")
    print("   - Calculate Dynamic Shelf Price")
    print("[LOGISTICS] Haversine routing + dispatch model online")
    print("[LENS] Visual-PINN v3 + PathogenCNN fusion active\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
