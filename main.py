# pylint: disable=import-error, no-member, redefined-outer-name, unused-argument
"""
Main Orchestrator: FastAPI Backend & Zero-Cost SQLite Database.
Acts as the central Source of Truth.
"""
import asyncio
import os
import sys
import threading

import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from agent_broker.negotiation_agent import trigger_negotiation_swarm

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
    Mocks a PINN inference by extracting the expected output.
    """
    return row['actual_shelf_life_hours']

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
    from pinn_engine.inference import run_inference

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


if __name__ == "__main__":
    import uvicorn
    # Application Lifespan Events
    print("\n[AI BROKER] Initializing Swarm Infrastructure...")
    print("[AI BROKER] Supply Chain Routing Agent successfully loaded tools:")
    print("   - Calculate Biomass Carbon Yield")
    print("   - Dispatch Gig Driver")
    print("   - Calculate Dynamic Shelf Price\n")

    # FOSS tunnel binds to 8000 natively.
    uvicorn.run(app, host="127.0.0.1", port=8000)
