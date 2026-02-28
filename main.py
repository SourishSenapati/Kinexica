"""
Main orchestrator script to monitor asset degradation and trigger the swarm.
"""
import os
import sys
import time

import pandas as pd

from agent_broker.negotiation_agent import trigger_negotiation_swarm

# Add current directory to path so we can import modules properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def mock_pinn_inference(row):
    """
    Mock inference function replacing the active PINN model for the sake of setting up the swarm.
    It takes an environmental state and simulates what the PINN would output.
    Our synthetic data already has 'actual_shelf_life_hours'.
    """
    return row['actual_shelf_life_hours']


def monitor_asset(csv_path="data/synthetic_sensor_data.csv"):
    """
    Simulates a real-time monitoring loop ingesting data (e.g., from an edge sensor/PINN).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_csv_path = os.path.join(base_dir, csv_path)

    print("[SYSTEM] Starting Continuous Monitoring Loop...")

    if not os.path.exists(full_csv_path):
        print(
            f"Error: {full_csv_path} not found. Please run sensor_simulator.py first.")
        return

    df = pd.read_csv(full_csv_path)

    for _, row in df.iterrows():
        # 1. Ingest predicted shelf life from the (mock) PINN
        predicted_shelf_life = mock_pinn_inference(row)

        # Log to terminal (could be sent to dashboard later)
        print(f"Hour {int(row['timestamp_hour'])} - Temp: {row['temperature_c']}°C, "
              f"Ethylene: {row['ethylene_ppm']} ppm | "
              f"Estimated Shelf Life: {predicted_shelf_life:.2f} hrs")

        # 2. Threshold Breach Logic
        # The moment predicted_shelf_life drops below 120.0 hours, halt and trigger
        if predicted_shelf_life < 120.0:
            print("\n================================================")
            print("[ALERT] THERMODYNAMIC ANOMALY DETECTED.")
            print(
                f"[ALERT] Predicted shelf life ({predicted_shelf_life:.2f}h) "
                f"dropped below critical threshold (120h)."
            )
            print("[ALERT] Halting standard monitoring loop...")
            print("================================================\n")

            # 3. Data Extraction & Packaging
            payload = {
                "asset_id": "Pallet-4B-Tomatoes",
                "current_temp_c": row['temperature_c'],
                "peak_ethylene_ppm": row['ethylene_ppm'],
                "estimated_hours_remaining": predicted_shelf_life,
                "timestamp_hour": row['timestamp_hour']
            }

            # 4. Handoff to Swarm
            trigger_negotiation_swarm(payload)

            # Break the loop because asset is liquidated
            break

        # Simulate real-time delay (fast-forwarded for testing)
        time.sleep(0.05)


if __name__ == "__main__":
    monitor_asset()
