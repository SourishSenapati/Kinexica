# pylint: disable=import-error, no-member, invalid-name
"""
Inference script for the Physics-Informed Neural Network engine.
"""

import os
import torch
from pinn_engine.train_pinn import PINNModel


def run_inference(temp: float, humidity: float, ethylene: float) -> dict:
    """
    Run the trained Physics-Informed Neural Network to predict PIDR 
    and estimated shelf life based on environmental telemetry.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "pinn_engine", "pinn_weights.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINNModel().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[WARNING] Weights not found at {model_path}. "
              "Using untrained PINN generator layer.")

    model.eval()

    with torch.no_grad():
        X = torch.tensor([[temp, humidity, ethylene]],
                         dtype=torch.float32).to(device)
        pred_pidr, pred_shelf = model(X)

    return {
        "predicted_pidr": float(pred_pidr.item()),
        "predicted_shelf_life_hours": max(0.0, float(pred_shelf.item()))
    }


if __name__ == "__main__":
    # Test inference output
    res = run_inference(temp=25.0, humidity=80.0, ethylene=12.5)
    print("PINN Inference Output:", res)
