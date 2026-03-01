# pylint: disable=import-error, no-member, invalid-name
"""
Inference script for the Physics-Informed Neural Network engine.
"""

import os
import torch
from pinn_engine.train_pinn import PINNModel


def run_inference(
    temp: float, humidity: float, ethylene: float, cv_variance: float = 1000.0,
    cv_intensity: float = 100.0
) -> dict:
    """
    Run the trained Physics-Informed Neural Network to predict PIDR 
    and estimated shelf life based on environmental telemetry and vision features.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "pinn_engine", "kinexica_pinn.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINNModel().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[WARNING] Weights not found at {model_path}. "
              "Using untrained PINN generator layer.")

    model.eval()

    with torch.no_grad():
        X = torch.tensor([[temp, humidity, ethylene, cv_variance, cv_intensity]],
                         dtype=torch.float32).to(device)
        pred_pidr, pred_shelf = model(X)

    return {
        "predicted_pidr": float(pred_pidr.item()),
        "predicted_shelf_life_hours": max(0.0, float(pred_shelf.item()))
    }


if __name__ == "__main__":
    # Test inference output
    res = run_inference(temp=25.0, humidity=80.0, ethylene=12.5,
                        cv_variance=2500.0, cv_intensity=150.0)
    print("PINN Inference Output:", res)
