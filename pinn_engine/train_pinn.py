# pylint: disable=import-error, no-member, invalid-name, unused-variable
"""
Training script for the PINN engine.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network predicting PIDR and shelf life.
    """

    def __init__(self):
        super(PINNModel, self).__init__()
        # inputs: Temp, Humidity, Ethylene, CV_Variance, CV_Intensity
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out_pidr = nn.Linear(64, 1)        # Output 1: PIDR (Proxy for k)
        self.out_shelf = nn.Linear(64, 1)       # Output 2: Shelf life

        self.relu = nn.ReLU()

    def forward(self, x):
        """ Forward pass """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        pidr = self.out_pidr(x)
        shelf = self.out_shelf(x)
        return pidr, shelf


def train_model():
    """
    Load data, initialize model, and run epochs. Save weights on interruption.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "pytorch_training_matrix.csv")

    if not os.path.exists(data_path):
        print(f"Dataset missing at {data_path}. Please generate it first.")
        return

    print("Loading PINN data...")
    df = pd.read_csv(data_path)

    # Input Features
    X = df[['temperature_c', 'humidity_percent', 'ethylene_ppm',
            'variance_of_laplacian', 'mean_intensity']].values

    # Targets: PIDR, Shelf Life
    y_pidr = df[['pidr']].values
    y_shelf = df[['actual_shelf_life_hours']].values

    # Convert vectors to PyTorch tensors and push to device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_pidr_tensor = torch.tensor(y_pidr, dtype=torch.float32).to(device)
    y_shelf_tensor = torch.tensor(y_shelf, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_pidr_tensor, y_shelf_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = PINNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_mse = nn.MSELoss()

    model_save_path_pinn = os.path.join(
        base_dir, "pinn_engine", "kinexica_pinn.pth")
    model_save_path_visual = os.path.join(
        base_dir, "pinn_engine", "visual_pinn.pth")

    epochs = 500
    print(
        f"Beginning 5-Tier CV+PINN training on {device}... (Ctrl+C to stop and save)")

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y_pidr, batch_y_shelf in loader:
                optimizer.zero_grad()

                # Forward pass
                pred_pidr, pred_shelf = model(batch_X)

                # Data-driven Loss
                loss_pidr = criterion_mse(pred_pidr, batch_y_pidr)
                loss_shelf = criterion_mse(pred_shelf, batch_y_shelf)

                # Total loss (can heavily weigh PIDR vs Shelf Life)
                total_loss = loss_pidr + (loss_shelf * 0.01)

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\n\n[WARNING] Training Interrupted via Keyboard.")
        print("Triggering SAVE ON INTERRUPT protocol. Saving weights...")
        torch.save(model.state_dict(), model_save_path_pinn)
        torch.save(model.state_dict(), model_save_path_visual)
        print("Weights saved successfully. Exiting gracefully.")
        return

    # Save standard upon completion
    print("\nTraining Complete! Saving dual-weights...")
    torch.save(model.state_dict(), model_save_path_pinn)
    torch.save(model.state_dict(), model_save_path_visual)
    print("Done.")


if __name__ == "__main__":
    train_model()
