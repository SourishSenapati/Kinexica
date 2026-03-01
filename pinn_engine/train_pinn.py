# pylint: disable=import-error, no-member, invalid-name, unused-variable
"""
Training script for the PINN engine.
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Force CUDA
device = torch.device('cuda')


class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network predicting PIDR and shelf life.
    """

    def __init__(self):
        super(PINNModel, self).__init__()
        # inputs: Temp, Humidity, Ethylene, CV_Variance, CV_Intensity
        self.bn_in = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(5, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out_pidr = nn.Linear(128, 1)        # Output 1: PIDR (Proxy for k)
        self.out_shelf = nn.Linear(128, 1)       # Output 2: Shelf life

        self.relu = nn.ReLU()

    def forward(self, x):
        """ Forward pass """
        x = self.bn_in(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    def mape_loss(pred, target):
        return torch.mean(torch.abs((pred - target) / (target.abs() + 1e-5)))

    model_save_path_pinn = os.path.join(
        base_dir, "pinn_engine", "kinexica_pinn.pth")
    model_save_path_visual = os.path.join(
        base_dir, "pinn_engine", "visual_pinn.pth")

    epochs = 20000
    print(
        f"Beginning 5-Tier CV+PINN training on {device}... (Ctrl+C to stop and save)")

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            # Approximation proxy for "accuracy"
            correct_preds = 0
            total_preds = 0

            for batch_X, batch_y_pidr, batch_y_shelf in loader:
                optimizer.zero_grad()

                # Forward pass
                pred_pidr, pred_shelf = model(batch_X)

                # Data-driven Loss
                loss_pidr = mape_loss(pred_pidr, batch_y_pidr)
                loss_shelf = mape_loss(pred_shelf, batch_y_shelf)

                # Total loss (can heavily weigh PIDR vs Shelf Life)
                total_loss = loss_pidr + loss_shelf

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            # Evaluate exact deterministic accuracy outside of train() running stats
            model.eval()
            with torch.no_grad():
                correct_preds = 0
                total_preds = 0
                for batch_X, batch_y_pidr, batch_y_shelf in loader:
                    pred_pidr, pred_shelf = model(batch_X)
                    mask_pidr = torch.abs(
                        pred_pidr - batch_y_pidr) < (0.05 * batch_y_pidr.abs() + 1e-5)
                    mask_shelf = torch.abs(
                        pred_shelf - batch_y_shelf) < (0.05 * batch_y_shelf.abs() + 1e-5)
                    correct_preds += (mask_pidr & mask_shelf).sum().item()
                    total_preds += batch_X.size(0)

            avg_loss = epoch_loss / len(loader)
            accuracy = (correct_preds / total_preds) * \
                100.0 if total_preds > 0 else 0.0

            scheduler.step(avg_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Total Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}%"
            )

            if accuracy >= 99.999:
                print(f"Reached 99.999% accuracy! Stopping at epoch {epoch+1}")
                break

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
