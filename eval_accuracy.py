"""
Kinexica PINN Accuracy Evaluation
Computes R², RMSE, MAE, MRE, and sigma coverage against the full training matrix.
"""
from pinn_engine.train_pinn import PINNModel
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Load Data ──────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "pytorch_training_matrix.csv")
model_path = os.path.join(base_dir, "pinn_engine", "kinexica_pinn.pth")

print("=" * 60)
print("  KINEXICA PINN — Accuracy & Sigma Evaluation")
print("=" * 60)

df = pd.read_csv(data_path)

X_cols = ['temperature_c', 'humidity_percent', 'ethylene_ppm',
          'variance_of_laplacian', 'mean_intensity']
y_col = 'actual_shelf_life_hours'

X = df[X_cols].values.astype(np.float32)
y_true = df[y_col].values.astype(np.float32)

# ── Load Model ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINNModel().to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    print(f"\n✔  Weights loaded from: {model_path}")
else:
    print(f"\n⚠  No weights found — using untrained model (random init).")

model.eval()

# ── Inference ──────────────────────────────────────────────────────────────
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    _, pred_shelf = model(X_tensor)
    y_pred = pred_shelf.cpu().numpy().flatten()
    y_pred = np.maximum(y_pred, 0.0)   # clamp negatives

# ── Metrics ────────────────────────────────────────────────────────────────
residuals = y_true - y_pred
abs_err = np.abs(residuals)
rel_err = abs_err / (np.abs(y_true) + 1e-9)

mae = float(np.mean(abs_err))
rmse = float(np.sqrt(np.mean(residuals**2)))
mre = float(np.mean(rel_err)) * 100          # %
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r2 = float(1 - ss_res / ss_tot)
r2_pct = r2 * 100

sigma = float(np.std(residuals))
within_1s = float(np.mean(abs_err <= 1 * sigma)) * 100
within_2s = float(np.mean(abs_err <= 2 * sigma)) * 100
within_3s = float(np.mean(abs_err <= 3 * sigma)) * 100

y_range = float(y_true.max() - y_true.min())
rmse_norm = (rmse / y_range) * 100                 # % of output range

print(f"\n{'─'*60}")
print(f"  Dataset           : {len(df):,} samples  |  Features: {len(X_cols)}")
print(f"  Shelf-life range  : {y_true.min():.1f} – {y_true.max():.1f} hrs")
print(f"{'─'*60}")
print(f"\n  ── Regression Accuracy ──")
print(f"  R²  (Variance Explained)  : {r2:.4f}  →  {r2_pct:.2f}%")
print(f"  RMSE (hrs)                : {rmse:.4f}")
print(f"  RMSE (% of range)         : {rmse_norm:.2f}%")
print(f"  MAE  (hrs)                : {mae:.4f}")
print(f"  MRE  (Mean Relative Error): {mre:.2f}%")
print(f"  Accuracy (1 − MRE)        : {100 - mre:.2f}%")

print(f"\n  ── Sigma Coverage (Gaussian Process) ──")
print(f"  Residual σ (std dev)  : ± {sigma:.4f} hrs")
print(f"  Within 1σ             : {within_1s:.2f}%   (ideal ≈ 68.27%)")
print(f"  Within 2σ             : {within_2s:.2f}%   (ideal ≈ 95.45%)")
print(f"  Within 3σ             : {within_3s:.2f}%   (ideal ≈ 99.73%)")

print(f"\n  ── Classification Accuracy (Stable / Distressed) ──")
clf_true = (y_true >= 120).astype(int)   # 1 = Stable, 0 = Distressed
clf_pred = (y_pred >= 120).astype(int)
clf_acc = float(np.mean(clf_true == clf_pred)) * 100
print(f"  Threshold = 120 hrs")
print(f"  Classifier Accuracy       : {clf_acc:.2f}%")

print(f"\n{'═'*60}\n")
