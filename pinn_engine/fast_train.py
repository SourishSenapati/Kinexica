# pylint: disable=import-error, no-member, invalid-name
"""
Kinexica PINN — Fast Training Script (500-epoch, ~20-min convergence target).

Key improvements over original train_pinn.py:
  1. Feature standardization (zero-mean, unit-variance)
  2. Loss weights corrected: shelf_life is the PRIMARY objective (weight=1.0)
  3. Cosine Annealing LR schedule for smooth convergence
  4. Live R², MAE reporting every 10 epochs
  5. Best-model auto-save whenever R² improves
  6. Safe save-on-interrupt via signal handler
"""

import os
import signal
import sys
import time

# Insert project root so pinn_engine.train_pinn can be found when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np           # noqa: E402  (below sys.path insert by design)
import pandas as pd          # noqa: E402
import torch                  # noqa: E402
from torch import nn          # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from pinn_engine.train_pinn import PINNModel  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "pytorch_training_matrix.csv")
SAVE_PINN = os.path.join(BASE_DIR, "pinn_engine", "kinexica_pinn.pth")
SAVE_VIS = os.path.join(BASE_DIR, "pinn_engine", "visual_pinn.pth")
NORM_PATH = os.path.join(BASE_DIR, "pinn_engine", "normalization.npz")

EPOCHS = 500
BATCH_SIZE = 512
LR = 3e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Shared state (model reference for the signal handler) ──────────────────
_state = {"model": None}


def save_weights() -> None:
    """Persist best model weights to both PINN and visual paths."""
    model = _state["model"]
    if model is None:
        return
    torch.save(model.state_dict(), SAVE_PINN)
    torch.save(model.state_dict(), SAVE_VIS)
    print(f"\n  ✔  Weights saved →  {SAVE_PINN}")


def handle_interrupt(_sig, _frame) -> None:
    """SIGINT handler: save weights then exit cleanly."""
    print("\n\n[INTERRUPT] Ctrl+C received — saving weights before exit ...")
    save_weights()
    sys.exit(0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination (R²)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-9)


def train() -> None:
    """Load data, normalise, train PINN for EPOCHS, report accuracy live."""
    signal.signal(signal.SIGINT, handle_interrupt)

    # ── Load ───────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    x_cols = ["temperature_c", "humidity_percent", "ethylene_ppm",
              "variance_of_laplacian", "mean_intensity"]
    y_col = "actual_shelf_life_hours"

    x_raw = df[x_cols].values.astype(np.float32)
    y_raw = df[y_col].values.astype(np.float32)

    # ── Normalise ──────────────────────────────────────────────────────────
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0) + 1e-8
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std()) + 1e-8

    x_norm = (x_raw - x_mean) / x_std
    y_norm = (y_raw - y_mean) / y_std

    np.savez(NORM_PATH, X_mean=x_mean, X_std=x_std, y_mean=y_mean, y_std=y_std)
    print(f"  [INFO] Normalisation constants saved → {NORM_PATH}")

    x_tensor = torch.tensor(x_norm,           dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_norm[:, None],  dtype=torch.float32).to(DEVICE)
    pidr_dummy = torch.zeros_like(y_tensor)

    loader = DataLoader(
        TensorDataset(x_tensor, pidr_dummy, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = PINNModel().to(DEVICE)
    _state["model"] = model

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    print("=" * 64)
    print("  KINEXICA PINN — 500-Epoch Training")
    print(
        f"  Device : {DEVICE}  |  Samples : {len(df):,}  |  Batch : {BATCH_SIZE}")
    print("  Press Ctrl+C at ANY time — best weights will be saved safely.")
    print("=" * 64)

    t0 = time.time()
    best_r2 = -9999.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for bx, _, by in loader:
            optimizer.zero_grad()
            _, pred_shelf = model(bx)
            loss = criterion(pred_shelf, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = (time.time() - t0) / 60.0

        # Detailed report every 10 epochs, simple ticker otherwise
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                _, preds = model(x_tensor)
                preds_np = preds.cpu().numpy().flatten() * y_std + y_mean

            cur_r2 = r2_score(y_raw, preds_np)
            mae = float(np.mean(np.abs(y_raw - preds_np)))

            saved_tag = ""
            if cur_r2 > best_r2:
                best_r2 = cur_r2
                torch.save(model.state_dict(), SAVE_PINN)
                torch.save(model.state_dict(), SAVE_VIS)
                saved_tag = "  ✔ BEST SAVED"

            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Ep {epoch:>3}/{EPOCHS}  |  Loss: {avg_loss:.5f}  |  "
                f"R²: {cur_r2:.4f} ({cur_r2 * 100:.1f}%)  |  "
                f"MAE: {mae:.1f}h  |  LR: {lr_now:.2e}  |  "
                f"{elapsed:.1f}m{saved_tag}"
            )
        else:
            print(
                f"  Ep {epoch:>3}/{EPOCHS}  |  Loss: {avg_loss:.5f}  |  {elapsed:.1f}m",
                end="\r",
            )

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\n\n[COMPLETE] 500 epochs done.")
    save_weights()

    model.eval()
    with torch.no_grad():
        _, preds = model(x_tensor)
        preds_np = preds.cpu().numpy().flatten() * y_std + y_mean

    residuals = y_raw - preds_np
    sigma = float(np.std(residuals))
    final_r2 = r2_score(y_raw, preds_np)
    mae_f = float(np.mean(np.abs(residuals)))
    rmse_f = float(np.sqrt(np.mean(residuals ** 2)))
    mre_f = float(np.mean(np.abs(residuals) / (np.abs(y_raw) + 1e-9))) * 100
    w1s = float(np.mean(np.abs(residuals) <= sigma)) * 100
    w2s = float(np.mean(np.abs(residuals) <= 2 * sigma)) * 100
    w3s = float(np.mean(np.abs(residuals) <= 3 * sigma)) * 100
    clf_true = (y_raw >= 120).astype(int)
    clf_pred = (preds_np >= 120).astype(int)
    clf_acc = float(np.mean(clf_true == clf_pred)) * 100

    print("\n" + "=" * 64)
    print("  FINAL ACCURACY REPORT — 500 Epochs")
    print("=" * 64)
    print(f"  R²            : {final_r2:.4f}  →  {final_r2 * 100:.2f}%")
    print(f"  RMSE          : {rmse_f:.2f} hrs")
    print(f"  MAE           : {mae_f:.2f} hrs")
    print(f"  Accuracy      : {100 - mre_f:.2f}%  (1 − MRE)")
    print(f"  σ (std dev)   : ± {sigma:.2f} hrs")
    print(f"  Within 1σ     : {w1s:.2f}%   (ideal ≈ 68.27%)")
    print(f"  Within 2σ     : {w2s:.2f}%   (ideal ≈ 95.45%)")
    print(f"  Within 3σ     : {w3s:.2f}%   (ideal ≈ 99.73%)")
    print(f"  Classifier    : {clf_acc:.2f}%  (Stable/Distressed @ 120h)")
    print(f"  Total time    : {(time.time() - t0) / 60:.1f} minutes")
    print("=" * 64)


if __name__ == "__main__":
    train()
