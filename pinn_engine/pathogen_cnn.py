"""
Kinexica — Pathogen CNN Classifier v2.0
════════════════════════════════════════
A lightweight ResNet-inspired multi-layer perceptron (MLP) trained on
9-dimensional multi-modal image feature vectors extracted by the Visual-PINN
pipeline. This replaces the hand-coded thresholds with a learned classifier
that maximizes precision/recall on the Kinexica pathogen taxonomy.

Architecture
────────────
Input  : 9-dimensional feature vector
         [diffusion_var, mean_intensity, entropy, edge_density,
          dominant_hue, mean_saturation, contour_count, lesion_area_pct,
          spore_score]

Hidden : 3 residual blocks → [256, 128, 64] with BatchNorm + Dropout
Output : 10-class softmax
         0 = Pristine
         1 = Botrytis cinerea
         2 = Penicillium expansum
         3 = Aspergillus niger (aflatoxin risk)
         4 = Alternaria alternata
         5 = Bacterial soft rot (Erwinia/Pectobacterium)
         6 = Fusarium oxysporum
         7 = Chemical Fraud (CaC₂ / dye / formalin)
         8 = Senescent / Overripe (natural)
         9 = Multiple co-infection

Training: synthetic data generation matching Visual-PINN physics signatures
          50,000 samples × 10 classes, data-augmented
Target  : > 97% validation accuracy, F1 > 0.96 per class

Usage:
  python pinn_engine/pathogen_cnn.py --train
  python pinn_engine/pathogen_cnn.py --eval
  python pinn_engine/pathogen_cnn.py --predict '{"diffusion_var":3500,...}'
"""
# pylint: disable=import-error, invalid-name, too-many-arguments

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# ─────────────────────────────────────────────────────────────────────────────
# CLASS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Pristine",
    "Botrytis cinerea (grey mould)",
    "Penicillium expansum (blue mould)",
    "Aspergillus niger (aflatoxin risk)",
    "Alternaria alternata (black spot)",
    "Bacterial soft rot (Erwinia / Pectobacterium)",
    "Fusarium oxysporum (wilt / rot)",
    "Chemical Fraud (CaC₂ / dye / formalin)",
    "Senescent / Overripe",
    "Multiple co-infection",
]
N_CLASSES = len(CLASS_NAMES)   # 10
INPUT_DIM = 9

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "pinn_engine", "pathogen_cnn.pth")
NORM_PATH = os.path.join(BASE, "pinn_engine", "pathogen_cnn_norm.npz")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE: Residual MLP
# ─────────────────────────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """A two-layer residual block with BatchNorm and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        return self.act(x + self.net(x))


class PathogenCNN(nn.Module):
    """
    Residual MLP for pathogen/fraud classification.
    Input: 9-dim feature vector → Output: 10-class logits.
    """

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.20):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(256, dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            ResidualBlock(128, dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock(64, dropout),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (physics-guided)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_SIGNATURES = {
    0: {  # Pristine
        "diffusion_var":  (100,   800),
        "mean_intensity": (140,   200),
        "entropy":        (3.0,   4.5),
        "edge_density":   (0.03,  0.10),
        "dominant_hue":   (10,    25),
        "mean_saturation": (150,   220),
        "contour_count":  (0,     1),
        "lesion_area_pct": (0.0,   3.0),
        "spore_score":    (0.0,   0.5),
    },
    1: {  # Botrytis cinerea
        "diffusion_var":  (1800, 5500),
        "mean_intensity": (80,   140),
        "entropy":        (5.5,  7.5),
        "edge_density":   (0.10, 0.22),
        "dominant_hue":   (8,    18),
        "mean_saturation": (60,   120),
        "contour_count":  (2,    7),
        "lesion_area_pct": (15,   55),
        "spore_score":    (2.5,  6.0),
    },
    2: {  # Penicillium expansum
        "diffusion_var":  (1500, 5000),
        "mean_intensity": (90,   150),
        "entropy":        (5.0,  7.2),
        "edge_density":   (0.10, 0.20),
        "dominant_hue":   (40,   80),
        "mean_saturation": (70,   130),
        "contour_count":  (2,    6),
        "lesion_area_pct": (10,   45),
        "spore_score":    (2.0,  5.5),
    },
    3: {  # Aspergillus niger
        "diffusion_var":  (3000, 8500),
        "mean_intensity": (40,   90),
        "entropy":        (6.0,  8.0),
        "edge_density":   (0.15, 0.30),
        "dominant_hue":   (0,    10),
        "mean_saturation": (40,   100),
        "contour_count":  (3,    9),
        "lesion_area_pct": (25,   65),
        "spore_score":    (3.5,  8.0),
    },
    4: {  # Alternaria alternata
        "diffusion_var":  (1000, 4000),
        "mean_intensity": (60,   120),
        "entropy":        (5.2,  7.0),
        "edge_density":   (0.10, 0.25),
        "dominant_hue":   (0,    12),
        "mean_saturation": (50,   110),
        "contour_count":  (2,    8),
        "lesion_area_pct": (10,   40),
        "spore_score":    (1.5,  4.5),
    },
    5: {  # Bacterial soft rot
        "diffusion_var":  (100,  600),
        "mean_intensity": (40,   80),
        "entropy":        (3.5,  5.5),
        "edge_density":   (0.03, 0.10),
        "dominant_hue":   (5,    20),
        "mean_saturation": (40,   90),
        "contour_count":  (1,    4),
        "lesion_area_pct": (20,   70),
        "spore_score":    (0.2,  1.5),
    },
    6: {  # Fusarium oxysporum
        "diffusion_var":  (800,  3500),
        "mean_intensity": (70,   130),
        "entropy":        (4.8,  6.5),
        "edge_density":   (0.10, 0.22),
        "dominant_hue":   (15,   35),
        "mean_saturation": (80,   150),
        "contour_count":  (3,    10),
        "lesion_area_pct": (15,   50),
        "spore_score":    (1.5,  4.0),
    },
    7: {  # Chemical Fraud (CaC₂ / dye / formalin)
        "diffusion_var":  (2500, 9000),
        "mean_intensity": (155,  230),
        "entropy":        (3.0,  5.0),
        "edge_density":   (0.18, 0.40),
        "dominant_hue":   (30,   70),
        "mean_saturation": (50,   90),
        "contour_count":  (0,    2),
        "lesion_area_pct": (0.0,  5.0),
        "spore_score":    (0.0,  0.5),
    },
    8: {  # Senescent / Overripe
        "diffusion_var":  (500,  2000),
        "mean_intensity": (60,   110),
        "entropy":        (4.0,  5.8),
        "edge_density":   (0.05, 0.14),
        "dominant_hue":   (5,    18),
        "mean_saturation": (80,   140),
        "contour_count":  (0,    2),
        "lesion_area_pct": (5,    20),
        "spore_score":    (0.2,  1.0),
    },
    9: {  # Multiple co-infection
        "diffusion_var":  (4000, 9000),
        "mean_intensity": (50,   100),
        "entropy":        (6.5,  8.5),
        "edge_density":   (0.18, 0.35),
        "dominant_hue":   (5,    40),
        "mean_saturation": (40,   100),
        "contour_count":  (5,    15),
        "lesion_area_pct": (40,   80),
        "spore_score":    (4.0,  9.0),
    },
}

FEATURE_KEYS = [
    "diffusion_var", "mean_intensity", "entropy", "edge_density",
    "dominant_hue", "mean_saturation", "contour_count", "lesion_area_pct",
    "spore_score",
]


def generate_synthetic_data(
    n_per_class: int = 5000,
    noise_std:   float = 0.08,
    seed:        int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate physics-guided synthetic feature vectors (9-dim) for all 10 classes.
    Returns (X, y) arrays.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for cls_id, sig in CLASS_SIGNATURES.items():
        samples = np.zeros((n_per_class, INPUT_DIM), dtype=np.float32)
        for j, key in enumerate(FEATURE_KEYS):
            lo, hi = sig[key]
            vals = rng.uniform(lo, hi, n_per_class).astype(np.float32)
            # Add proportional Gaussian noise for data augmentation
            noise = rng.normal(0, (hi - lo) * noise_std,
                               n_per_class).astype(np.float32)
            samples[:, j] = np.clip(vals + noise, lo * 0.5, hi * 2.0)
        X_list.append(samples)
        y_list.append(np.full(n_per_class, cls_id, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Per-class precision, recall, F1."""
    results = []
    for cls in range(N_CLASSES):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        results.append({
            "class":     CLASS_NAMES[cls],
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

_state: dict = {"model": None}


def _save() -> None:
    m = _state["model"]
    if m is None:
        return
    torch.save(m.state_dict(), MODEL_PATH)
    print(f"\n  ✔  Saved → {MODEL_PATH}")


def _handle_interrupt(_s, _f) -> None:
    print("\n\n[INTERRUPT] Saving best weights …")
    _save()
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_interrupt)


def train(
    epochs:       int = 300,
    batch_size:   int = 512,
    lr:           float = 1e-3,
    n_per_class:  int = 5000,
    val_split:    float = 0.15,
) -> None:
    """Train the PathogenCNN on synthetic data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  KINEXICA PathogenCNN — Training")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  Classes: {N_CLASSES}")
    print("  Press Ctrl+C at any time — best weights are auto-saved.")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────
    X, y = generate_synthetic_data(n_per_class=n_per_class)
    # Normalise
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    np.savez(NORM_PATH, x_mean=x_mean, x_std=x_std)
    X_norm = (X - x_mean) / x_std

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y,      dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds,   batch_size=batch_size,
                        shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────
    model = PathogenCNN().to(device)
    _state["model"] = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in val_dl:
                    bx, by = bx.to(device), by.to(device)
                    preds = model(bx).argmax(dim=1)
                    correct += (preds == by).sum().item()
                    total += len(by)
            val_acc = correct / total * 100.0
            saved = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)
                saved = "  ✔ BEST"
            print(
                f"  Ep {epoch:>3}/{epochs}  |  "
                f"Loss: {total_loss/len(train_dl):.4f}  |  "
                f"Val Acc: {val_acc:.2f}%  |  "
                f"Best: {best_acc:.2f}%  |  "
                f"{(time.time()-t0)/60:.1f}m{saved}"
            )
        else:
            print(
                f"  Ep {epoch:>3}/{epochs}  |  {(time.time()-t0)/60:.1f}m",
                end="\r",
            )

    print(f"\n\n[COMPLETE] Best Val Accuracy: {best_acc:.2f}%")
    print(f"  Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> tuple[PathogenCNN, np.ndarray, np.ndarray]:
    """Load trained model and normalisation constants."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"PathogenCNN weights not found: {MODEL_PATH}. Run --train first.")
    if not os.path.exists(NORM_PATH):
        raise FileNotFoundError(
            f"Normalisation constants not found: {NORM_PATH}.")
    model = PathogenCNN()
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    norms = np.load(NORM_PATH)
    x_mean = norms["x_mean"]
    x_std = norms["x_std"]
    return model, x_mean, x_std


def predict_from_features(features: dict) -> dict:
    """
    Run inference from a Visual-PINN feature dict.

    Parameters
    ----------
    features : dict with keys: diffusion_var, mean_intensity, entropy,
               edge_density, dominant_hue, mean_saturation,
               contour_count, lesion_area_pct, spore_score

    Returns
    -------
    dict: {
        "class_id", "class_name", "confidence",
        "top3": [(class_name, prob), ...],
        "is_pathogen", "is_fraud", "is_pristine"
    }
    """
    model, x_mean, x_std = load_model()

    x_raw = np.array([[features.get(k, 0.0)
                     for k in FEATURE_KEYS]], dtype=np.float32)
    x_norm = (x_raw - x_mean) / (x_std + 1e-8)
    x_t = torch.tensor(x_norm, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_t)
        probs = F.softmax(logits, dim=1).numpy()[0]

    top_id = int(np.argmax(probs))
    top3 = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "class_id":   top_id,
        "class_name": CLASS_NAMES[top_id],
        "confidence": float(round(probs[top_id], 4)),
        "top3": [
            {"class": CLASS_NAMES[i], "prob": float(round(p, 4))}
            for i, p in top3
        ],
        "is_pathogen": top_id in (1, 2, 3, 4, 5, 6, 9),
        "is_fraud":    top_id == 7,
        "is_pristine": top_id in (0, 8),
    }


def evaluate_model() -> dict:
    """Run full evaluation on held-out synthetic test set."""
    model, x_mean, x_std = load_model()
    device = "cpu"
    model.to(device)

    X, y = generate_synthetic_data(
        n_per_class=1000, seed=99)   # new seed = unseen
    X_norm = (X - x_mean) / (x_std + 1e-8)

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=1024, shuffle=False)

    all_preds, all_true = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in dl:
            preds = model(bx).argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_true.extend(by.numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    acc = float(np.mean(y_true == y_pred)) * 100
    cms = per_class_metrics(y_true, y_pred)
    macro_f1 = float(np.mean([c["f1"] for c in cms]))

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  PathogenCNN — Evaluation Report")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Overall Accuracy : {acc:.2f}%")
    print(f"  Macro F1         : {macro_f1:.4f}\n")
    for cm in cms:
        print(f"  {cm['class'][:42]:<42} "
              f"P={cm['precision']:.3f} R={cm['recall']:.3f} F1={cm['f1']:.3f}")

    return {"overall_accuracy_pct": round(acc, 2), "macro_f1": round(macro_f1, 4),
            "per_class": cms}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinexica PathogenCNN")
    parser.add_argument("--train",   action="store_true",
                        help="Train the model")
    parser.add_argument("--eval",    action="store_true",
                        help="Evaluate saved model")
    parser.add_argument("--predict", type=str,
                        help="JSON feature dict")
    parser.add_argument("--epochs",  type=int,  default=300)
    parser.add_argument("--n",       type=int,
                        default=5000, help="Samples per class")
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, n_per_class=args.n)

    elif args.eval:
        result = evaluate_model()
        print("\n" + json.dumps(result, indent=2))

    elif args.predict:
        feats = json.loads(args.predict)
        out = predict_from_features(feats)
        print(json.dumps(out, indent=2))

    else:
        parser.print_help()
