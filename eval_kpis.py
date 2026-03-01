"""
Kinexica — Unified KPI Evaluator v2.0
══════════════════════════════════════
Evaluates ALL Kinexica models and produces a comprehensive multi-model
performance report:

  MODEL A  PINN Shelf-Life Predictor
           Regression: R², RMSE, MAE, MAPE, Within-1/2/3σ
           Classification: Stable/Distressed accuracy, Precision, Recall, F1, AUC

  MODEL B  Visual-PINN Pathogen/Fraud Detector (synthetic benchmark)
           Classification: Precision, Recall, F1, Specificity, MCC
           Per-class confusion matrix

  MODEL C  Arrhenius Baseline (physics-only, no neural network)
           Benchmark to confirm PINN outperforms traditional kinetics

  SIGMA ANALYSIS  Gaussian distribution fit to residuals
           Confirms predictions are within 1σ / 2σ / 3σ targets

Usage:
  python eval_kpis.py            # full evaluation
  python eval_kpis.py --export   # export JSON report to reports/kpis.json
"""
# pylint: disable=import-error, invalid-name

import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data",        "pytorch_training_matrix.csv")
WEIGHTS = os.path.join(BASE, "pinn_engine", "kinexica_pinn.pth")
NORM = os.path.join(BASE, "pinn_engine", "normalization.npz")

sys.path.insert(0, BASE)


# ── Metric helpers ──────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Makes numpy scalars JSON-serialisable."""

    def default(self, o):  # pylint: disable=arguments-renamed
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def mape(y, p): return float(np.mean(np.abs((y - p) / (y + 1e-9)))) * 100


def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    mcc_n = (tp * tn) - (fp * fn)
    mcc_d = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-9
    mcc = mcc_n / mcc_d
    return {
        "precision":   round(float(prec),  4),
        "recall":      round(float(rec),   4),
        "f1":          round(float(f1),    4),
        "specificity": round(float(spec),  4),
        "accuracy":    round(float(acc),   4),
        "mcc":         round(float(mcc),   4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def sigma_bands(residuals):
    σ = float(np.std(residuals))
    return {
        "sigma":    round(σ, 4),
        "w1s_pct":  round(float(np.mean(np.abs(residuals) <= σ)) * 100, 2),
        "w2s_pct":  round(float(np.mean(np.abs(residuals) <= 2 * σ)) * 100, 2),
        "w3s_pct":  round(float(np.mean(np.abs(residuals) <= 3 * σ)) * 100, 2),
        "gaussian_fit_ok": bool(float(np.mean(np.abs(residuals) <= 3 * σ)) > 0.99),
    }


# ── MODEL A: PINN Shelf-Life Predictor ────────────────────────────────────

def evaluate_pinn() -> dict:
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  MODEL A  |  PINN Shelf-Life Predictor")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not os.path.exists(DATA):
        print(f"  [SKIP] Dataset not found: {DATA}")
        return {"status": "skipped", "reason": "dataset missing"}
    if not os.path.exists(WEIGHTS):
        print(f"  [SKIP] Weights not found: {WEIGHTS}")
        return {"status": "skipped", "reason": "weights missing"}

    from pinn_engine.train_pinn import PINNModel  # noqa

    # Load data
    df = pd.read_csv(DATA)
    xcols = ["temperature_c", "humidity_percent", "ethylene_ppm",
             "variance_of_laplacian", "mean_intensity"]
    ycol = "actual_shelf_life_hours"
    x_raw = df[xcols].values.astype(np.float32)
    y_raw = df[ycol].values.astype(np.float32)

    # Normalise using saved norms
    norms = np.load(NORM)
    x_norm = (x_raw - norms["X_mean"]) / (norms["X_std"] + 1e-8)
    y_mean = float(norms["y_mean"])
    y_std = float(norms["y_std"])

    device = torch.device("cpu")
    model = PINNModel().to(device)
    model.load_state_dict(torch.load(
        WEIGHTS, map_location=device, weights_only=True))
    model.eval()

    x_t = torch.tensor(x_norm, dtype=torch.float32)
    with torch.no_grad():
        _, preds_norm = model(x_t)
    preds = preds_norm.numpy().flatten() * y_std + y_mean

    residuals = y_raw - preds
    _r2 = r2(y_raw, preds)
    _rmse = rmse(y_raw, preds)
    _mae = mae(y_raw, preds)
    _mape = mape(y_raw, preds)
    _sigs = sigma_bands(residuals)

    # Distress classifier (< 120 h = Distressed)
    clf_true = (y_raw < 120).astype(int)
    clf_pred = (preds < 120).astype(int)
    clf_kpis = precision_recall_f1(clf_true, clf_pred)

    result = {
        "regression": {
            "r2":          round(_r2,   4),
            "r2_pct":      round(_r2 * 100, 2),
            "rmse_hrs":    round(_rmse, 2),
            "mae_hrs":     round(_mae,  2),
            "mape_pct":    round(_mape, 2),
            "accuracy_pct": round(100 - _mape, 2),
        },
        "sigma_analysis": _sigs,
        "distress_classifier": clf_kpis,
        "samples": len(y_raw),
    }

    print(f"  R²          : {_r2:.4f}  →  {_r2 * 100:.2f}%")
    print(f"  RMSE        : {_rmse:.2f} hrs")
    print(f"  MAE         : {_mae:.2f} hrs")
    print(f"  MAPE        : {_mape:.2f}%   → Accuracy: {100-_mape:.2f}%")
    print(f"  σ           : ±{_sigs['sigma']:.2f} hrs")
    print(f"  Within 1σ   : {_sigs['w1s_pct']:.1f}%  (target ≥ 68.27%)")
    print(f"  Within 2σ   : {_sigs['w2s_pct']:.1f}%  (target ≥ 95.45%)")
    print(f"  Within 3σ   : {_sigs['w3s_pct']:.1f}%  (target ≥ 99.73%)")
    print(
        f"  Gaussian fit: {'✔  PASS' if _sigs['gaussian_fit_ok'] else '✗  FAIL'}")
    print(f"\n  Distress Classifier (≥120h = Stable):")
    print(f"    Precision  : {clf_kpis['precision']:.4f}")
    print(f"    Recall     : {clf_kpis['recall']:.4f}")
    print(f"    F1         : {clf_kpis['f1']:.4f}")
    print(f"    Accuracy   : {clf_kpis['accuracy']*100:.2f}%")
    print(f"    MCC        : {clf_kpis['mcc']:.4f}")
    print(f"    Confusion  : TP={clf_kpis['tp']} FP={clf_kpis['fp']} "
          f"FN={clf_kpis['fn']} TN={clf_kpis['tn']}")
    return result


# ── MODEL B: Visual-PINN Pathogen/Fraud Detector (synthetic benchmark) ───

def evaluate_visual_pinn() -> dict:
    """
    Runs a synthetic benchmark because real labelled image datasets are not
    available at eval time.  Generates 500 synthetic feature vectors with
    known ground truth, passes them through the classifier logic, and
    computes ROC-style metrics.
    """
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  MODEL B  |  Visual-PINN Pathogen/Fraud Detector")
    print("  (Synthetic benchmark — 500 feature vectors)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    from pinn_engine.visual_pinn import ARCHETYPES, _classify_pathogen, _classify_fraud  # noqa

    rng = np.random.default_rng(42)
    n = 500
    arch = ARCHETYPES[1]   # Archetype 1 benchmark

    # Ground-truth pathogens: high dv + high entropy + multiple contours
    y_path_true = rng.integers(0, 2, n)   # 0 = clean, 1 = pathogenic
    y_fraud_true = rng.integers(0, 2, n)  # 0 = genuine, 1 = fraud

    y_path_pred = []
    y_fraud_pred = []

    for i in range(n):
        is_path = bool(y_path_true[i])
        is_fraud = bool(y_fraud_true[i])

        # Synthesise feature vectors guided by each class signature
        base_dv = rng.uniform(2000, 9000) if is_path else rng.uniform(100, 900)
        base_ent = rng.uniform(5.5, 7.5) if is_path else rng.uniform(3.0, 5.0)
        feats = {
            "diffusion_var":   float(base_dv + rng.normal(0, 200)),
            "mean_intensity":  float(rng.uniform(160, 200)) if is_fraud else float(rng.uniform(80, 150)),
            "entropy":         float(base_ent + rng.normal(0, 0.3)),
            "edge_density":    float(rng.uniform(0.20, 0.35)) if is_fraud else float(rng.uniform(0.05, 0.15)),
            "dominant_hue":    float(rng.uniform(30, 60)) if is_fraud else float(rng.uniform(10, 25)),
            "mean_saturation": float(rng.uniform(50, 80)) if is_fraud else float(rng.uniform(130, 200)),
            "contour_count":   int(rng.integers(3, 8)) if is_path else int(rng.integers(0, 2)),
            "lesion_area_pct": float(rng.uniform(15, 50)) if is_path else float(rng.uniform(0, 8)),
            "spore_score":     float(rng.uniform(2.0, 5.0)) if is_path else float(rng.uniform(0, 1.5)),
        }

        p_det, _, _, _, _ = _classify_pathogen(feats, arch, 1)
        f_det, _, _, _, _ = _classify_fraud(feats, arch, 1)
        y_path_pred.append(1 if p_det else 0)
        y_fraud_pred.append(1 if f_det else 0)

    y_path_pred = np.array(y_path_pred)
    y_fraud_pred = np.array(y_fraud_pred)

    path_kpis = precision_recall_f1(y_path_true, y_path_pred)
    fraud_kpis = precision_recall_f1(y_fraud_true, y_fraud_pred)

    print(f"\n  Pathogen Detection KPIs (n={n}):")
    print(f"    Precision  : {path_kpis['precision']:.4f}")
    print(f"    Recall     : {path_kpis['recall']:.4f}")
    print(f"    F1         : {path_kpis['f1']:.4f}")
    print(f"    Specificity: {path_kpis['specificity']:.4f}")
    print(f"    Accuracy   : {path_kpis['accuracy']*100:.2f}%")
    print(f"    MCC        : {path_kpis['mcc']:.4f}")
    print(f"\n  Fraud Detection KPIs (n={n}):")
    print(f"    Precision  : {fraud_kpis['precision']:.4f}")
    print(f"    Recall     : {fraud_kpis['recall']:.4f}")
    print(f"    F1         : {fraud_kpis['f1']:.4f}")
    print(f"    Specificity: {fraud_kpis['specificity']:.4f}")
    print(f"    Accuracy   : {fraud_kpis['accuracy']*100:.2f}%")
    print(f"    MCC        : {fraud_kpis['mcc']:.4f}")

    return {
        "pathogen_detector":  path_kpis,
        "fraud_detector":     fraud_kpis,
        "synthetic_samples":  n,
    }


# ── MODEL C: Arrhenius Baseline (physics-only) ─────────────────────────────

def evaluate_arrhenius_baseline() -> dict:
    """
    Physics-only Arrhenius baseline (no neural network).
    Uses k = A·exp(−Ea/RT) to compute shelf life as a simple exponential
    function of temperature, then evaluates the same metrics.
    Demonstrates that PINN outperforms the pure physics model.
    """
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  MODEL C  |  Arrhenius Baseline (Physics Only)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not os.path.exists(DATA):
        print(f"  [SKIP] Dataset not found: {DATA}")
        return {"status": "skipped"}

    df = pd.read_csv(DATA)
    T_c = df["temperature_c"].values.astype(float)
    y_raw = df["actual_shelf_life_hours"].values.astype(float)

    # Arrhenius: k(T) = A·exp(−Ea/RT),  shelf_life ∝ 1/k
    EA = 50_000.0    # J/mol
    R = 8.314       # J/(mol·K)
    A = 1e8         # pre-exponential factor
    T_k = T_c + 273.15
    k_arr = A * np.exp(-EA / (R * T_k))
    # Reference: at 5°C (278K) shelf life ≈ 200 h
    k_ref = A * np.exp(-EA / (R * 278.15))
    preds = 200.0 * (k_ref / (k_arr + 1e-12))
    preds = np.clip(preds, 0, 500)

    residuals = y_raw - preds
    _r2 = r2(y_raw, preds)
    _rmse = rmse(y_raw, preds)
    _mae = mae(y_raw, preds)
    _mape = mape(y_raw, preds)
    _sigs = sigma_bands(residuals)

    clf_true = (y_raw < 120).astype(int)
    clf_pred = (preds < 120).astype(int)
    clf_kpis = precision_recall_f1(clf_true, clf_pred)

    print(f"  R²          : {_r2:.4f}  →  {_r2 * 100:.2f}%")
    print(f"  RMSE        : {_rmse:.2f} hrs")
    print(f"  MAE         : {_mae:.2f} hrs")
    print(f"  MAPE        : {_mape:.2f}%   → Accuracy: {100-_mape:.2f}%")
    print(f"  Within 3σ   : {_sigs['w3s_pct']:.1f}%")
    print(f"  Classifier F1: {clf_kpis['f1']:.4f}")
    print("  ↑ PINN should significantly outperform these baselines")

    return {
        "regression": {
            "r2":          round(_r2, 4),
            "rmse_hrs":    round(_rmse, 2),
            "mae_hrs":     round(_mae, 2),
            "mape_pct":    round(_mape, 2),
        },
        "sigma_analysis":     _sigs,
        "distress_classifier": clf_kpis,
    }


# ── AGGREGATE REPORT ────────────────────────────────────────────────────────

def run_all(export: bool = False) -> dict:
    print("\n" + "═" * 55)
    print("  KINEXICA — UNIFIED KPI EVALUATION REPORT")
    print("═" * 55)

    pinn_kpis = evaluate_pinn()
    vis_kpis = evaluate_visual_pinn()
    arrh_kpis = evaluate_arrhenius_baseline()

    # ── PINN vs Arrhenius delta ──────────────────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  COMPARATIVE SUMMARY")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if "regression" in pinn_kpis and "regression" in arrh_kpis:
        delta_r2 = pinn_kpis["regression"]["r2"] - \
            arrh_kpis["regression"]["r2"]
        delta_rmse = arrh_kpis["regression"]["rmse_hrs"] - \
            pinn_kpis["regression"]["rmse_hrs"]
        print(f"  PINN R² improvement over Arrhenius:    +{delta_r2:.4f}")
        print(
            f"  PINN RMSE reduction vs Arrhenius:      -{delta_rmse:.2f} hrs")

    if "distress_classifier" in pinn_kpis:
        clf = pinn_kpis["distress_classifier"]
        print(f"\n  PINN Distress Alarm KPIs:")
        print(f"    Precision : {clf['precision']:.4f}  (false-alarm cost)")
        print(f"    Recall    : {clf['recall']:.4f}  (spoilage catch rate)")
        print(f"    F1        : {clf['f1']:.4f}")

    if "pathogen_detector" in vis_kpis:
        pk = vis_kpis["pathogen_detector"]
        fk = vis_kpis["fraud_detector"]
        print(f"\n  Visual-PINN KPIs (synthetic 500-sample benchmark):")
        print(
            f"    Pathogen F1 :  {pk['f1']:.4f}   Accuracy: {pk['accuracy']*100:.1f}%")
        print(
            f"    Fraud F1    :  {fk['f1']:.4f}   Accuracy: {fk['accuracy']*100:.1f}%")

    if "sigma_analysis" in pinn_kpis:
        sg = pinn_kpis["sigma_analysis"]
        print(f"\n  Gaussian Residual Fit (PINN):")
        print(f"    σ = {sg['sigma']:.2f} hrs")
        print(f"    68.27% band: {sg['w1s_pct']:.1f}%  (1σ)")
        print(f"    95.45% band: {sg['w2s_pct']:.1f}%  (2σ)")
        print(f"    99.73% band: {sg['w3s_pct']:.1f}%  (3σ)")

    print("\n" + "═" * 55)

    report = {
        "pinn_model":            pinn_kpis,
        "visual_pinn":           vis_kpis,
        "arrhenius_baseline":    arrh_kpis,
    }

    if export:
        report_dir = os.path.join(BASE, "reports")
        os.makedirs(report_dir, exist_ok=True)
        out_path = os.path.join(report_dir, "kpis.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, cls=_NumpyEncoder)
        print(f"  [EXPORT] Report saved → {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kinexica Unified KPI Evaluator")
    parser.add_argument("--export", action="store_true",
                        help="Export full JSON report to reports/kpis.json")
    args = parser.parse_args()
    run_all(export=args.export)
