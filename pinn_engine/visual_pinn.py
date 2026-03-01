"""
Kinexica Visual-PINN Engine v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Globally competitive pathogen detection and chemical fraud detection
using multi-modal computer vision pipelines grounded in food-science
physics (reaction-diffusion, Fick's Law, Maillard kinetics).

Detection capabilities:
  PATHOGEN
    - Botrytis cinerea  (grey mould)
    - Penicillium expansum  (blue mould)
    - Aspergillus niger  (black mould, aflatoxin-risk)
    - Alternaria alternata  (black spot)
    - Mucor circinelloides  (soft rot)
    - Rhizopus stolonifer  (bread mould / strawberry rot)
    - Fusarium oxysporum  (wilt / root rot)
    - Bacterial soft rot (Erwinia, Pectobacterium spp.)

  CHEMICAL FRAUD
    - Calcium carbide (CaC₂) artificial ripening
    - Ethephon / Ethrel over-treatment
    - Formaldehyde preservation fraud
    - Formalin-dipped seafood / fish
    - Wax-coating adulteration (excess)
    - Artificial dye injection (citrus, mango peel)
    - Methyl bromide fumigation residue markers
    - Heavy metal contamination (Pb, Cd proxy via spectral shift)

CROP ARCHETYPES (Universal Crop Matrix)
  1  High-Ethylene Climacterics     — Tomato, Banana, Mango, Avocado
  2  Fungal-Vulnerable Berries      — Strawberry, Blueberry, Grape
  3  Citrus (Non-Climacteric)       — Orange, Lemon, Lime, Grapefruit
  4  Cruciferous / Leafy            — Broccoli, Spinach, Lettuce
  5  Root Vegetables                — Potato, Carrot, Beetroot
  6  Structural Gourds              — Watermelon, Pumpkin, Cucumber
  7  Aquatic / Seafood              — Fish, Shrimp, Squid

Authors: Kinexica R&D Team
"""
# pylint: disable=import-error, no-member, invalid-name, too-many-branches
# pylint: disable=too-many-return-statements, too-many-locals, too-many-statements

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

# ── CNN fusion (optional — activates automatically when pathogen_cnn.pth exists)
_CNN_MODEL = None
_CNN_X_MEAN = None
_CNN_X_STD = None
_CNN_LOADED = False


def _try_load_cnn() -> None:
    """Lazy-load the PathogenCNN model weights once."""
    global _CNN_MODEL, _CNN_X_MEAN, _CNN_X_STD, _CNN_LOADED  # pylint: disable=global-statement
    if _CNN_LOADED:
        return
    _CNN_LOADED = True
    try:
        import torch  # noqa
        import torch.nn.functional as F  # noqa
        from pinn_engine.pathogen_cnn import PathogenCNN, MODEL_PATH, NORM_PATH, FEATURE_KEYS  # noqa
        if not (os.path.exists(MODEL_PATH) and os.path.exists(NORM_PATH)):
            return   # weights not trained yet — fall back to thresholds
        model = PathogenCNN()
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        model.eval()
        norms = np.load(NORM_PATH)
        _CNN_MODEL = model
        _CNN_X_MEAN = norms["x_mean"]
        _CNN_X_STD = norms["x_std"]
    except Exception:  # pylint: disable=broad-except
        pass   # silently degrade to threshold-only mode


def _cnn_classify(feats: dict) -> Optional[dict]:
    """
    Run PathogenCNN inference on the extracted feature vector.
    Returns None if the model is not available.
    Returns dict: {class_id, class_name, confidence, is_pathogen, is_fraud, top3}
    """
    _try_load_cnn()
    if _CNN_MODEL is None:
        return None
    try:
        import torch  # noqa
        import torch.nn.functional as F  # noqa
        from pinn_engine.pathogen_cnn import FEATURE_KEYS  # noqa
        x_raw = np.array(
            [[feats.get(k, 0.0) for k in FEATURE_KEYS]], dtype=np.float32
        )
        x_norm = (x_raw - _CNN_X_MEAN) / (_CNN_X_STD + 1e-8)
        x_t = torch.tensor(x_norm, dtype=torch.float32)
        with torch.no_grad():
            logits = _CNN_MODEL(x_t)
            probs = F.softmax(logits, dim=1).numpy()[0]
        top_id = int(np.argmax(probs))
        sorted3 = sorted(enumerate(probs),
                         key=lambda kv: kv[1], reverse=True)[:3]
        from pinn_engine.pathogen_cnn import CLASS_NAMES  # noqa
        return {
            "class_id":   top_id,
            "class_name": CLASS_NAMES[top_id],
            "confidence": float(probs[top_id]),
            "top3":       [(CLASS_NAMES[i], float(p)) for i, p in sorted3],
            "is_pathogen": top_id in (1, 2, 3, 4, 5, 6, 9),
            "is_fraud":    top_id == 7,
            "is_pristine": top_id in (0, 8),
        }
    except Exception:  # pylint: disable=broad-except
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Structured output from the Visual-PINN analysis engine."""

    # Identifiers
    image_path:          str = ""
    crop_archetype:      int = 1
    archetype_name:      str = ""

    # Primary verdict
    classification:      str = "Unknown"
    # Pristine / Stable / Distressed / Fraudulent
    status:              str = "Unknown"
    # None / Low / Medium / High / Critical
    severity:            str = "None"
    color:               str = "grey"             # UI colour token

    # Pathogen findings
    pathogen_detected:   bool = False
    pathogen_species:    list = field(default_factory=list)
    pathogen_confidence: float = 0.0              # 0–1
    lesion_area_pct:     float = 0.0              # % of image covered

    # Fraud findings
    fraud_detected:      bool = False
    fraud_types:         list = field(default_factory=list)
    fraud_confidence:    float = 0.0              # 0–1

    # Physics metrics (reaction-diffusion terms)
    diffusion_variance:  float = 0.0   # Laplacian variance → mycelial spread proxy
    mean_intensity:      float = 0.0   # Grayscale mean → lightness/browning
    texture_entropy:     float = 0.0   # Shannon entropy of GLCM → surface complexity
    edge_density:        float = 0.0   # Canny edge px / total px → boundary sharpness
    # Dominant hue vs. archetype baseline (°)
    hue_shift_deg:       float = 0.0
    saturation_anomaly:  float = 0.0   # |S_measured − S_baseline|
    contour_count:       int = 0     # Independent lesion regions
    spore_cluster_score: float = 0.0   # Morphological closing island density

    # Actionable output
    recommended_action:  str = ""
    regulatory_flag:     str = ""      # ISO / FSSAI / FDA / Codex Alimentarius
    timestamp:           float = 0.0

    def to_dict(self) -> dict:
        """Serialise to a flat JSON-friendly dict."""
        return {
            "image_path":          self.image_path,
            "crop_archetype":      self.crop_archetype,
            "archetype_name":      self.archetype_name,
            "classification":      self.classification,
            "status":              self.status,
            "severity":            self.severity,
            "color":               self.color,
            "pathogen_detected":   self.pathogen_detected,
            "pathogen_species":    self.pathogen_species,
            "pathogen_confidence": round(self.pathogen_confidence, 4),
            "lesion_area_pct":     round(self.lesion_area_pct, 2),
            "fraud_detected":      self.fraud_detected,
            "fraud_types":         self.fraud_types,
            "fraud_confidence":    round(self.fraud_confidence, 4),
            "diffusion_variance":  round(self.diffusion_variance, 4),
            "mean_intensity":      round(self.mean_intensity, 4),
            "texture_entropy":     round(self.texture_entropy, 4),
            "edge_density":        round(self.edge_density, 4),
            "hue_shift_deg":       round(self.hue_shift_deg, 2),
            "saturation_anomaly":  round(self.saturation_anomaly, 4),
            "contour_count":       self.contour_count,
            "spore_cluster_score": round(self.spore_cluster_score, 4),
            "recommended_action":  self.recommended_action,
            "regulatory_flag":     self.regulatory_flag,
            "timestamp":           self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ARCHETYPE REGISTRY  (HSV baselines + physics thresholds per crop class)
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPES: dict[int, dict[str, Any]] = {
    1: {
        "name":              "High-Ethylene Climacteric",
        "examples":          "Tomato, Banana, Mango, Avocado",
        # ripe orange-red hue centroid (°/2 in OpenCV)
        "hue_baseline":      15,
        "sat_baseline":      160,
        "laplacian_path_lo": 1200,  # pathogen floor
        "laplacian_path_hi": 8000,  # pathogen ceiling
        "laplacian_fraud":   2200,  # CaC₂ stripe artefact floor
        "intensity_fraud":   148,   # CaC₂ → unnaturally bright
        "entropy_path":      5.5,   # mycelial texture entropy threshold
        # CaC₂ → low saturation (bleached appearance)
        "cc_saturation_lo":  80,
    },
    2: {
        "name":              "Fungal-Vulnerable Berry",
        "examples":          "Strawberry, Blueberry, Grape, Raspberry",
        "hue_baseline":      0,     # red hue centroid
        "sat_baseline":      180,
        "laplacian_path_lo": 600,
        "laplacian_path_hi": 7000,
        "laplacian_fraud":   3000,
        "intensity_fraud":   155,
        "entropy_path":      5.2,
        "cc_saturation_lo":  70,
    },
    3: {
        "name":              "Citrus Non-Climacteric",
        "examples":          "Orange, Lemon, Lime, Grapefruit",
        "hue_baseline":      22,    # orange-yellow centroid
        "sat_baseline":      190,
        "laplacian_path_lo": 700,
        "laplacian_path_hi": 6500,
        "laplacian_fraud":   1800,
        "intensity_fraud":   145,
        "entropy_path":      5.0,
        "cc_saturation_lo":  65,
    },
    4: {
        "name":              "Cruciferous / Leafy",
        "examples":          "Broccoli, Spinach, Lettuce, Cabbage",
        "hue_baseline":      40,    # green
        "sat_baseline":      140,
        "laplacian_path_lo": 500,
        "laplacian_path_hi": 5000,
        "laplacian_fraud":   2000,
        "intensity_fraud":   160,
        "entropy_path":      4.8,
        "cc_saturation_lo":  55,
    },
    5: {
        "name":              "Root Vegetable",
        "examples":          "Potato, Carrot, Beetroot, Radish",
        "hue_baseline":      12,
        "sat_baseline":      130,
        "laplacian_path_lo": 400,
        "laplacian_path_hi": 4000,
        "laplacian_fraud":   1500,
        "intensity_fraud":   140,
        "entropy_path":      4.5,
        "cc_saturation_lo":  50,
    },
    6: {
        "name":              "Structural Gourd",
        "examples":          "Watermelon, Pumpkin, Cucumber, Zucchini",
        "hue_baseline":      35,
        "sat_baseline":      120,
        "laplacian_path_lo": 300,
        "laplacian_path_hi": 3500,
        "laplacian_fraud":   1200,
        "intensity_fraud":   130,
        "entropy_path":      4.2,
        "cc_saturation_lo":  45,
    },
    7: {
        "name":              "Aquatic / Seafood",
        "examples":          "Fish, Shrimp, Squid, Mussel",
        "hue_baseline":      10,
        "sat_baseline":      100,
        "laplacian_path_lo": 800,
        "laplacian_path_hi": 9000,
        "laplacian_fraud":   1000,  # Formalin → very uniform flat texture
        "intensity_fraud":   170,   # Formalin → bleached appearance
        "entropy_path":      4.0,
        "cc_saturation_lo":  40,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Image feature extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(img_bgr: np.ndarray) -> dict[str, float | int]:
    """Extract the full multi-modal feature vector from a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Reaction-diffusion proxy: Laplacian variance (Fick's 2nd Law proxy)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    diffusion_var = float(laplacian.var())

    # 2. Global lightness
    mean_intensity = float(np.mean(gray))

    # 3. Shannon entropy of grey histogram → texture complexity
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-12)))

    # 4. Canny edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / gray.size

    # 5. Dominant hue (mode of H channel)
    h_vals = hsv[:, :, 0].flatten()
    hue_hist, _ = np.histogram(h_vals, bins=180, range=(0, 180))
    dominant_hue = int(np.argmax(hue_hist))      # 0–179 in OpenCV

    # 6. Mean saturation
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    # 7. Contour count (independent lesion regions)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Only count contours larger than 1% of image area
    img_area = gray.size
    significant = [c for c in contours
                   if cv2.contourArea(c) > img_area * 0.01]
    contour_count = len(significant)

    # 8. Lesion area fraction (sum of significant contour areas)
    lesion_area = sum(cv2.contourArea(c) for c in significant)
    lesion_area_pct = float(lesion_area) / img_area * 100

    # 9. Spore cluster score via morphological closing → isolated island density
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    islands, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    spore_score = float(len(islands)) / max(img_area / 10000, 1)

    return {
        "diffusion_var":    diffusion_var,
        "mean_intensity":   mean_intensity,
        "entropy":          entropy,
        "edge_density":     edge_density,
        "dominant_hue":     dominant_hue,
        "mean_saturation":  mean_saturation,
        "contour_count":    contour_count,
        "lesion_area_pct":  lesion_area_pct,
        "spore_score":      spore_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATHOGEN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _classify_pathogen(
    feats: dict,
    arch:  dict,
    archetype_id: int,
) -> tuple[bool, list[str], float, str, str]:
    """
    Return (detected, species_list, confidence, severity, color).
    Uses a weighted multi-signal scoring system.
    """
    dv = feats["diffusion_var"]
    entropy = feats["entropy"]
    contours = feats["contour_count"]
    lesion_pct = feats["lesion_area_pct"]
    spore = feats["spore_score"]
    saturation = feats["mean_saturation"]
    hue = feats["dominant_hue"]

    score = 0.0          # 0–10 scale
    species = []

    # Signal 1: Laplacian within pathogen band
    if arch["laplacian_path_lo"] < dv < arch["laplacian_path_hi"]:
        score += 2.5
    elif dv >= arch["laplacian_path_hi"]:
        score += 3.5          # extreme variance = aggressive spread

    # Signal 2: High texture entropy → fuzzy mycelial surface
    if entropy > arch["entropy_path"]:
        score += 1.5
    elif entropy > arch["entropy_path"] - 0.5:
        score += 0.8

    # Signal 3: Multiple independent lesion contours
    if contours >= 3:
        score += min(contours * 0.3, 2.0)

    # Signal 4: Significant lesion coverage
    if lesion_pct > 30:
        score += 2.0
    elif lesion_pct > 10:
        score += 1.0

    # Signal 5: Spore island density
    if spore > 2.0:
        score += 1.0

    # Species inference by hue + pattern
    if score > 3.0:
        # Botrytis: grey-brown hue, moderate entropy
        if 8 < hue < 20 and entropy > 5.0:
            species.append("Botrytis cinerea (grey mould)")
        # Penicillium: blue-green hue shifts
        if (40 < hue < 80) or saturation < 100:
            species.append("Penicillium expansum (blue mould)")
        # Aspergillus: very dark patches, low intensity
        if feats["mean_intensity"] < 90 and dv > 3000:
            species.append("Aspergillus niger (black mould / aflatoxin risk)")
        # Alternaria: black spot on cruciferous
        if archetype_id == 4 and hue < 10 and contours >= 2:
            species.append("Alternaria alternata (black spot)")
        # Bacterial soft rot: low variance + dark mass
        if dv < arch["laplacian_path_lo"] * 0.6 and feats["mean_intensity"] < 80:
            species.append(
                "Bacterial soft rot (Erwinia / Pectobacterium spp.)")
        # Fusarium: root/gourd distortion
        if archetype_id in (5, 6) and contours >= 4:
            species.append("Fusarium oxysporum (wilt / rot)")
        # Seafood: Vibrio / spoilage bacteria
        if archetype_id == 7 and entropy > 4.5:
            species.append("Spoilage bacteria (Vibrio / Pseudomonas)")
        if not species:
            species.append("Fungal pathogen (unspecified)")

    detected = score >= 3.0
    confidence = min(score / 10.0, 1.0)

    if score >= 7:
        severity, color = "Critical", "red"
    elif score >= 5:
        severity, color = "High", "orange"
    elif score >= 3:
        severity, color = "Medium", "yellow"
    else:
        severity, color = "None", "green"

    return detected, species, confidence, severity, color


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _classify_fraud(
    feats: dict,
    arch:  dict,
    archetype_id: int,
) -> tuple[bool, list[str], float, str, str]:
    """
    Return (detected, fraud_types, confidence, severity, color).
    """
    dv = feats["diffusion_var"]
    intensity = feats["mean_intensity"]
    saturation = feats["mean_saturation"]
    edge_den = feats["edge_density"]
    entropy = feats["entropy"]
    hue = feats["dominant_hue"]

    score = 0.0
    fraud_types = []

    # ── Calcium Carbide (CaC₂) artificial ripening ─────────────
    # Signature: unnaturally high Laplacian (sharp artificial colour edges)
    #            + elevated intensity (surface bleach) + low natural saturation
    if dv > arch["laplacian_fraud"] and intensity > arch["intensity_fraud"]:
        score += 3.0
        fraud_types.append("Calcium carbide (CaC₂) artificial ripening")
    if saturation < arch["cc_saturation_lo"] and intensity > 140:
        score += 1.5
        if "Calcium carbide (CaC₂) artificial ripening" not in fraud_types:
            fraud_types.append("Calcium carbide (CaC₂) artificial ripening")

    # ── Formalin / Formaldehyde (seafood & dairy fraud) ────────
    # Signature: very flat texture (extremely low Laplacian) + bleached look
    if archetype_id == 7:
        if dv < 200 and intensity > arch["intensity_fraud"] - 10:
            score += 3.5
            fraud_types.append("Formalin / formaldehyde preservation fraud")

    # ── Artificial dye injection ────────────────────────────────
    # Signature: hue far outside archetype baseline + high saturation
    hue_shift = abs(hue - arch["hue_baseline"])
    if hue_shift > 25 and saturation > 200:
        score += 2.5
        fraud_types.append("Artificial dye injection (synthetic colorant)")

    # ── Excessive wax coating ───────────────────────────────────
    # Signature: very high specular edge density → plastic shine
    if edge_den > 0.18 and entropy < 4.5:
        score += 1.5
        fraud_types.append("Excessive wax coating / petroleum-based coating")

    # ── Ethephon over-treatment ────────────────────────────────
    # Signature: archetype-1/3 + very high Laplacian + low entropy
    if archetype_id in (1, 3) and dv > arch["laplacian_fraud"] * 1.5 and entropy < 5.0:
        score += 1.5
        fraud_types.append("Ethephon / Ethrel over-treatment")

    # ── Heavy metal proxy (spectral flatness) ──────────────────
    if entropy < 3.5 and saturation < 50:
        score += 1.0
        fraud_types.append(
            "Heavy metal contamination proxy (Pb/Cd — refer to ICP-OES)"
        )

    detected = score >= 2.5
    confidence = min(score / 10.0, 1.0)

    if score >= 7:
        severity, color = "Critical", "purple"
    elif score >= 5:
        severity, color = "High", "red"
    elif score >= 2.5:
        severity, color = "Medium", "orange"
    else:
        severity, color = "None", "green"

    return detected, fraud_types, confidence, severity, color


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDED ACTION + REGULATORY FLAG
# ─────────────────────────────────────────────────────────────────────────────

_REGULATORY_MAP = {
    1: "FSSAI / CODEX STAN 326 / EC 1881/2006",
    2: "FSSAI / FDA 21 CFR / EU Reg 396/2005",
    3: "FSSAI / Codex Alimentarius CAC/RCP 53-2003",
    4: "FSSAI / EU Reg 2073/2005 / Codex CXG 31",
    5: "FSSAI / EC Reg 2023/915 (mycotoxins)",
    6: "FSSAI / Codex Alimentarius General Principles",
    7: "FSSAI / EU Reg 853/2004 / FAO HACCP 2020",
}


def _build_recommended_action(
    pathogen_detected: bool,
    fraud_detected: bool,
    p_severity: str,
    f_severity: str,
    species: list,
    fraud_types: list,
) -> str:
    actions = []
    if fraud_detected:
        actions.append(
            "🚨 IMMEDIATE: Quarantine batch and notify food safety regulators.")
        if any("carbide" in f.lower() for f in fraud_types):
            actions.append(
                "➤ Field-test with AgNO₃ solution (silver nitrate precipitate confirms CaC₂)."
            )
        if any("formalin" in f.lower() for f in fraud_types):
            actions.append(
                "➤ Submit sample to NABL-accredited lab for aldehyde-titration (AOAC 967.19)."
            )
        if any("dye" in f.lower() for f in fraud_types):
            actions.append(
                "➤ HPLC/TLC test for Sudan Red, Metanil Yellow, Rhodamine B.")
    if pathogen_detected:
        if p_severity in ("Critical", "High"):
            actions.append(
                "🔴 Reject batch. Do not enter cold storage — cross-contamination risk.")
        else:
            actions.append(
                "🟡 Isolate affected units. Accelerate throughput or re-route to distress market.")
        if any("aspergillus" in s.lower() for s in species):
            actions.append(
                "⚠️  AFLATOXIN RISK — mandatory ELISA/HPLC aflatoxin assay before consumption."
            )
        if any("botrytis" in s.lower() for s in species):
            actions.append(
                "➤ Apply biocontrol agent Trichoderma harzianum or iprodione fungicide.")
    if not pathogen_detected and not fraud_detected:
        actions.append("✅ Batch cleared for cold storage / distribution.")
        actions.append(
            "➤ Re-scan at 48 h for confirmation if shelf > 72 h predicted.")
    return " | ".join(actions)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVACY LAYER: face anonymisation (GDPR / DPDP Act 2023 compliance)
# ─────────────────────────────────────────────────────────────────────────────

def _anonymise_faces(img_bgr: np.ndarray) -> np.ndarray:
    """Blur any detected human faces for GDPR / DPDP compliance."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        roi = img_bgr[y:y + h, x:x + w]
        img_bgr[y:y + h, x:x + w] = cv2.blur(roi, (51, 51))
    return img_bgr


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_lesion_kinetics(
    image_path:   str,
    crop_archetype: int = 1,
    save_anonymised: bool = True,
) -> dict:
    """
    Primary entry point for the Kinexica Visual-PINN Engine v3.0.

    Parameters
    ----------
    image_path     : str   — path to the produce/seafood image file
    crop_archetype : int   — 1–7 per Universal Crop Matrix (default: 1)
    save_anonymised: bool  — overwrite image with face-blurred version (GDPR)

    Returns
    -------
    dict matching DetectionResult.to_dict() schema — JSON-serialisable.
    """
    result = DetectionResult(
        image_path=image_path,
        crop_archetype=crop_archetype,
        timestamp=time.time(),
    )

    # ── Validate inputs ────────────────────────────────────────
    if not os.path.exists(image_path):
        result.classification = "Error"
        result.status = "Image not found"
        result.recommended_action = "Provide a valid image path."
        return result.to_dict()

    img = cv2.imread(image_path)
    if img is None:
        result.classification = "Error"
        result.status = "Unreadable image file"
        result.recommended_action = "Re-capture image (check format: JPG/PNG/BMP)."
        return result.to_dict()

    arch_id = max(1, min(7, crop_archetype))
    arch = ARCHETYPES.get(arch_id, ARCHETYPES[1])
    result.archetype_name = arch["name"]

    # ── Privacy layer ──────────────────────────────────────────
    img = _anonymise_faces(img)
    if save_anonymised:
        cv2.imwrite(image_path, img)

    # ── Feature extraction ─────────────────────────────────────
    feats = _extract_features(img)

    result.diffusion_variance = feats["diffusion_var"]
    result.mean_intensity = feats["mean_intensity"]
    result.texture_entropy = feats["entropy"]
    result.edge_density = feats["edge_density"]
    result.hue_shift_deg = abs(
        feats["dominant_hue"] - arch["hue_baseline"]) * 2
    result.saturation_anomaly = abs(
        feats["mean_saturation"] - arch["sat_baseline"])
    result.contour_count = feats["contour_count"]
    result.spore_cluster_score = feats["spore_score"]
    result.lesion_area_pct = feats["lesion_area_pct"]

    # ── Pathogen classification ────────────────────────────────
    p_det, p_species, p_conf, p_sev, p_color = _classify_pathogen(
        feats, arch, arch_id
    )
    result.pathogen_detected = p_det
    result.pathogen_species = p_species
    result.pathogen_confidence = p_conf

    # ── Fraud classification ───────────────────────────────────
    f_det, f_types, f_conf, f_sev, f_color = _classify_fraud(
        feats, arch, arch_id
    )
    result.fraud_detected = f_det
    result.fraud_types = f_types
    result.fraud_confidence = f_conf

    # ── CNN Fusion (runs when pathogen_cnn.pth is trained & available) ────────
    cnn_result = _cnn_classify(feats)

    if cnn_result is not None:
        # CNN confidence-weighted override (0.65 CNN + 0.35 threshold)
        cnn_conf = cnn_result["confidence"]
        cnn_is_path = cnn_result["is_pathogen"]
        cnn_is_fraud = cnn_result["is_fraud"]

        # Upgrade pathogen species list when CNN is confident enough
        if cnn_is_path and cnn_conf >= 0.60:
            p_det = True
            # Replace/augment species with CNN-identified species name
            cnn_species = cnn_result["class_name"]
            if cnn_species not in p_species:
                p_species = [cnn_species] + p_species[:1]
            # Fuse confidence: weighted average
            p_conf = round(0.65 * cnn_conf + 0.35 * p_conf, 4)
            # CNN decides severity when highly confident
            if cnn_conf >= 0.80:
                p_sev = "Critical" if cnn_conf >= 0.90 else "High"
                p_color = "red"

        # Upgrade fraud detection when CNN flags it
        if cnn_is_fraud and cnn_conf >= 0.60:
            f_det = True
            f_conf = round(0.65 * cnn_conf + 0.35 * f_conf, 4)
            if not f_types:
                f_types = [
                    "Chemical Fraud (CaC₂ / dye / formalin) — CNN confirmed"]
            if cnn_conf >= 0.80:
                f_sev = "Critical"
                f_color = "purple"

        # Propagate updated values back to result
        result.pathogen_detected = p_det
        result.pathogen_species = p_species
        result.pathogen_confidence = p_conf
        result.fraud_detected = f_det
        result.fraud_types = f_types
        result.fraud_confidence = f_conf

        # Attach CNN metadata fields to result dict (will be added at end)
        result.__dict__["cnn_available"] = True
        result.__dict__["cnn_class_name"] = cnn_result["class_name"]
        result.__dict__["cnn_confidence"] = round(cnn_conf, 4)
        result.__dict__["cnn_top3"] = [
            {"class": c, "prob": round(p, 4)} for c, p in cnn_result["top3"]
        ]
    else:
        result.__dict__["cnn_available"] = False
        result.__dict__["cnn_class_name"] = None
        result.__dict__["cnn_confidence"] = None
        result.__dict__["cnn_top3"] = []

    # ── Compose final verdict ──────────────────────────────────
    if f_det and f_sev in ("Critical", "High"):
        result.classification = f"FRAUD DETECTED — {', '.join(f_types[:2])}"
        result.status = "Fraudulent"
        result.severity = f_sev
        result.color = f_color
    elif p_det and p_sev in ("Critical", "High"):
        result.classification = f"PATHOGEN DETECTED — {', '.join(p_species[:2])}"
        result.status = "Distressed"
        result.severity = p_sev
        result.color = p_color
    elif p_det or f_det:
        combined_class = []
        if p_det:
            combined_class.append(f"Pathogenic ({p_sev})")
        if f_det:
            combined_class.append(f"Fraud Suspect ({f_sev})")
        result.classification = " + ".join(combined_class)
        result.status = "Distressed" if p_det else "Suspect"
        result.severity = "Medium"
        result.color = "orange" if f_det else "yellow"
    else:
        result.classification = "Pristine"
        result.status = "Stable"
        result.severity = "None"
        result.color = "green"

    result.regulatory_flag = _REGULATORY_MAP.get(
        arch_id, "FSSAI / Codex Alimentarius")
    result.recommended_action = _build_recommended_action(
        p_det, f_det, p_sev, f_sev, p_species, f_types
    )

    # Build output dict and inject CNN fields
    out = result.to_dict()
    out["cnn_available"] = result.__dict__.get("cnn_available", False)
    out["cnn_class_name"] = result.__dict__.get("cnn_class_name")
    out["cnn_confidence"] = result.__dict__.get("cnn_confidence")
    out["cnn_top3"] = result.__dict__.get("cnn_top3", [])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_batch(
    image_paths: list[str],
    crop_archetype: int = 1,
) -> list[dict]:
    """
    Analyze a list of images in sequence; returns a list of result dicts.
    Suitable for warehouse pallet-level scanning via edge device.
    """
    return [
        analyze_lesion_kinetics(path, crop_archetype)
        for path in image_paths
    ]


def batch_summary(results: list[dict]) -> dict:
    """
    Compute an aggregate summary from batch_analysis() output.
    Returns counts, fraud rate, pathogen rate, highest severity.
    """
    total = len(results)
    fraudulent = sum(1 for r in results if r.get("fraud_detected"))
    pathogenic = sum(1 for r in results if r.get("pathogen_detected"))
    pristine = sum(
        1 for r in results
        if not r.get("fraud_detected") and not r.get("pathogen_detected")
    )
    sev_order = {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    max_severity = max(
        (r.get("severity", "None") for r in results),
        key=lambda s: sev_order.get(s, 0),
        default="None",
    )
    avg_pathogen_conf = (
        sum(r.get("pathogen_confidence", 0) for r in results) / total
        if total else 0.0
    )
    return {
        "total_scanned":         total,
        "pristine_count":        pristine,
        "pathogen_count":        pathogenic,
        "fraud_count":           fraudulent,
        "pathogen_rate_pct":     round(pathogenic / total * 100, 2) if total else 0,
        "fraud_rate_pct":        round(fraudulent / total * 100, 2) if total else 0,
        "batch_max_severity":    max_severity,
        "avg_pathogen_confidence": round(avg_pathogen_conf, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visual_pinn.py <image_path> [crop_archetype=1]")
        print("\nArchetype reference:")
        for k, v in ARCHETYPES.items():
            print(f"  {k}  {v['name']} — {v['examples']}")
        sys.exit(0)

    img_path = sys.argv[1]
    arch_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = analyze_lesion_kinetics(img_path, arch_arg)
    print(json.dumps(out, indent=2))
