#!/usr/bin/env python
"""
Prototype: End-to-end AI vs Real receipt classifier.

Pipeline:
1. Input image → PaddleOCR text detection → text crops
2. Text crops → 14 Elective features
3. Features → SVM classifier → prediction + explanation
"""

import argparse
import json
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from paddleocr import TextDetection

# ─── Import feature extraction logic (copied from receipt-ai-feature-extractor-elective) ───

EDGE_GRADIENT_THRESHOLD = 40.0
DEFAULT_MIN_CROPS = 15

FEATURE_COLS = [
    'crop_count', 'crop_area_ratio_sum', 'crop_width_norm_avg', 'crop_height_norm_avg',
    'crop_aspect_ratio_avg', 'gray_mean_avg', 'gray_mean_std', 'gray_std_avg',
    'ink_fraction_avg', 'edge_density_avg', 'edge_density_std',
    'laplacian_variance_avg', 'local_variance_avg', 'entropy_avg'
]

# Class means from training data (900 AI + 900 Real) - for explanation
CLASS_MEANS = {
    0: {  # Real
        'crop_count': 55.2,
        'crop_area_ratio_sum': 0.265,
        'crop_width_norm_avg': 0.215,
        'crop_height_norm_avg': 0.022,
        'crop_aspect_ratio_avg': 5.15,
        'gray_mean_avg': 202.3,
        'gray_mean_std': 11.2,
        'gray_std_avg': 58.7,
        'ink_fraction_avg': 0.235,
        'edge_density_avg': 0.332,
        'edge_density_std': 0.062,
        'laplacian_variance_avg': 5842.1,
        'local_variance_avg': 1289.4,
        'entropy_avg': 5.82,
    },
    1: {  # AI
        'crop_count': 54.0,
        'crop_area_ratio_sum': 0.246,
        'crop_width_norm_avg': 0.198,
        'crop_height_norm_avg': 0.024,
        'crop_aspect_ratio_avg': 5.68,
        'gray_mean_avg': 198.7,
        'gray_mean_std': 9.8,
        'gray_std_avg': 64.2,
        'ink_fraction_avg': 0.218,
        'edge_density_avg': 0.371,
        'edge_density_std': 0.058,
        'laplacian_variance_avg': 7234.5,
        'local_variance_avg': 1862.3,
        'entropy_avg': 6.01,
    }
}


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    weighted = np.arange(256, dtype=np.float64) * hist
    cumulative_count = np.cumsum(hist)
    cumulative_weight = np.cumsum(weighted)
    denominator = cumulative_count * (total - cumulative_count)
    numerator = (cumulative_weight * total - cumulative_weight[-1] * cumulative_count) ** 2
    score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return int(np.argmax(score))


def pixel_features(image_path: Path) -> dict[str, float]:
    """Extract 7 pixel-level features from a single crop."""
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return {k: 0.0 for k in ["gray_mean", "gray_std", "ink_fraction", "edge_density",
                                  "laplacian_variance", "local_variance_mean", "entropy"]}

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    threshold = otsu_threshold(gray_u8)
    mask = (gray_u8 <= threshold) if gray_u8.std() >= 3.0 else np.zeros_like(gray_u8, dtype=bool)

    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.hypot(gx, gy)
    edge_density = float(np.mean(gradient > EDGE_GRADIENT_THRESHOLD))

    if gray.shape[0] > 2 and gray.shape[1] > 2:
        center = gray[1:-1, 1:-1]
        laplacian = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center)
        laplacian_variance = float(np.var(laplacian))
    else:
        laplacian_variance = 0.0

    padded = np.pad(gray, 1, mode="edge")
    neighborhoods = np.stack(
        [padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]] for dy in range(3) for dx in range(3)],
        axis=0,
    )
    local_variance_mean = float(np.var(neighborhoods, axis=0).mean())

    histogram = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / float(gray.size)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "ink_fraction": float(mask.mean()),
        "edge_density": edge_density,
        "laplacian_variance": laplacian_variance,
        "local_variance_mean": local_variance_mean,
        "entropy": entropy,
    }


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmax(diffs)]
    ordered[3] = points[np.argmin(diffs)]
    return ordered


def warp_polygon(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    rect = order_points(polygon)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(2, int(round(max(width_a, width_b))))
    height = max(2, int(round(max(height_a, height_b))))
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def tight_crop(warped: np.ndarray, padding_ratio: float = 0.12) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if gray.std() < 3:
        return warped, "uniform"
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return warped, "no_ink"
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    ink_ratio = float(np.mean(mask > 0))
    if ink_ratio > 0.65:
        return warped, "dense_fallback"
    pad = max(2, int(round(max(y1 - y0 + 1, 1) * padding_ratio)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(warped.shape[1] - 1, x1 + pad)
    y1 = min(warped.shape[0] - 1, y1 + pad)
    return warped[y0:y1 + 1, x0:x1 + 1], "otsu_ink_bounds"


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


# ─── Pipeline stages ───

def run_text_detection(image_path: Path, detector: TextDetection, output_dir: Path) -> list[dict]:
    """Run PaddleOCR text detection and save crops. Returns list of crop info dicts."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    result = detector.predict(str(image_path))[0]
    polygons = np.asarray(result["dt_polys"])
    scores = list(result["dt_scores"])

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_info = []

    for index, (polygon, score) in enumerate(zip(polygons, scores), start=1):
        polygon = np.asarray(polygon, dtype=np.float32)
        warped = warp_polygon(image, polygon)
        crop, trim_mode = tight_crop(warped)

        crop_id = f"crop_{index:03d}"
        crop_path = output_dir / f"{crop_id}.png"
        cv2.imwrite(str(crop_path), crop)

        # Get crop bounds for feature extraction
        x0, y0 = polygon[:, 0].min(), polygon[:, 1].min()
        x1, y1 = polygon[:, 0].max(), polygon[:, 1].max()
        width = int(x1 - x0 + 1)
        height = int(y1 - y0 + 1)

        crops_info.append({
            "crop_id": crop_id,
            "crop_path": crop_path,
            "score": float(score),
            "polygon": polygon.astype(int).tolist(),
            "source_x": int(x0),
            "source_y": int(y0),
            "width": width,
            "height": height,
            "trim_mode": trim_mode,
        })

    return crops_info


def extract_features(image_path: Path, crops_info: list[dict], source_width: int, source_height: int) -> dict:
    """Extract 14 receipt-level features from crops."""
    if not crops_info:
        return {col: None for col in FEATURE_COLS}

    width_values = []
    height_values = []
    aspect_values = []
    area_values = []
    gray_mean_values = []
    gray_std_values = []
    ink_values = []
    edge_values = []
    laplacian_values = []
    local_variance_values = []
    entropy_values = []

    for crop in crops_info:
        pixel = pixel_features(crop["crop_path"])
        width_norm = crop["width"] / source_width
        height_norm = crop["height"] / source_height
        aspect = crop["width"] / crop["height"]
        area_norm = (crop["width"] * crop["height"]) / (source_width * source_height)

        width_values.append(width_norm)
        height_values.append(height_norm)
        aspect_values.append(aspect)
        area_values.append(area_norm)
        gray_mean_values.append(pixel["gray_mean"])
        gray_std_values.append(pixel["gray_std"])
        ink_values.append(pixel["ink_fraction"])
        edge_values.append(pixel["edge_density"])
        laplacian_values.append(pixel["laplacian_variance"])
        local_variance_values.append(pixel["local_variance_mean"])
        entropy_values.append(pixel["entropy"])

    return {
        "crop_count": len(crops_info),
        "crop_area_ratio_sum": float(sum(area_values)) if area_values else None,
        "crop_width_norm_avg": mean(width_values),
        "crop_height_norm_avg": mean(height_values),
        "crop_aspect_ratio_avg": mean(aspect_values),
        "gray_mean_avg": mean(gray_mean_values),
        "gray_mean_std": std(gray_mean_values),
        "gray_std_avg": mean(gray_std_values),
        "ink_fraction_avg": mean(ink_values),
        "edge_density_avg": mean(edge_values),
        "edge_density_std": std(edge_values),
        "laplacian_variance_avg": mean(laplacian_values),
        "local_variance_avg": mean(local_variance_values),
        "entropy_avg": mean(entropy_values),
    }


def load_classifier(model_path: Path):
    """Load the trained classifier package."""
    package = joblib.load(model_path)
    return package


def predict(features: dict, package: dict) -> dict:
    """Run prediction and return result with explanation."""
    model = package['model']
    scaler = package['scaler']
    uses_scaled = package['uses_scaled']

    # Prepare feature vector in correct order
    X = np.array([[features[col] for col in FEATURE_COLS]])

    # Handle None values (replace with 0)
    X = np.nan_to_num(X, nan=0.0)

    # Scale if needed
    if uses_scaled and scaler is not None:
        X = scaler.transform(X)

    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None

    # Explanation: compare features to class means
    explanation = {}
    for col in FEATURE_COLS:
        val = features[col] if features[col] is not None else 0
        real_mean = CLASS_MEANS[0][col]
        ai_mean = CLASS_MEANS[1][col]
        
        # Distance to each class mean (normalized by typical range)
        diff_real = abs(val - real_mean)
        diff_ai = abs(val - ai_mean)
        
        # Which class is this feature closer to?
        closer_to = "AI" if diff_ai < diff_real else "Real"
        
        # Strength of signal (how much closer to one class vs the other)
        if diff_real + diff_ai > 0:
            signal_strength = abs(diff_real - diff_ai) / (diff_real + diff_ai)
        else:
            signal_strength = 0
        
        explanation[col] = {
            "value": val,
            "real_mean": real_mean,
            "ai_mean": ai_mean,
            "closer_to": closer_to,
            "signal_strength": signal_strength,
            "diff_from_real": diff_real,
            "diff_from_ai": diff_ai,
        }

    # Sort by signal strength (most discriminative first)
    sorted_expl = dict(sorted(explanation.items(), key=lambda x: x[1]['signal_strength'], reverse=True))

    return {
        "prediction": int(pred),
        "class_name": "ai_generated" if pred == 1 else "non_ai_generated",
        "probability_ai": float(prob) if prob is not None else None,
        "probability_real": float(1 - prob) if prob is not None else None,
        "features": features,
        "explanation": sorted_expl,
    }


def print_result(result: dict, image_path: Path):
    """Pretty print the prediction result with explanation."""
    pred = result['prediction']
    prob_ai = result['probability_ai']
    prob_real = result['probability_real']
    cls_name = result['class_name']

    print("\n" + "=" * 80)
    print(f"PREDICTION RESULT: {image_path.name}")
    print("=" * 80)
    print(f"  Class:      {cls_name.upper()} (label={pred})")
    print(f"  Confidence: AI={prob_ai:.1%} | Real={prob_real:.1%}" if prob_ai else "  Confidence: N/A")

    print(f"\n  Extracted Features ({len([v for v in result['features'].values() if v is not None])}/14):")
    for feat, val in result['features'].items():
        print(f"    {feat:30s} = {val:.4f}" if val is not None else f"    {feat:30s} = N/A")

    if result['explanation']:
        print("\n  WHY THIS PREDICTION? (feature comparison to training class means)")
        print("  " + "-" * 76)
        for i, (feat, info) in enumerate(list(result['explanation'].items())[:10], 1):
            val = info['value']
            closer = info['closer_to']
            strength = info['signal_strength']
            real_mean = info['real_mean']
            ai_mean = info['ai_mean']
            
            # Visual indicator (ASCII for Windows compatibility)
            arrow = "[AI]" if closer == "AI" else "[RL]"
            bar_len = int(strength * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            
            print(f"    {i:2d}. {arrow} {feat:28s} = {val:>8.4f}")
            print(f"         Real mean: {real_mean:>8.4f} | AI mean: {ai_mean:>8.4f} | Signal: {bar} {strength:.0%}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="End-to-end AI vs Real receipt classifier")
    parser.add_argument("image", type=Path, help="Path to receipt image")
    parser.add_argument("--model", type=Path,
                        default=Path(r"C:\Users\dev\Workspace\school\elective\programs\thesis-classifier\models\best_classifier.pkl"),
                        help="Path to trained model")
    parser.add_argument("--device", default="cpu", help="PaddleOCR device (cpu or gpu:0)")
    parser.add_argument("--min-crops", type=int, default=DEFAULT_MIN_CROPS, help="Minimum crops for quality")
    parser.add_argument("--keep-temps", action="store_true", help="Keep temporary crop files")
    args = parser.parse_args()

    image_path = args.image
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    # Load classifier
    print(f"Loading model from {args.model}...")
    package = load_classifier(args.model)
    print(f"  Model: {package['model_name']} (uses_scaled={package['uses_scaled']})")

    # Initialize PaddleOCR detector
    print(f"Initializing PaddleOCR detector (device={args.device})...")
    detector = TextDetection(
        model_name="PP-OCRv5_server_det",
        engine="paddle_static",
        device=args.device,
        enable_mkldnn=False,
    )

    # Create temp directory for crops
    temp_dir = Path(tempfile.mkdtemp(prefix="receipt_crops_"))
    print(f"Temp directory: {temp_dir}")

    try:
        # Get source image dimensions
        with Image.open(image_path) as img:
            source_width, source_height = img.size

        # Stage 1: Text detection
        print("\n[1/3] Running text detection...")
        crops_info = run_text_detection(image_path, detector, temp_dir)
        print(f"  Detected {len(crops_info)} text regions")

        if len(crops_info) < args.min_crops:
            print(f"  ⚠ Warning: Only {len(crops_info)} crops (minimum={args.min_crops})")

        # Stage 2: Feature extraction
        print("\n[2/3] Extracting features...")
        features = extract_features(image_path, crops_info, source_width, source_height)
        valid_features = sum(1 for v in features.values() if v is not None)
        print(f"  Extracted {valid_features}/14 features")

        # Stage 3: Classification
        print("\n[3/3] Running classification...")
        result = predict(features, package)

        # Output
        print_result(result, image_path)

        # Save JSON result
        output_json = image_path.with_suffix('.prediction.json')
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull result saved to: {output_json}")

    finally:
        detector.close()
        if not args.keep_temps:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\nCleaned up temp directory")
        else:
            print(f"\nTemp files kept at: {temp_dir}")


if __name__ == "__main__":
    main()