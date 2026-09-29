#!/usr/bin/env python
"""
Batch prediction script - process multiple receipt images at once.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def predict_single(image_path: Path, model_path: Path, device: str) -> dict:
    """Run predict_receipt.py on a single image and return parsed result."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "predict_receipt.py"),
        str(image_path),
        "--model", str(model_path),
        "--device", device,
    ]
    
    # Set PYTHONPATH for paddleocr
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = r"C:\Users\dev\Workspace\school\elective\programs\receipt-ai-text-cropper-paddleocr\python_deps"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    
    # Find the JSON output file
    json_path = image_path.with_suffix('.prediction.json')
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    else:
        return {"error": result.stderr, "image": str(image_path)}


def main():
    parser = argparse.ArgumentParser(description="Batch predict AI vs Real for multiple receipt images")
    parser.add_argument("input_dir", type=Path, help="Directory containing receipt images")
    parser.add_argument("--output", type=Path, default=Path("batch_results.json"), help="Output JSON file")
    parser.add_argument("--model", type=Path,
                        default=Path(r"C:\Users\dev\Workspace\school\elective\programs\thesis-classifier\models\best_classifier.pkl"))
    parser.add_argument("--device", default="cpu", help="PaddleOCR device (cpu or gpu:0)")
    parser.add_argument("--pattern", default="*.jpg,*.png", help="File patterns (comma-separated)")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    args = parser.parse_args()

    # Find images
    patterns = args.pattern.split(",")
    images = []
    for pattern in patterns:
        images.extend(args.input_dir.glob(pattern.strip()))
    images = sorted(images)
    
    if args.limit > 0:
        images = images[:args.limit]
    
    print(f"Found {len(images)} images to process")
    
    results = []
    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing {img.name}...")
        result = predict_single(img, args.model, args.device)
        result["image_file"] = img.name
        results.append(result)
        
        if "prediction" in result:
            cls = "AI" if result["prediction"] == 1 else "Real"
            conf = result.get("probability_ai", 0) if result["prediction"] == 1 else result.get("probability_real", 0)
            print(f"  -> {cls} ({conf:.1%})")
        else:
            print(f"  -> ERROR: {result.get('error', 'Unknown error')[:100]}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Summary
    ai_count = sum(1 for r in results if r.get("prediction") == 1)
    real_count = sum(1 for r in results if r.get("prediction") == 0)
    error_count = sum(1 for r in results if "error" in r)
    print(f"\nSUMMARY: {len(results)} total | AI: {ai_count} | Real: {real_count} | Errors: {error_count}")


if __name__ == "__main__":
    main()