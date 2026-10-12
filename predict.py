"""
Single Image Prediction Script for Skin Cancer Classification CNN.

Usage:
    python predict.py --image path/to/skin_lesion.jpg
"""

import sys
import argparse
from pathlib import Path

# Provide clear beginner-friendly notice if dependencies are not yet installed
try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print("=" * 72)
    print("  [DEPENDENCY NOTICE] Required Python Libraries Not Found")
    print("=" * 72)
    print(f"Missing package error: {e}")
    print("\nPlease activate your virtual environment and install dependencies:")
    print("  .\\venv\\Scripts\\Activate.ps1")
    print("  pip install -r requirements.txt")
    print("=" * 72)
    sys.exit(1)

from src.dataset import LESION_CLASSES, CLASS_NAMES, preprocess_single_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict skin lesion category for a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to skin lesion image (.jpg, .jpeg, .png)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/skin_cancer_cnn.keras",
        help="Path to trained model file (default: models/skin_cancer_cnn.keras)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Model input image dimension (default: 128)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)

    # Check model presence
    if not model_path.exists():
        print("=" * 72)
        print("  [PREDICTION NOTICE] Trained Model Not Found")
        print("=" * 72)
        print(f"Looked for: '{model_path}'")
        print("\nPlease train the model before running predictions:")
        print("  python train.py --epochs 15")
        print("=" * 72)
        sys.exit(1)

    # Check image presence
    if not image_path.exists():
        print(f"[ERROR] Specified image file does not exist: '{image_path}'")
        sys.exit(1)

    print("=================================================================")
    print("  Skin Cancer Classification Using CNN - Image Prediction")
    print("=================================================================")
    print(f"Image Path : {image_path}")
    print(f"Model Path : {model_path}")
    print("-----------------------------------------------------------------")

    # Load and preprocess image
    try:
        input_tensor = preprocess_single_image(image_path, target_size=(args.img_size, args.img_size))
    except Exception as e:
        print(f"[ERROR] Failed to load or process image: {e}")
        sys.exit(1)

    # Load model and run inference
    print("[INFO] Loading model and computing predictions...")
    model = tf.keras.models.load_model(str(model_path))
    predictions = model.predict(input_tensor, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_code = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100.0

    lesion_info = LESION_CLASSES.get(predicted_code, {})

    print("\n-----------------------------------------------------------------")
    print("  PREDICTION RESULT:")
    print("-----------------------------------------------------------------")
    print(f"  Predicted Category  : {lesion_info.get('full_name', predicted_code)} ({predicted_code})")
    print(f"  Classification Type : {lesion_info.get('category', 'Unknown')}")
    print(f"  Clinical Risk Level : {lesion_info.get('risk_level', 'Unknown')}")
    print(f"  Top Confidence Score: {confidence:.2f}%")
    print("-----------------------------------------------------------------")

    print("\nAll Class Confidence Probabilities (Ranked):")
    sorted_indices = np.argsort(predictions)[::-1]
    for rank, idx in enumerate(sorted_indices, start=1):
        code = CLASS_NAMES[idx]
        name = LESION_CLASSES[code]["full_name"]
        score = predictions[idx] * 100.0
        bar = "#" * int(score // 5)
        print(f"  {rank}. [{score:6.2f}%] {code.upper():<6} - {name} {bar}")

    print("\n" + "=" * 72)
    print("  IMPORTANT EDUCATIONAL & CLINICAL DISCLAIMER:")
    print("  1. Confidence scores represent mathematical pattern similarity to the")
    print("     training dataset, NOT the absolute biological probability of cancer.")
    print("  2. This application is an educational engineering demonstration.")
    print("     It is NOT a medical device and CANNOT provide a clinical diagnosis.")
    print("  3. Any suspicious, changing, or irregular skin lesion must be evaluated")
    print("     in person by a certified dermatologist or licensed physician.")
    print("=" * 72)


if __name__ == "__main__":
    main()
