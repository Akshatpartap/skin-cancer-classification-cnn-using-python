"""
Model Evaluation Pipeline for Skin Cancer Classification CNN.

Evaluates the trained model on the held-out test split (patient-level separated)
and produces:
1. Test loss and categorical accuracy.
2. Confusion matrix heatmap saved in reports/figures/confusion_matrix.png.
3. Per-class Precision, Recall, and F1-score saved in reports/classification_report.txt.
4. Summary metrics JSON saved in reports/metrics.json.
"""

import sys
import argparse
from pathlib import Path

# Provide clear beginner-friendly notice if dependencies are not yet installed
try:
    import numpy as np
    import pandas as pd
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

from src.dataset import CLASS_NAMES
from src.utils import (
    plot_confusion_matrix,
    generate_classification_report,
    save_metrics_json
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN on held-out test set."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/skin_cancer_cnn.keras",
        help="Path to trained .keras model file (default: models/skin_cancer_cnn.keras)"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/test.csv",
        help="Path to processed test split CSV (default: data/processed/test.csv)"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation reports and plots (default: reports)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for test evaluation (default: 32)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Input image size (default: 128)"
    )
    return parser.parse_args()


def load_test_dataset(test_csv_path: Path, img_size: int, batch_size: int):
    """Loads images and labels from the held-out test split."""
    if not test_csv_path.exists():
        raise FileNotFoundError(
            f"Test split CSV not found at '{test_csv_path}'.\n"
            f"Please run 'python train.py' first to prepare splits and train the model."
        )

    df = pd.read_csv(test_csv_path)

    def _parse_image(file_path, label):
        img_bytes = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label_idx"].values))
    dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return df, dataset


def main():
    args = parse_args()
    model_path = Path(args.model)
    test_csv_path = Path(args.test_csv)
    reports_dir = Path(args.reports_dir)

    print("=================================================================")
    print("  Skin Cancer Classification Using CNN - Evaluation Pipeline")
    print("=================================================================")

    # Step 1: Check model file existence
    if not model_path.exists():
        print("=" * 72)
        print("  [EVALUATION NOTICE] Trained Model Not Found")
        print("=" * 72)
        print(f"Looked for model file at: '{model_path}'")
        print("\nTo generate and save the model, please run training first:")
        print("  python train.py --epochs 15")
        print("=" * 72)
        sys.exit(1)

    # Step 2: Load test dataset
    try:
        test_df, test_ds = load_test_dataset(test_csv_path, args.img_size, args.batch_size)
    except FileNotFoundError as e:
        print("=" * 72)
        print("  [EVALUATION NOTICE] Test Split Not Found")
        print("=" * 72)
        print(e)
        sys.exit(1)

    print(f"[INFO] Evaluating model on {len(test_df)} held-out test images...")

    # Step 3: Load model
    print(f"[INFO] Loading trained model from '{model_path}'...")
    model = tf.keras.models.load_model(str(model_path))

    # Step 4: Run evaluation metrics
    eval_results = model.evaluate(test_ds, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, eval_results))

    print("\n-----------------------------------------------------------------")
    print("  ACTUAL TEST EVALUATION RESULTS:")
    print("-----------------------------------------------------------------")
    for metric_name, val in metrics_dict.items():
        print(f"  {metric_name.replace('_', ' ').capitalize():<20}: {val:.4f}")
    print("-----------------------------------------------------------------")

    # Step 5: Generate predictions for confusion matrix & classification report
    print("\n[INFO] Computing test set predictions...")
    y_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_df["label_idx"].values

    # Step 6: Plot confusion matrix
    cm_path = reports_dir / "figures" / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, save_path=cm_path)

    # Step 7: Classification report
    cr_path = reports_dir / "classification_report.txt"
    report_str = generate_classification_report(y_true, y_pred, CLASS_NAMES, save_path=cr_path)
    print("\nClassification Report:\n")
    print(report_str)

    # Step 8: Save metrics JSON
    metrics_json_path = reports_dir / "metrics.json"
    save_metrics_json(metrics_dict, metrics_json_path)

    print("=" * 72)
    print("  EVALUATION COMPLETE")
    print(f"  Metrics file : {metrics_json_path}")
    print(f"  Report file  : {cr_path}")
    print(f"  Matrix plot  : {cm_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
