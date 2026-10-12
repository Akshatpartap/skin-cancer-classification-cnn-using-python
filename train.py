"""
Training Pipeline for Skin Cancer Classification CNN.

Features:
- Handles missing dataset with educational instructions.
- Lesion-level patient split (prevents data leakage).
- Class weight compensation for severe class imbalance.
- Data augmentation pipeline on training split.
- EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks.
- Generates accuracy and loss visualization curves in reports/figures/.
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

from src.dataset import (
    CLASS_NAMES,
    LABEL_TO_IDX,
    load_and_prepare_metadata,
    split_dataset_by_lesion,
    compute_balanced_class_weights
)
from src.model import build_skin_cancer_cnn, compile_model
from src.utils import plot_training_curves, save_class_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN model for skin cancer classification on HAM10000 dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to directory containing HAM10000 images (default: data/raw)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/raw/HAM10000_metadata.csv",
        help="Path to HAM10000 metadata CSV (default: data/raw/HAM10000_metadata.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the trained model (default: models)"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save training curves and reports (default: reports)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum training epochs (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for training (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial Adam learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Target square image dimension in pixels (default: 128)"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on total samples (for fast CPU testing or dry runs)"
    )
    return parser.parse_args()


def create_tf_dataset(
    filepaths: np.ndarray,
    labels: np.ndarray,
    img_size: int,
    batch_size: int,
    is_training: bool = False
) -> tf.data.Dataset:
    """
    Creates a high-performance tf.data pipeline for image loading, resizing, and normalization.
    """
    def _parse_image(file_path, label):
        img_bytes = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), 1000), seed=42)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def print_dataset_notice(raw_dir: Path, metadata_path: Path):
    """Prints a clear, beginner-friendly notice if dataset is not found."""
    print("=" * 72)
    print("  [DATASET NOTICE] HAM10000 Dataset Not Found on Local Disk")
    print("=" * 72)
    print(f"Looked for:")
    print(f"  - Metadata file: {metadata_path}")
    print(f"  - Image directory: {raw_dir}")
    print("\nHow to obtain the dataset:")
    print("  1. Visit Kaggle (Recommended for one-click zip download):")
    print("     https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    print("     Or Harvard Dataverse (Official primary archive):")
    print("     https://doi.org/10.7910/DVN/DBW86T")
    print("  2. Place 'HAM10000_metadata.csv' inside 'data/raw/'.")
    print("  3. Extract image archives into 'data/raw/'.")
    print("  4. Once downloaded, re-run: python train.py --epochs 15")
    print("=" * 72)


def main():
    args = parse_args()

    raw_dir = Path(args.data_dir)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    processed_dir = Path("data/processed")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  Skin Cancer Classification Using CNN - Model Training Pipeline")
    print("=================================================================")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Target Image Size : {args.img_size}x{args.img_size}x3")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Max Epochs        : {args.epochs}")
    print("-----------------------------------------------------------------")

    # Step 1: Check dataset presence
    try:
        df = load_and_prepare_metadata(metadata_path, raw_dir)
    except FileNotFoundError as e:
        print_dataset_notice(raw_dir, metadata_path)
        print(f"\n[Detail] {e}")
        sys.exit(1)

    if args.sample_limit and len(df) > args.sample_limit:
        print(f"[INFO] Limiting dataset to {args.sample_limit} samples for fast testing.")
        df = df.sample(n=args.sample_limit, random_state=42).reset_index(drop=True)

    print(f"[INFO] Successfully loaded {len(df)} images matching metadata records.")

    # Step 2: Patient/Lesion-level split to prevent data leakage
    print("[INFO] Splitting dataset by lesion_id (70% train, 15% validation, 15% test)...")
    train_df, val_df, test_df = split_dataset_by_lesion(df)

    print(f"  - Training samples  : {len(train_df)}")
    print(f"  - Validation samples: {len(val_df)}")
    print(f"  - Test samples      : {len(test_df)}")

    # Save splits for full reproducibility
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    print(f"[INFO] Saved dataset split records to '{processed_dir}/'")

    # Step 3: Compute balanced class weights for severe class imbalance
    class_weights = compute_balanced_class_weights(train_df)
    print("[INFO] Computed balanced class weights:")
    for cls_idx, weight in class_weights.items():
        print(f"  - Class {cls_idx} ({CLASS_NAMES[cls_idx]}): {weight:.3f}")

    # Step 4: Build tf.data datasets
    train_ds = create_tf_dataset(
        train_df["filepath"].values,
        train_df["label_idx"].values,
        args.img_size,
        args.batch_size,
        is_training=True
    )
    val_ds = create_tf_dataset(
        val_df["filepath"].values,
        val_df["label_idx"].values,
        args.img_size,
        args.batch_size,
        is_training=False
    )

    # Step 5: Build and compile CNN model
    print("[INFO] Building Convolutional Neural Network architecture...")
    model = build_skin_cancer_cnn(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=len(CLASS_NAMES),
        include_augmentation=True
    )
    compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    # Step 6: Setup training callbacks
    model_save_path = output_dir / "skin_cancer_cnn.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Step 7: Train the model
    print(f"\n[INFO] Starting model training for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Step 8: Save class index mapping and training visual curves
    save_class_indices(LABEL_TO_IDX, output_dir / "class_indices.json")
    print(f"[INFO] Saved class indices mapping to '{output_dir / 'class_indices.json'}'")

    curves_path = reports_dir / "figures" / "accuracy_loss_curves.png"
    plot_training_curves(history, save_path=curves_path)

    print("\n" + "=" * 72)
    print("  TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 72)
    print(f"  Model saved to        : {model_save_path}")
    print(f"  Learning curves saved : {curves_path}")
    print("\nNext Recommended Steps:")
    print("  1. Evaluate model on held-out test set:")
    print("     python evaluate.py")
    print("  2. Test a single image prediction:")
    print("     python predict.py --image <path-to-image>")
    print("  3. Launch interactive Streamlit web dashboard:")
    print("     streamlit run app.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
