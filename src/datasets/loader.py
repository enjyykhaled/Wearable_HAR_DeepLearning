import os
import numpy as np
import tensorflow as tf

# 9 Inertial Sensor Channels (3 axes X, Y, Z per sensor)
SIGNAL_TYPES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_"
]

# 6 Target Activity Classes (0-indexed)
ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

def _load_signals(dataset_path, subset):
    """Load and stack the 9 inertial signals into shape (Samples, 128, 9)."""
    signals_data = []
    for signal in SIGNAL_TYPES:
        filename = os.path.join(dataset_path, subset, "Inertial Signals", f"{signal}{subset}.txt")
        data = np.loadtxt(filename, dtype=np.float32)
        signals_data.append(data)
    return np.transpose(np.array(signals_data), (1, 2, 0))

def _load_labels(dataset_path, subset):
    """Load activity labels and convert from 1..6 to 0..5."""
    filename = os.path.join(dataset_path, subset, f"y_{subset}.txt")
    return np.loadtxt(filename, dtype=np.int32) - 1

def _load_subjects(dataset_path, subset):
    """Load subject IDs to enforce a leakage-free subject-wise split."""
    filename = os.path.join(dataset_path, subset, f"subject_{subset}.txt")
    return np.loadtxt(filename, dtype=np.int32)

def get_har_data(dataset_path="data/UCI HAR Dataset", batch_size=64, val_subjects=(1, 3, 5, 6)):
    """
    Load UCI HAR dataset, perform a leakage-free subject-wise split,
    apply training-only Z-score standardization, and return tf.data.Datasets + NumPy arrays.
    """
    # 1. Load raw train and held-out test sets
    X_train_full = _load_signals(dataset_path, "train")
    y_train_full = _load_labels(dataset_path, "train")
    subjects_train = _load_subjects(dataset_path, "train")

    X_test = _load_signals(dataset_path, "test")
    y_test = _load_labels(dataset_path, "test")

    # 2. Subject-wise partition (17 Train Subjects vs. 4 Validation Subjects)
    val_mask = np.isin(subjects_train, val_subjects)
    train_mask = ~val_mask

    X_train, y_train = X_train_full[train_mask], y_train_full[train_mask]
    X_val, y_val = X_train_full[val_mask], y_train_full[val_mask]

    # 3. Leakage-free Z-score standardization (fit strictly on X_train)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # 4. Build TensorFlow tf.data.Dataset pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    stats = {"mean": mean, "std": std}
    raw_arrays = (X_train, y_train, X_val, y_val, X_test, y_test)
    return train_ds, val_ds, test_ds, raw_arrays, stats

if __name__ == "__main__":
    train_ds, val_ds, test_ds, raw_arrays, _ = get_har_data()
    X_train, y_train, X_val, y_val, X_test, y_test = raw_arrays
    print("✅ TensorFlow Data Pipeline Ready!")
    print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Val   shape: {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test  shape: {X_test.shape}, Labels: {y_test.shape}")
    
