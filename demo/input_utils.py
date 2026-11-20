"""Helpers for the live demo: turn a user-uploaded sensor CSV into model-ready windows.

No TensorFlow/Streamlit dependency, so it can be unit-tested on its own.
Channel order MUST match src/datasets/loader.py (SIGNAL_TYPES):
    0-2 body_acc (x,y,z)   3-5 body_gyro (x,y,z)   6-8 total_acc (x,y,z)
UCI HAR units: accelerometers in g, gyroscope in rad/s, 50 Hz, 128-sample windows.
"""
import io

import numpy as np
import pandas as pd

FS = 50          # sampling rate (Hz)
WINDOW = 128     # samples per window (2.56 s)
STRIDE = 64      # 50 % overlap, same as UCI HAR
G = 9.80665      # m/s^2 per g

CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _read_table(file_like):
    raw = file_like.read() if hasattr(file_like, "read") else open(file_like, "rb").read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("The file is empty.")
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None,
                     sep=r"[,;\s]+", engine="python", dtype=str)
    return df.dropna(axis=1, how="all")


def parse_recording(file_like, layout="uci9", acc_in_ms2=False):
    """Return a raw (T, 9) float32 array in loader channel order.

    layout="uci9":  9 columns already in UCI HAR layout (body_acc, body_gyro, total_acc).
                    If the header has the 9 channel names, columns are re-ordered by name.
    layout="phone6": 6 columns = total_acc xyz + gyro xyz (what a phone records).
                    body_acc is derived by removing gravity with a 0.3 Hz 3rd-order
                    Butterworth low-pass (the same filter UCI HAR used).
    acc_in_ms2: accelerometer values are in m/s^2 -> converted to g.
    """
    df = _read_table(file_like)

    names = None
    if pd.to_numeric(df.iloc[0], errors="coerce").isna().any():     # first row is a header
        names = [str(v).strip().lower() for v in df.iloc[0]]
        df = df.iloc[1:]
    data = df.apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any():
        raise ValueError("The file contains non-numeric or missing values.")
    arr = data.to_numpy(dtype=np.float64)

    if layout == "uci9":
        if arr.shape[1] != 9:
            raise ValueError(f"Expected 9 columns, found {arr.shape[1]}.")
        if names is not None and set(CHANNELS) <= set(names):
            arr = arr[:, [names.index(c) for c in CHANNELS]]
        if acc_in_ms2:
            arr[:, [0, 1, 2, 6, 7, 8]] /= G
        rec = arr
    elif layout == "phone6":
        if arr.shape[1] != 6:
            raise ValueError(f"Expected 6 columns (total_acc xyz + gyro xyz), found {arr.shape[1]}.")
        total, gyro = arr[:, :3].copy(), arr[:, 3:6]
        if acc_in_ms2:
            total /= G
        from scipy.signal import butter, filtfilt
        b, a = butter(3, 0.3 / (FS / 2), btype="low")
        gravity = filtfilt(b, a, total, axis=0)
        rec = np.hstack([total - gravity, gyro, total])
    else:
        raise ValueError(f"Unknown layout: {layout}")

    if len(rec) < WINDOW:
        raise ValueError(f"Need at least {WINDOW} rows (2.56 s at 50 Hz); got {len(rec)}.")
    return rec.astype(np.float32)


def make_windows(rec, stride=STRIDE):
    """(T, 9) -> (n, 128, 9) sliding windows."""
    n = (len(rec) - WINDOW) // stride + 1
    return np.stack([rec[i * stride: i * stride + WINDOW] for i in range(n)])


def normalize(windows, stats):
    """Apply the TRAIN-set z-score (stats from get_har_data) to raw windows."""
    return ((windows - stats["mean"]) / stats["std"]).astype(np.float32)


def denormalize(windows, stats):
    return windows * stats["std"] + stats["mean"]


def to_csv_bytes(rec):
    """Raw (T, 9) array -> CSV bytes with the 9 channel names as header."""
    rec = np.asarray(rec).reshape(-1, 9)   # tolerate (1, T, 9) inputs
    return pd.DataFrame(rec, columns=CHANNELS).to_csv(index=False).encode("utf-8")