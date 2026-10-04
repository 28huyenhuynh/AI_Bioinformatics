"""Preprocess NinaPro DB1 EMG recordings for modeling."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


DATA_DIR = "data/db1"
OUTPUT_PATH = "outputs/db1_preprocessed.npz"
SAMPLING_RATE_HZ = 2_000
LOWCUT_HZ = 20.0
HIGHCUT_HZ = 450.0
WINDOW_DURATION_SEC = 0.2
WINDOW_OVERLAP = 0.5


def _metadata_vector(mat: dict, preferred: str, fallback: str) -> np.ndarray:
    """Return a flattened metadata vector from a MAT-file dictionary."""
    values = mat.get(preferred, mat.get(fallback))
    if values is None:
        raise KeyError(f"MAT file is missing both '{preferred}' and '{fallback}'.")
    return np.asarray(values).reshape(-1)


def _majority_value(values: np.ndarray) -> int:
    """Return the most common value in a window."""
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def bandpass_filter(
    emg: np.ndarray,
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
    lowcut_hz: float = LOWCUT_HZ,
    highcut_hz: float = HIGHCUT_HZ,
    filter_order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to samples x channels."""
    if emg.ndim != 2:
        raise ValueError(f"Expected a 2-D EMG array, got shape {emg.shape}.")
    nyquist_hz = sampling_rate_hz / 2
    if not 0 < lowcut_hz < highcut_hz < nyquist_hz:
        raise ValueError(
            f"Bandpass must satisfy 0 < lowcut < highcut < Nyquist ({nyquist_hz:g} Hz); "
            f"received {lowcut_hz:g}-{highcut_hz:g} Hz."
        )

    sos = butter(
        filter_order,
        [lowcut_hz, highcut_hz],
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    return sosfiltfilt(sos, emg, axis=0)


def zscore_channels(emg: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Z-score each channel using statistics from one complete recording."""
    means = np.mean(emg, axis=0, keepdims=True)
    standard_deviations = np.std(emg, axis=0, keepdims=True)
    standard_deviations = np.where(standard_deviations < epsilon, 1.0, standard_deviations)
    return (emg - means) / standard_deviations


def segment_recording(
    emg: np.ndarray,
    stimulus: np.ndarray,
    repetition: np.ndarray,
    window_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-size windows and majority-vote labels and repetitions."""
    if not (len(emg) == len(stimulus) == len(repetition)):
        raise ValueError("EMG, stimulus, and repetition must have the same sample count.")
    starts = range(0, len(emg) - window_size + 1, step)
    windows, labels, repetitions = [], [], []
    for start in starts:
        end = start + window_size
        windows.append(emg[start:end])
        labels.append(_majority_value(stimulus[start:end]))
        repetitions.append(_majority_value(repetition[start:end]))
    if not windows:
        return (
            np.empty((0, window_size, emg.shape[1]), dtype=np.float32),
            np.empty(0, dtype=stimulus.dtype),
            np.empty(0, dtype=repetition.dtype),
        )
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=stimulus.dtype),
        np.asarray(repetitions, dtype=repetition.dtype),
    )


def preprocess_file(
    path: str,
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
    lowcut_hz: float = LOWCUT_HZ,
    highcut_hz: float = HIGHCUT_HZ,
    window_duration_sec: float = WINDOW_DURATION_SEC,
    overlap: float = WINDOW_OVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess one MAT file and return windows, labels, and repetitions."""
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")
    window_size = round(window_duration_sec * sampling_rate_hz)
    step = round(window_size * (1 - overlap))
    if window_size < 1 or step < 1:
        raise ValueError("Window duration and overlap produce an invalid window step.")

    mat = loadmat(path)
    emg = np.asarray(mat["emg"], dtype=np.float64)
    stimulus = _metadata_vector(mat, "restimulus", "stimulus")
    repetition = _metadata_vector(mat, "rerepetition", "repetition")
    filtered = bandpass_filter(emg, sampling_rate_hz, lowcut_hz, highcut_hz)
    normalized = zscore_channels(filtered)
    return segment_recording(normalized, stimulus, repetition, window_size, step)


def _subject_id(path: str) -> str:
    """Derive the subject folder name from a MAT-file path."""
    return os.path.basename(os.path.dirname(path))


def build_dataset(
    files: Iterable[str],
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
    lowcut_hz: float = LOWCUT_HZ,
    highcut_hz: float = HIGHCUT_HZ,
    window_duration_sec: float = WINDOW_DURATION_SEC,
    overlap: float = WINDOW_OVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build X, y, subject_id, and repetition_id arrays from MAT files."""
    all_windows, all_labels, all_subjects, all_repetitions = [], [], [], []
    for path in sorted(files):
        windows, labels, repetitions = preprocess_file(
            path,
            sampling_rate_hz=sampling_rate_hz,
            lowcut_hz=lowcut_hz,
            highcut_hz=highcut_hz,
            window_duration_sec=window_duration_sec,
            overlap=overlap,
        )
        all_windows.append(windows)
        all_labels.append(labels)
        all_subjects.append(np.full(len(labels), _subject_id(path), dtype="U32"))
        all_repetitions.append(repetitions)

    window_size = round(window_duration_sec * sampling_rate_hz)
    n_channels = 0
    if all_windows:
        n_channels = all_windows[0].shape[2]
    X = (
        np.concatenate(all_windows, axis=0)
        if all_windows
        else np.empty((0, window_size, n_channels), dtype=np.float32)
    )
    return (
        X,
        np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.uint8),
        np.concatenate(all_subjects) if all_subjects else np.empty(0, dtype="U32"),
        np.concatenate(all_repetitions) if all_repetitions else np.empty(0, dtype=np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "S*", "*.mat")))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No MAT files found under '{args.data_dir}'.")

    X, y, subject_id, repetition_id = build_dataset(files)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        subject_id=subject_id,
        repetition_id=repetition_id,
    )
    print(f"Saved {len(X)} windows to {args.output}")
    print(f"X shape: {X.shape}; y shape: {y.shape}")


if __name__ == "__main__":
    main()
