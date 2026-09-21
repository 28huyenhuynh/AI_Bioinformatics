import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# CONFIG 
# ---------------------------------------------------------------------------
DATA_DIR = "data/db1"          # root folder 
SUBJECT_TO_INSPECT = "S1"      # which subject to use for detailed plots
EXERCISE_TO_INSPECT = "E1"     # which exercise file to use for detailed plots
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
SAMPLING_RATE_HZ = 100          # DB1 spec

os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. INVENTORY
# ---------------------------------------------------------------------------
def inventory_dataset(data_dir):
    """Scan data_dir and report which subjects/exercises are present."""
    pattern = os.path.join(data_dir, "S*", "*.mat")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[!] No .mat files found under '{data_dir}'.")
        print("    Check DATA_DIR, or see the download instructions at the top of this script.")
        return []

    print(f"Found {len(files)} .mat files under '{data_dir}':")
    for f in files:
        print(f"   - {f}")
    return files


# ---------------------------------------------------------------------------
# 2. LOADING
# ---------------------------------------------------------------------------
def load_subject_exercise(data_dir, subject, exercise):
    """Load a single subject/exercise .mat file into a dict of arrays."""
    matches = glob.glob(os.path.join(data_dir, subject, f"{subject}_*_{exercise}.mat"))
    if not matches:
        raise FileNotFoundError(
            f"No file found for subject={subject}, exercise={exercise} under {data_dir}. "
            f"Expected something like {subject}_A1_{exercise}.mat"
        )
    mat = loadmat(matches[0])
    return {
        "emg": mat["emg"],
        "stimulus": mat.get("restimulus", mat.get("stimulus")).ravel(),
        "repetition": mat.get("rerepetition", mat.get("repetition")).ravel(),
        "path": matches[0],
    }


# ---------------------------------------------------------------------------
# 3. SUMMARY STATS
# ---------------------------------------------------------------------------
def summarize_signal(d, report_lines):
    emg = d["emg"]
    n_samples, n_channels = emg.shape
    duration_sec = n_samples / SAMPLING_RATE_HZ

    report_lines.append(f"\nFile: {d['path']}")
    report_lines.append(f"  Samples: {n_samples}  |  Channels: {n_channels}")
    report_lines.append(f"  Duration: {duration_sec:.1f} sec (~{duration_sec/60:.1f} min) at {SAMPLING_RATE_HZ} Hz")

    nan_count = np.isnan(emg).sum()
    report_lines.append(f"  NaN values in EMG: {nan_count}")

    df = pd.DataFrame(emg, columns=[f"ch{i+1}" for i in range(n_channels)])
    stats = df.describe().T[["mean", "std", "min", "max"]]
    report_lines.append("  Per-channel stats:")
    report_lines.append(stats.to_string())

    labels, counts = np.unique(d["stimulus"], return_counts=True)
    report_lines.append(f"\n  Movement classes present: {len(labels)} (including rest=0)")
    class_df = pd.DataFrame({"label": labels, "n_samples": counts})
    report_lines.append(class_df.to_string(index=False))

    reps, rep_counts = np.unique(d["repetition"], return_counts=True)
    report_lines.append(f"\n  Repetitions present: {reps}")

    return df, class_df


# ---------------------------------------------------------------------------
# 4. PLOTS
# ---------------------------------------------------------------------------
def plot_class_distribution(class_df, out_path):
    plt.figure(figsize=(10, 4))
    plt.bar(class_df["label"].astype(str), class_df["n_samples"])
    plt.xlabel("Movement label (0 = rest)")
    plt.ylabel("Number of samples")
    plt.title("Class distribution — samples per movement")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_channel_boxplot(df, out_path):
    plt.figure(figsize=(10, 5))
    df.boxplot()
    plt.ylabel("EMG amplitude")
    plt.title("Per-channel amplitude distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sample_timeseries(d, out_path, window_sec=5):
    """Plot a short window of raw EMG for rest vs. the first active movement."""
    emg, stim = d["emg"], d["stimulus"]
    n_window = window_sec * SAMPLING_RATE_HZ

    rest_idx = np.where(stim == 0)[0]
    active_labels = [l for l in np.unique(stim) if l != 0]
    active_idx = np.where(stim == active_labels[0])[0] if active_labels else np.array([])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    if len(rest_idx) > n_window:
        seg = emg[rest_idx[0]: rest_idx[0] + n_window]
        axes[0].plot(seg)
        axes[0].set_title("Rest (label 0)")

    if len(active_idx) > n_window:
        seg = emg[active_idx[0]: active_idx[0] + n_window]
        axes[1].plot(seg)
        axes[1].set_title(f"Active movement (label {active_labels[0]})")

    axes[1].set_xlabel("Sample index")
    for ax in axes:
        ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    report_lines = ["NinaPro DB1 — EDA Summary Report", "=" * 40]

    files = inventory_dataset(DATA_DIR)
    if not files:
        report_lines.append(
            "\nNo data found. Download DB1 from https://ninapro.hevs.ch/ and set "
            "DATA_DIR to point at it, then re-run this script."
        )
        _write_report(report_lines)
        return

    try:
        d = load_subject_exercise(DATA_DIR, SUBJECT_TO_INSPECT, EXERCISE_TO_INSPECT)
    except FileNotFoundError as e:
        report_lines.append(f"\n[!] {e}")
        _write_report(report_lines)
        return

    df, class_df = summarize_signal(d, report_lines)

    plot_class_distribution(class_df, os.path.join(FIG_DIR, "class_distribution.png"))
    plot_channel_boxplot(df, os.path.join(FIG_DIR, "channel_boxplot.png"))
    plot_sample_timeseries(d, os.path.join(FIG_DIR, "sample_timeseries.png"))

    report_lines.append(f"\nFigures saved to: {FIG_DIR}/")
    _write_report(report_lines)


def _write_report(lines):
    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(OUTPUT_DIR, "eda_db1_summary.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n[Report written to {out_path}]")


if __name__ == "__main__":
    main()
