import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


DATA_DIR = "data/db1"
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
WINDOW_SIZE = 200
WINDOW_STEP = 100
MAX_WINDOWS_PER_FILE = 500
RANDOM_STATE = 42

os.makedirs(FIG_DIR, exist_ok=True)


def extract_window_features(emg_window):
    """Extract simple time-domain features for each EMG channel."""
    centered = emg_window - np.mean(emg_window, axis=0, keepdims=True)
    features = [
        np.mean(np.abs(emg_window), axis=0),
        np.sqrt(np.mean(emg_window ** 2, axis=0)),
        np.sum(np.abs(np.diff(emg_window, axis=0)), axis=0),
        np.sum(np.diff(np.signbit(centered), axis=0), axis=0),
    ]
    return np.concatenate(features)


def build_ml_dataset(
    files,
    window_size=WINDOW_SIZE,
    step=WINDOW_STEP,
    max_windows_per_file=MAX_WINDOWS_PER_FILE,
):
    """Build window-level features, labels, and subject groups from all files."""
    feature_rows, labels, groups = [], [], []

    for path in files:
        mat = loadmat(path)
        emg = np.asarray(mat["emg"], dtype=float)
        stimulus = mat.get("restimulus", mat.get("stimulus")).ravel()
        subject = os.path.basename(os.path.dirname(path))

        starts = np.arange(0, len(stimulus) - window_size + 1, step)
        if len(starts) > max_windows_per_file:
            selected = np.linspace(0, len(starts) - 1, max_windows_per_file, dtype=int)
            starts = starts[selected]

        for start in starts:
            end = start + window_size
            window_labels, counts = np.unique(stimulus[start:end], return_counts=True)
            label = window_labels[np.argmax(counts)]
            feature_rows.append(extract_window_features(emg[start:end]))
            labels.append(label)
            groups.append(subject)

    return np.asarray(feature_rows), np.asarray(labels), np.asarray(groups)


def run_ml_baselines(files, report_lines):
    """Train kNN, SVM, and random-forest baselines with subject-held-out data."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                     classification_report, confusion_matrix,
                                     ConfusionMatrixDisplay)
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        report_lines.extend([
            "\nML baselines skipped: scikit-learn is not installed.",
            "Install it with: python -m pip install scikit-learn",
        ])
        return

    report_lines.append("\nBuilding window-level ML dataset...")
    X, y, groups = build_ml_dataset(files)
    if len(X) == 0 or len(np.unique(y)) < 2:
        report_lines.append("ML baselines skipped: insufficient windows or classes.")
        return

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    report_lines.append(
        f"  Windows: {len(X)}  |  Features: {X.shape[1]}  |  "
        f"Train subjects: {len(np.unique(groups[train_idx]))}  |  "
        f"Test subjects: {len(np.unique(groups[test_idx]))}"
    )

    models = {
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "RF": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    labels = np.unique(y)
    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        predicted = model.predict(X[test_idx])
        report_lines.extend([
            f"\n{name} baseline",
            f"  Accuracy: {accuracy_score(y[test_idx], predicted):.4f}",
            f"  Balanced accuracy: {balanced_accuracy_score(y[test_idx], predicted):.4f}",
            classification_report(y[test_idx], predicted, zero_division=0),
        ])

        matrix = confusion_matrix(y[test_idx], predicted, labels=labels)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 7))
        display.plot(ax=ax, xticks_rotation="vertical", colorbar=False)
        ax.set_title(f"{name} baseline confusion matrix")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{name.lower()}.png"), dpi=150)
        plt.close(fig)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "S*", "*.mat")))
    report_lines = ["NinaPro DB1 - ML Baseline Report", "=" * 40]
    if not files:
        report_lines.append(f"\nNo .mat files found under '{DATA_DIR}'.")
    else:
        run_ml_baselines(files, report_lines)

    text = "\n".join(report_lines)
    print(text)
    out_path = os.path.join(OUTPUT_DIR, "ml_baselines_summary.txt")
    with open(out_path, "w") as output_file:
        output_file.write(text)
    print(f"\n[Report written to {out_path}]")


if __name__ == "__main__":
    main()