"""Loading NinaPro DB1 and turning it into labelled windows.

Pipeline for one subject:
    3 .mat files (E1, E2, E3)
      -> concatenated signal with GLOBAL labels 0..52
      -> sliding windows that never cross a gesture/repetition boundary
      -> (optional) keep only a gesture subset, down-sample rest
"""
import glob
import os
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat

from . import config as C


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def list_subjects(data_dir=C.DATA_DIR):
    """Return sorted subject ids found as folders s1..s27 (any letter case)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    ids = []
    for name in os.listdir(data_dir):
        m = re.fullmatch(r"[sS](\d+)", name)
        if m and os.path.isdir(os.path.join(data_dir, name)):
            ids.append(int(m.group(1)))
    if not ids:
        raise FileNotFoundError(f"No subject folders (s1, s2, ...) inside {data_dir}")
    return sorted(ids)


def find_file(subject, exercise, data_dir=C.DATA_DIR):
    for folder in (f"s{subject}", f"S{subject}"):
        hits = glob.glob(os.path.join(data_dir, folder, f"*_E{exercise}.mat"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Missing file for subject {subject}, exercise E{exercise}")


def _fill_rest_repetitions(rep):
    """Rest samples have repetition 0. Give each rest sample the repetition of
    the movement just before it (forward fill), so a repetition-based
    train/test split also splits the rest periods."""
    rep = rep.astype(np.int16).copy()
    idx = np.where(rep > 0, np.arange(len(rep)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = rep[idx]
    # leading rest before the first movement -> first repetition seen
    first = np.argmax(rep > 0) if np.any(rep > 0) else 0
    filled[:first] = rep[first] if rep[first] > 0 else 1
    return filled


def _nearest_rest_repetitions(rep, max_gap=10 * C.FS):
    """Hartwell et al. (2020) rest labelling: each rest sample takes the
    repetition of the NEAREST movement sample (so a rest period is split half
    to the repetition before, half to the one after), up to `max_gap` samples
    away. Rest further than that from any movement keeps repetition 0."""
    rep = rep.astype(np.int16)
    n = len(rep)
    pos = np.arange(n)
    moving = rep > 0
    if not moving.any():
        return rep.copy()
    prev = np.where(moving, pos, -1)
    np.maximum.accumulate(prev, out=prev)
    nxt = np.where(moving, pos, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    far = np.iinfo(np.int64).max
    d_prev = np.where(prev >= 0, pos - prev, far)
    d_next = np.where(nxt < n, nxt - pos, far)
    src = np.where(d_prev <= d_next, prev, nxt)
    out = np.where(np.minimum(d_prev, d_next) <= max_gap, rep[np.clip(src, 0, n - 1)], 0).astype(np.int16)
    out[moving] = rep[moving]
    return out


REST_MODES = {"forward": _fill_rest_repetitions, "nearest": _nearest_rest_repetitions}


def load_subject(subject, data_dir=C.DATA_DIR, rest_mode="forward"):
    """Load E1-E3 of one subject with global labels.

    rest_mode: how rest samples (repetition 0 in the files) get a repetition,
    "forward" (default, see _fill_rest_repetitions) or "nearest" (Hartwell
    et al. 2020, see _nearest_rest_repetitions).

    Returns dict with
        emg      (N, 10) float32
        label    (N,)    int16   global label 0..52
        rep      (N,)    int16   repetition 1..10 (0 = unassigned rest, "nearest" only)
        exercise (N,)    int8
    """
    parts = {k: [] for k in ("emg", "label", "rep", "exercise")}
    for ex in (1, 2, 3):
        mat = loadmat(find_file(subject, ex, data_dir))
        emg = np.asarray(mat["emg"], dtype=np.float32)
        stim = np.asarray(mat["restimulus"]).ravel().astype(np.int16)
        rep = np.asarray(mat["rerepetition"]).ravel().astype(np.int16)
        if stim.max() > C.EXERCISE_SIZES[ex]:
            raise ValueError(f"S{subject} E{ex}: unexpected label {stim.max()}")
        label = np.where(stim > 0, stim + C.EXERCISE_OFFSETS[ex], 0).astype(np.int16)
        parts["emg"].append(emg)
        parts["label"].append(label)
        parts["rep"].append(REST_MODES[rest_mode](rep))
        parts["exercise"].append(np.full(len(stim), ex, dtype=np.int8))
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def make_windows(sig, window=C.WINDOW, step=C.STEP):
    """Cut a subject's signal into windows of shape (n, channels, window).

    A window is kept only if every sample in it has the same label, the same
    repetition and the same exercise, so no window mixes two gestures.
    """
    emg, label, rep, ex = sig["emg"], sig["label"], sig["rep"], sig["exercise"]
    n = len(label)
    if n < window:
        return np.empty((0, emg.shape[1], window), np.float32), np.empty(0, np.int16), np.empty(0, np.int16)

    key = label.astype(np.int64) * 10_000 + rep.astype(np.int64) * 10 + ex
    run = np.concatenate([[0], np.cumsum(key[1:] != key[:-1])])
    starts = np.arange(0, n - window + 1, step)
    pure = run[starts] == run[starts + window - 1]
    starts = starts[pure]

    views = sliding_window_view(emg, window, axis=0)  # (n-window+1, C, window)
    X = np.ascontiguousarray(views[starts], dtype=np.float32)
    return X, label[starts].copy(), rep[starts].copy()


def balance_rest(y, rng, ratio=C.REST_RATIO):
    """Indices that keep all movement windows and a random subset of rest
    windows (ratio x median movement-class count). Rest dominates DB1."""
    idx = np.arange(len(y))
    rest = idx[y == 0]
    move = idx[y != 0]
    if len(move) == 0 or ratio is None or ratio <= 0:
        return idx
    counts = np.bincount(y[move])
    target = int(ratio * np.median(counts[counts > 0]))
    if len(rest) > target:
        rest = rng.choice(rest, size=target, replace=False)
    return np.sort(np.concatenate([rest, move]))


def subject_windows(subject, gestures=C.ALL_GESTURES, window=C.WINDOW, step=C.STEP,
                    rest_ratio=C.REST_RATIO, seed=C.SEED, data_dir=C.DATA_DIR,
                    cache_dir=C.CACHE_DIR, use_cache=True):
    """Windows for one subject restricted to `gestures`, with caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"s{subject}_w{window}_s{step}.npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        X, y, r = z["X"], z["y"], z["rep"]
    else:
        X, y, r = make_windows(load_subject(subject, data_dir), window, step)
        if use_cache:
            np.savez(cache, X=X, y=y, rep=r)

    keep = np.isin(y, gestures)
    X, y, r = X[keep], y[keep], r[keep]
    rng = np.random.default_rng(seed + subject)
    idx = balance_rest(y, rng, rest_ratio) if 0 in gestures else np.arange(len(y))
    return X[idx], y[idx], r[idx]


def build_dataset(subjects, gestures=C.ALL_GESTURES, window=C.WINDOW, step=C.STEP,
                  rest_ratio=C.REST_RATIO, seed=C.SEED, data_dir=C.DATA_DIR, verbose=True):
    """Stack windows of several subjects.

    Returns dict with X (n, C, W), y (global labels), rep, subject.
    """
    Xs, ys, rs, ss = [], [], [], []
    for s in subjects:
        X, y, r = subject_windows(s, gestures, window, step, rest_ratio, seed, data_dir)
        Xs.append(X); ys.append(y); rs.append(r)
        ss.append(np.full(len(y), s, dtype=np.int16))
        if verbose:
            print(f"  subject {s:>2}: {len(y):>6} windows", flush=True)
    return {"X": np.concatenate(Xs), "y": np.concatenate(ys),
            "rep": np.concatenate(rs), "subject": np.concatenate(ss)}


def resolve_gestures(name_or_list):
    if isinstance(name_or_list, str):
        if name_or_list in C.GESTURE_SETS:
            return C.GESTURE_SETS[name_or_list]
        return [int(x) for x in name_or_list.split(",")]
    return list(name_or_list)
