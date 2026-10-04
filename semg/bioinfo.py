"""Muscle-channel ("bioinformatics") analysis.

The analogy with expression profiling: subjects are samples, channels are
"genes", gestures are conditions. We build a subject x gesture x channel
activation tensor and ask:

  1. Activation profiles  : which channels (muscles) each gesture recruits
                            (gesture x channel heatmap).
  2. Inter-subject variability : coefficient of variation of each
                            gesture/channel cell across subjects.
  3. Subject atypicality  : how far each subject's profile is from the rest of
                            the population; correlated with that subject's LOSO
                            score (do atypical subjects generalise worse?).
  4. Channel importance   : RF importance, drop-one-channel LOSO, and greedy
                            backward elimination (minimum channel subset).
"""
import numpy as np
from scipy.stats import spearmanr

from . import config as C
from .data import load_subject


def activation_tensor(subjects, gestures, data_dir=C.DATA_DIR, pct=99):
    """A[s, g, c] = mean envelope of channel c during gesture g for subject s,
    divided by that subject's 99th-percentile amplitude on channel c, so
    subjects with stronger signals / better skin contact become comparable."""
    A = np.full((len(subjects), len(gestures), C.N_CHANNELS), np.nan)
    for i, s in enumerate(subjects):
        sig = load_subject(s, data_dir)
        scale = np.percentile(sig["emg"], pct, axis=0) + 1e-9
        for j, g in enumerate(gestures):
            m = sig["label"] == g
            if m.any():
                A[i, j] = sig["emg"][m].mean(0) / scale
    return A


def coefficient_of_variation(A):
    """CV across subjects for every gesture x channel cell."""
    return np.nanstd(A, axis=0) / (np.nanmean(A, axis=0) + 1e-9)


def subject_atypicality(A):
    """1 - Pearson correlation between a subject's flattened profile and the
    mean profile of all OTHER subjects (leave-one-out). 0 = typical."""
    n = A.shape[0]
    flat = A.reshape(n, -1)
    out = np.empty(n)
    for i in range(n):
        ref = np.nanmean(np.delete(flat, i, axis=0), axis=0)
        ok = ~np.isnan(flat[i]) & ~np.isnan(ref)
        out[i] = 1 - np.corrcoef(flat[i, ok], ref[ok])[0, 1]
    return out


def correlate(x, y):
    r, p = spearmanr(x, y, nan_policy="omit")
    return float(r), float(p)


def rf_channel_importance(F, y, feat_channel, seed=C.SEED):
    """Sum of random-forest impurity importances over each channel's features."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(F, y)
    imp = np.zeros(C.N_CHANNELS)
    np.add.at(imp, feat_channel, rf.feature_importances_)
    return imp


def loso_score_with_channels(F, y, subjects_arr, feat_channel, channels, test_subjects, model="LDA"):
    """Mean LOSO macro-F1 using only the features of `channels`."""
    from sklearn.metrics import f1_score
    from .classical import make_model
    cols = np.isin(feat_channel, channels)
    scores = []
    for s in test_subjects:
        tr, te = subjects_arr != s, subjects_arr == s
        clf = make_model(model).fit(F[tr][:, cols], y[tr])
        scores.append(f1_score(y[te], clf.predict(F[te][:, cols]), average="macro", zero_division=0))
    return float(np.mean(scores))


def drop_one_channel(F, y, subjects_arr, feat_channel, test_subjects, model="LDA"):
    full = loso_score_with_channels(F, y, subjects_arr, feat_channel, list(range(C.N_CHANNELS)), test_subjects, model)
    drops = np.zeros(C.N_CHANNELS)
    for c in range(C.N_CHANNELS):
        keep = [k for k in range(C.N_CHANNELS) if k != c]
        drops[c] = full - loso_score_with_channels(F, y, subjects_arr, feat_channel, keep, test_subjects, model)
        print(f"    without ch{c + 1}: drop {drops[c]:+.4f}", flush=True)
    return full, drops


def backward_elimination(F, y, subjects_arr, feat_channel, test_subjects, model="LDA"):
    """Greedily remove the channel whose removal hurts LOSO macro-F1 the least.
    Returns list of (n_channels, score, channels_kept)."""
    kept = list(range(C.N_CHANNELS))
    path = [(len(kept), loso_score_with_channels(F, y, subjects_arr, feat_channel, kept, test_subjects, model), kept.copy())]
    print(f"    {len(kept)} channels: F1={path[-1][1]:.4f}", flush=True)
    while len(kept) > 1:
        trials = []
        for c in kept:
            cand = [k for k in kept if k != c]
            trials.append((loso_score_with_channels(F, y, subjects_arr, feat_channel, cand, test_subjects, model), c))
        best_score, removed = max(trials)
        kept.remove(removed)
        path.append((len(kept), best_score, kept.copy()))
        print(f"    {len(kept)} channels: F1={best_score:.4f}  (removed ch{removed + 1})", flush=True)
    return path
