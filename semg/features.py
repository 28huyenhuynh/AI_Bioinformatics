"""Hand-crafted per-channel features for the classical models.

Input windows have shape (n, channels, window). Output is (n, channels * F),
ordered feature-major: [MAV_ch1..MAV_ch10, RMS_ch1..., ...]. `feature_names`
returns matching names so the bioinformatics step can map importances back to
channels.

Feature sets
    td    : MAV, RMS, WL, SD                 (classic time-domain set)
    mdwt  : marginal Haar DWT, 3 levels      (Atzori et al. 2014 used mDWT)
    all   : td + mdwt
Note: zero crossings / slope sign changes are meaningless on DB1 because the
signal is already a rectified RMS envelope, so they are not included.
"""
import numpy as np

from . import config as C

TD = ["MAV", "RMS", "WL", "SD"]


def _td(X):
    return {
        "MAV": np.mean(np.abs(X), axis=-1),
        "RMS": np.sqrt(np.mean(X ** 2, axis=-1)),
        "WL": np.sum(np.abs(np.diff(X, axis=-1)), axis=-1),
        "SD": np.std(X, axis=-1),
    }


def _mdwt(X, levels=3):
    """Marginal discrete wavelet transform with a Haar wavelet: sum of
    absolute detail coefficients at each level + the final approximation."""
    out = {}
    a = X.astype(np.float64)
    for lev in range(1, levels + 1):
        if a.shape[-1] < 2:
            break
        if a.shape[-1] % 2:
            a = np.concatenate([a, a[..., -1:]], axis=-1)
        even, odd = a[..., 0::2], a[..., 1::2]
        out[f"mDWT_d{lev}"] = np.sum(np.abs(even - odd) / np.sqrt(2), axis=-1)
        a = (even + odd) / np.sqrt(2)
    out["mDWT_a"] = np.sum(np.abs(a), axis=-1)
    return out


def extract(X, feature_set="all"):
    feats = {}
    if feature_set in ("td", "all"):
        feats.update(_td(X))
    if feature_set in ("mdwt", "all"):
        feats.update(_mdwt(X))
    if not feats:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return np.concatenate([feats[k] for k in feats], axis=1).astype(np.float32)


def feature_names(feature_set="all", n_channels=C.N_CHANNELS, window=C.WINDOW):
    dummy = np.zeros((1, n_channels, window), np.float32)
    names = []
    blocks = {}
    if feature_set in ("td", "all"):
        blocks.update(_td(dummy))
    if feature_set in ("mdwt", "all"):
        blocks.update(_mdwt(dummy))
    for k in blocks:
        names += [f"{k}_ch{c + 1}" for c in range(n_channels)]
    return names


def feature_channel_index(names):
    """Channel index (0-based) for each feature name like 'RMS_ch3'."""
    return np.array([int(n.rsplit("_ch", 1)[1]) - 1 for n in names])
