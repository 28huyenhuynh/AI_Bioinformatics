"""Simulated real-world signal degradation, applied to RAW windows
(before features / normalisation), because they model physical changes.

    noise     : additive Gaussian sensor noise at a given SNR (dB)
    fatigue   : amplitude change. Muscle fatigue mainly increases EMG
                amplitude (and lowers its frequency content, which DB1's 100 Hz
                envelope cannot show), so we model it as a gain > 1 with
                channel-to-channel spread.
    rotation  : electrode shift = rotating the 8-electrode forearm ring by k
                positions (ch1->ch2 ...). Channels 9-10 are left in place.
    dropout   : electrode failure = n channels set to zero.
"""
import numpy as np

from . import config as C


def add_noise(X, snr_db, rng):
    power = np.mean(X ** 2, axis=(0, 2), keepdims=True)  # per-channel signal power
    sigma = np.sqrt(power / (10 ** (snr_db / 10)))
    noisy = X + rng.normal(size=X.shape).astype(np.float32) * sigma.astype(np.float32)
    return np.clip(noisy, 0, None)  # DB1 values are an envelope: never negative


def fatigue(X, gain, rng, spread=0.1):
    g = gain * (1 + spread * rng.standard_normal(X.shape[1])).astype(np.float32)
    return X * np.clip(g, 0.05, None)[None, :, None]


def rotate_ring(X, k):
    Y = X.copy()
    ring = np.array(C.RING_CHANNELS)
    Y[:, ring] = np.roll(X[:, ring], shift=k, axis=1)
    return Y


def drop_channels(X, n, rng):
    Y = X.copy()
    if n > 0:
        Y[:, rng.choice(X.shape[1], size=n, replace=False)] = 0.0
    return Y


def apply(X, kind, level, seed=0):
    """Return a perturbed copy of X. level=0 (or None) means clean."""
    rng = np.random.default_rng(seed)
    is_clean = (kind == "clean" or level is None
                or (kind == "fatigue" and level == 1.0)
                or (kind in ("rotation", "dropout") and level == 0))
    # note: noise level 0 means SNR = 0 dB (noise as strong as signal), NOT clean
    if is_clean:
        return X
    if kind == "noise":
        return add_noise(X, level, rng).astype(np.float32)
    if kind == "fatigue":
        return fatigue(X, level, rng).astype(np.float32)
    if kind == "rotation":
        return rotate_ring(X, int(level))
    if kind == "dropout":
        return drop_channels(X, int(level), rng)
    raise ValueError(kind)


# Default sweep used by run_robustness.py. Level "0"/1.0 is the clean baseline.
DEFAULT_GRID = {
    "noise": [None, 30, 20, 10, 5, 0],      # SNR in dB (None = clean)
    "fatigue": [1.0, 1.25, 1.5, 2.0, 3.0],  # amplitude gain
    "rotation": [0, 1, 2, 3, 4],            # ring positions (4 = opposite side)
    "dropout": [0, 1, 2, 3, 4],             # channels lost
}


def augment_batch(X, rng, p_rot=0.5, p_noise=0.5, p_gain=0.5):
    """Training-time augmentation (optional) to make models shift/noise
    tolerant, following the idea of training on shifted electrode positions."""
    X = X.copy()
    n = len(X)
    rot = rng.random(n) < p_rot
    if rot.any():
        k = rng.choice([-1, 1], size=rot.sum())
        ring = np.array(C.RING_CHANNELS)
        sub = X[rot][:, ring]
        for shift in (-1, 1):
            m = k == shift
            sub[m] = np.roll(sub[m], shift, axis=1)
        tmp = X[rot]
        tmp[:, ring] = sub
        X[rot] = tmp
    gain = rng.random(n) < p_gain
    if gain.any():
        X[gain] *= rng.uniform(0.8, 1.5, size=(gain.sum(), X.shape[1], 1)).astype(np.float32)
    noise = rng.random(n) < p_noise
    if noise.any():
        snr = rng.uniform(10, 30, size=(noise.sum(), 1, 1))
        power = np.mean(X[noise] ** 2, axis=2, keepdims=True)
        X[noise] += (rng.standard_normal(X[noise].shape) * np.sqrt(power / 10 ** (snr / 10))).astype(np.float32)
    return X
