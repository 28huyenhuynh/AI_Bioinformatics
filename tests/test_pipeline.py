"""Fast checks of the rules that silently ruin EMG results if broken.

Run:  python -m pytest tests -q      (or simply: python tests/test_pipeline.py)
Uses synthetic data, so no dataset is needed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semg import config as C  # noqa: E402
from semg import data, features, perturb, protocols, stats  # noqa: E402


def _fake_signal():
    # rest(30) g1 rep1 (40) rest(30) g1 rep2 (40)
    label = np.array([0] * 30 + [1] * 40 + [0] * 30 + [1] * 40, np.int16)
    rep = np.array([0] * 30 + [1] * 40 + [0] * 30 + [2] * 40, np.int16)
    emg = np.random.default_rng(0).random((len(label), C.N_CHANNELS)).astype(np.float32)
    return {"emg": emg, "label": label, "rep": data._fill_rest_repetitions(rep),
            "exercise": np.ones(len(label), np.int8)}


def test_rest_repetitions_filled():
    sig = _fake_signal()
    assert sig["rep"].min() >= 1
    assert np.all(sig["rep"][70:100] == 1)  # rest after rep 1 belongs to rep 1


def test_windows_never_mix_gestures():
    sig = _fake_signal()
    X, y, r = data.make_windows(sig, window=20, step=1)
    assert X.shape[1:] == (C.N_CHANNELS, 20)
    # verify purity directly
    starts = [i for i in range(len(sig["label"]) - 19)
              if len(set(sig["label"][i:i + 20])) == 1 and len(set(sig["rep"][i:i + 20])) == 1]
    assert len(starts) == len(y)


def test_global_label_offsets():
    assert C.EXERCISE_OFFSETS[2] == 12 and C.EXERCISE_OFFSETS[3] == 29
    assert 12 + 17 == 29 and 29 + 23 == 52


def test_features_shape_and_names():
    X = np.random.default_rng(1).random((5, C.N_CHANNELS, 20)).astype(np.float32)
    F = features.extract(X, "all")
    names = features.feature_names("all")
    assert F.shape == (5, len(names))
    ch = features.feature_channel_index(names)
    assert ch.min() == 0 and ch.max() == C.N_CHANNELS - 1


def test_ring_rotation_keeps_fds_eds():
    X = np.arange(C.N_CHANNELS, dtype=np.float32)[None, :, None].repeat(3, axis=2)
    Y = perturb.rotate_ring(X, 1)
    assert Y[0, 1, 0] == 0 and Y[0, 0, 0] == 7      # ring shifted
    assert Y[0, 8, 0] == 8 and Y[0, 9, 0] == 9      # ch9/ch10 untouched


def test_zero_db_noise_is_not_clean():
    X = np.ones((4, C.N_CHANNELS, 20), np.float32)
    assert not np.allclose(perturb.apply(X, "noise", 0, seed=1), X)
    assert np.allclose(perturb.apply(X, "noise", None), X)


def test_loso_val_subjects_never_include_test():
    d = {"subject": np.repeat(np.arange(1, 6), 10)}
    for s, tr, val in protocols.loso_folds(d, [1, 2, 3, 4, 5]):
        train_subjects = d["subject"][tr]
        assert s not in train_subjects
        assert set(train_subjects[val]).issubset(set(train_subjects))


def test_nearest_rest_split_half_and_half():
    # rest(4) rep1(3) rest(6) rep2(3) rest(30): the 6 middle rest samples split 3/3
    rep = np.array([0] * 4 + [1] * 3 + [0] * 6 + [2] * 3 + [0] * 30, np.int16)
    out = data._nearest_rest_repetitions(rep, max_gap=10)
    assert np.all(out[7:10] == 1) and np.all(out[10:13] == 2)
    assert np.all(out[16:26] == 2) and np.all(out[26:] == 0)  # beyond max_gap -> unassigned


def test_sota_param_counts_match_hartwell_tables():
    from semg import deep, sota
    if not deep.TORCH_OK:
        return
    rec = sota.RECIPES["hartwell"]
    assert sota.n_params(sota.make_net("TtS", 53, rec)) == 754_933          # Table I
    assert sota.n_params(sota.make_net("BaselineCNN", 53, rec)) == 776_341  # Table III
    # Jiang et al. 2024 report 0.89 M parameters for RIE
    assert round(sota.n_params(sota.make_net("RIE", 52, sota.RECIPES["jiang"])) / 1e6, 2) == 0.89


def test_forman_pooling_and_trial_vote():
    from semg import sota
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    rep = np.array([1, 1, 2, 1, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 1, 0, 1])
    m = sota.subject_metrics([sota.fold_result(y, rep, pred, 2)])
    assert np.isclose(m["micro"], 100 * 5 / 7) and np.isclose(m["macro"], 100 * (2 / 3 + 3 / 4) / 2)
    assert np.isclose(m["trial_acc"], 100 * 3 / 4)  # trial (0,2) voted 1 -> wrong


def test_holm():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
