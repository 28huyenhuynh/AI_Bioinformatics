"""Evaluation protocols shared by all scripts.

    within-subject : one model per subject, train on repetitions
                     1,3,4,6,8,9,10 and test on 2,5,7 (NinaPro standard;
                     used to REPRODUCE published DB1 numbers).
    LOSO           : leave-one-subject-out. Train on 26 subjects, test on the
                     unseen 27th, repeat 27 times (the project's main study).

Leakage rules enforced here:
  * feature scaling / z-scoring is fitted on the training fold only;
  * the deep-learning validation set (early stopping) comes from the training
    fold only: held-out repetition (within-subject) or held-out training
    subjects (LOSO). The test subject is never touched before testing.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from . import classical, deep, features
from . import config as C


DEFAULT_CFG = dict(feature_set="all", epochs=40, batch_size=256, lr=1e-3, log=True,
                   augment=False, seed=C.SEED, verbose=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, n_classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),  # = macro recall
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=np.arange(n_classes), zero_division=0),
    }


def confusion(y_true, y_pred, n_classes):
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


# ---------------------------------------------------------------------------
# A trained model, classical or deep, behind one interface
# ---------------------------------------------------------------------------
class FittedModel:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, {**DEFAULT_CFG, **(cfg or {})}
        self.is_deep = name in C.DEEP_MODELS

    def fit(self, X, y, n_classes, val_mask=None):
        cfg = self.cfg
        self.n_classes = n_classes
        if self.is_deep:
            if val_mask is None or val_mask.sum() == 0:
                rng = np.random.default_rng(cfg["seed"])
                val_mask = rng.random(len(y)) < 0.1
            self.scaler = deep.ChannelStandardizer(log=cfg["log"]).fit(X[~val_mask])
            Xtr, Xval = self.scaler.transform(X[~val_mask]), self.scaler.transform(X[val_mask])
            aug = None
            if cfg["augment"]:
                from .perturb import augment_batch
                mu, sd, log = self.scaler.mean_, self.scaler.std_, cfg["log"]
                eps = self.scaler.eps

                def aug(xb, rng):  # augment in raw space, then re-normalise
                    raw = xb * sd + mu
                    raw = np.exp(raw) - eps if log else raw
                    raw = np.clip(augment_batch(raw, rng), 0, None)
                    return self.scaler.transform(raw)
            self.net = deep.train_net(self.name, Xtr, y[~val_mask], Xval, y[val_mask], n_classes,
                                      epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                                      seed=cfg["seed"], augment=aug, verbose=cfg["verbose"])
        else:
            F = features.extract(X, cfg["feature_set"])
            if cfg["augment"]:
                from .perturb import augment_batch
                rng = np.random.default_rng(cfg["seed"])
                F = np.concatenate([F, features.extract(augment_batch(X, rng), cfg["feature_set"])])
                y = np.concatenate([y, y])
            self.model = classical.make_model(self.name, cfg["seed"], n_samples=len(y)).fit(F, y)
        return self

    def scores(self, X):
        if self.is_deep:
            return deep.predict_proba(self.net, self.scaler.transform(np.clip(X, 0, None)))
        return classical.predict_scores(self.model, features.extract(X, self.cfg["feature_set"]))

    def predict(self, X):
        if self.is_deep:
            return self.scores(X).argmax(1)
        return self.model.predict(features.extract(X, self.cfg["feature_set"]))


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
def encoder(gestures):
    classes = np.array(sorted(gestures))
    lut = np.full(classes.max() + 1, -1, dtype=np.int64)
    lut[classes] = np.arange(len(classes))
    return classes, lut


# ---------------------------------------------------------------------------
# Protocol runners
# ---------------------------------------------------------------------------
def run_within_subject(subjects, models, gestures, cfg, window=C.WINDOW, step=C.STEP, out_dir=None):
    from .data import subject_windows
    classes, lut = encoder(gestures)
    K = len(classes)
    rows, cms = [], {m: np.zeros((K, K), int) for m in models}
    for s in subjects:
        X, y, r = subject_windows(s, gestures, window, step, seed=cfg.get("seed", C.SEED))
        ye = lut[y]
        tr, te = np.isin(r, C.TRAIN_REPS), np.isin(r, C.TEST_REPS)
        for m in models:
            t0 = time.time()
            fm = FittedModel(m, cfg).fit(X[tr], ye[tr], K, val_mask=(r[tr] == C.TRAIN_REPS[-1]))
            pred = fm.predict(X[te])
            res = evaluate(ye[te], pred, K)
            cms[m] += confusion(ye[te], pred, K)
            rows.append({"protocol": "within_subject", "model": m, "subject": s,
                         "n_train": int(tr.sum()), "n_test": int(te.sum()), **res,
                         "seconds": round(time.time() - t0, 1)})
            print(f"  S{s:<2} {m:<8} acc={res['accuracy']:.3f}  bal_acc={res['balanced_accuracy']:.3f}"
                  f"  F1={res['macro_f1']:.3f}  ({rows[-1]['seconds']}s)", flush=True)
        if out_dir:
            _save(rows, cms, classes, out_dir)
    return pd.DataFrame(rows), cms, classes


def loso_folds(data, subjects, n_val_subjects=2, seed=C.SEED):
    """Yield (test_subject, train_mask, val_mask_within_train)."""
    rng = np.random.default_rng(seed)
    for s in subjects:
        train_mask = data["subject"] != s
        others = np.array([o for o in np.unique(data["subject"]) if o != s])
        val_subj = rng.choice(others, size=min(n_val_subjects, len(others) - 1), replace=False)
        val_mask = np.isin(data["subject"][train_mask], val_subj)
        yield s, train_mask, val_mask


def run_loso(data, subjects, models, gestures, cfg, out_dir=None, test_sets=None):
    """LOSO. `test_sets` maps a condition name to a function f(X_test)->X_test
    (used by the robustness study); default is just the clean test data."""
    classes, lut = encoder(gestures)
    K = len(classes)
    ye_all = lut[data["y"]]
    test_sets = test_sets or {"clean": lambda X: X}
    rows, cms = [], {m: np.zeros((K, K), int) for m in models}
    for s, tr, val in loso_folds(data, subjects, seed=cfg.get("seed", C.SEED)):
        te = data["subject"] == s
        Xtr, ytr = data["X"][tr], ye_all[tr]
        for m in models:
            t0 = time.time()
            fm = FittedModel(m, cfg).fit(Xtr, ytr, K, val_mask=val)
            fit_s = time.time() - t0
            for cond, f in test_sets.items():
                pred = fm.predict(f(data["X"][te]))
                res = evaluate(ye_all[te], pred, K)
                if cond == "clean":
                    cms[m] += confusion(ye_all[te], pred, K)
                kind, _, level = cond.partition(":")
                rows.append({"protocol": "loso", "model": m, "subject": s, "condition": kind,
                             "level": level or "", "n_train": int(tr.sum()), "n_test": int(te.sum()),
                             **res, "fit_seconds": round(fit_s, 1)})
            clean = [r for r in rows if r["model"] == m and r["subject"] == s][0]
            print(f"  test S{s:<2} {m:<8} acc={clean['accuracy']:.3f}  bal_acc={clean['balanced_accuracy']:.3f}"
                  f"  F1={clean['macro_f1']:.3f}  ({fit_s:.0f}s)", flush=True)
        if out_dir:
            _save(rows, cms, classes, out_dir)
    return pd.DataFrame(rows), cms, classes


def _save(rows, cms, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_subject.csv"), index=False)
    np.savez(os.path.join(out_dir, "confusion.npz"), classes=classes, **cms)
