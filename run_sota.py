"""Reproduce published deep models on NinaPro DB1 with each paper's own recipe.

    --paper hartwell   TtS CNN and Baseline CNN, Hartwell et al. (2020).
                       10 repetition splits, macro-average accuracy (Table VI).
    --paper hu         Attention-based hybrid CNN-RNN, Hu et al. (2018),
                       raw-image1 input, NinaPro split, window + trial accuracy (Table 4).

Each finished (model, subject, split) is saved under <out>/folds/, so an
interrupted run (e.g. a Colab disconnect) resumes where it stopped.
Settings the papers do not report are printed at start; see semg/sota.py.

Examples
    python run_sota.py --paper hartwell --subjects 1 --splits 1          # one fold, quick look
    python run_sota.py --paper hartwell                                  # full: 27 subjects x 10 splits (GPU)
    python run_sota.py --paper hu --subjects all
    python run_sota.py --paper hu --subjects 1 --max-train-windows 5000 --epochs 2   # CPU smoke test
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from semg import config as C
from semg import deep, sota
from semg.data import list_subjects


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", required=True, choices=list(sota.RECIPES))
    p.add_argument("--models", default=None, help="default: all models of the paper")
    p.add_argument("--subjects", default="all", help="'all' or comma list")
    p.add_argument("--splits", default="all", help="hartwell: 'all' or comma list of split numbers 1-10")
    p.add_argument("--epochs", type=int, default=None, help="override the paper's epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override the paper's batch size")
    p.add_argument("--lr", type=float, default=None, help="override the paper's learning rate")
    p.add_argument("--subsegments", type=int, default=None, help="hu: number of LSTM time steps")
    p.add_argument("--lowpass-hz", type=float, default=None, help="optional zero-phase Butterworth low-pass")
    p.add_argument("--max-train-windows", type=int, default=None,
                   help="random subsample of training windows (smoke tests only, not a reproduction)")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--out", default=None, help="default results/sota_<paper>")
    p.add_argument("--quiet", action="store_true", help="do not print epochs")
    return p.parse_args()


def build_recipe(args):
    r = dict(sota.RECIPES[args.paper])
    for key, val in (("epochs", args.epochs), ("batch_size", args.batch_size), ("lr", args.lr),
                     ("subsegments", args.subsegments)):
        if val is not None:
            r[key] = val
    if args.splits != "all":
        r["splits"] = [r["splits"][int(k) - 1] for k in args.splits.split(",")]
        r["split_ids"] = [int(k) for k in args.splits.split(",")]
    else:
        r["split_ids"] = list(range(1, len(r["splits"]) + 1))
    return r


def run_fold(model, s, k, X, y, rep, train_reps, test_reps, lut, K, recipe, args, fold_path):
    tr, te = np.isin(rep, train_reps), np.isin(rep, test_reps)
    tr_idx = np.where(tr)[0]
    if args.max_train_windows and len(tr_idx) > args.max_train_windows:
        tr_idx = np.sort(np.random.default_rng(args.seed).choice(tr_idx, args.max_train_windows, replace=False))
    scaler = deep.ChannelStandardizer(log=False).fit(X[tr_idx])
    t0 = time.time()
    net = sota.train(model, scaler.transform(X[tr_idx]), lut[y[tr_idx]], K, recipe,
                     seed=args.seed, verbose=not args.quiet)
    fit_s = time.time() - t0
    pred = deep.predict_proba(net, scaler.transform(X[te])).argmax(1)
    res = sota.fold_result(lut[y[te]], rep[te], pred, K)
    np.savez(fold_path, model=model, subject=s, split=k, n_train=len(tr_idx), n_test=int(te.sum()),
             fit_seconds=fit_s, **res)
    m = sota.subject_metrics([res])
    print(f"  S{s:<2} split {k:<2} {model:<11} macro {m['macro']:5.1f}  micro {m['micro']:5.1f}"
          f"  trial {m['trial_acc']:5.1f}  ({fit_s:.0f}s, {len(tr_idx)} train windows)", flush=True)


def aggregate(out, recipe, show=False):
    folds = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(out, "folds", "*.npz")))]
    if not folds:
        return
    rows = []
    for (model, s), grp in pd.DataFrame({"model": [str(f["model"]) for f in folds],
                                         "subject": [int(f["subject"]) for f in folds]}).groupby(["model", "subject"]):
        fs = [folds[i] for i in grp.index]
        rows.append({"model": model, "subject": s, "n_splits": len(fs), **sota.subject_metrics(fs)})
    df = pd.DataFrame(rows).sort_values(["model", "subject"])
    df.to_csv(os.path.join(out, "per_subject.csv"), index=False)

    lines = []
    for model, d in df.groupby("model"):
        targets = recipe["targets"].get(model, {})
        lines.append(f"{model}  ({len(d)} subjects, {d['n_splits'].min()}-{d['n_splits'].max()} splits each)")
        for metric in ("macro", "micro", "macro_no_rep1", "micro_no_rep1", "window_acc", "trial_acc"):
            if metric not in targets and metric not in ("macro", "micro"):
                continue
            ours = f"{d[metric].mean():5.1f} ({d[metric].std(ddof=1) if len(d) > 1 else 0:4.1f})"
            tgt = targets.get(metric)
            paper = "" if tgt is None else f"   paper {tgt[0]:.1f}" + (f" ({tgt[1]:.1f})" if tgt[1] else "")
            lines.append(f"  {metric:<14} ours {ours}{paper}")
    text = "Mean (SD) across subjects, %\n" + "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    if show:
        print("\n" + text)


def main():
    args = parse()
    if not deep.TORCH_OK:
        raise SystemExit("PyTorch is not installed: pip install torch")
    recipe = build_recipe(args)
    models = args.models.split(",") if args.models else recipe["models"]
    subjects = list_subjects(args.data_dir) if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]
    out = args.out or os.path.join(C.RESULTS_DIR, f"sota_{args.paper}")
    os.makedirs(os.path.join(out, "folds"), exist_ok=True)

    classes = np.array(recipe["gestures"])
    lut = np.full(classes.max() + 1, -1, np.int64)
    lut[classes] = np.arange(len(classes))
    K = len(classes)

    print(f"{args.paper}: models {models} | {len(subjects)} subjects | splits {recipe['split_ids']} | "
          f"{K} classes | window {recipe['window']} step {recipe['step']} | device {deep.get_device()}")
    for m in models:
        print(f"  {m}: {sota.n_params(sota.make_net(m, K, recipe)):,} parameters")
    print("Assumptions (not reported in the paper):")
    for a in sota.ASSUMPTIONS[args.paper]:
        print("  -", a)
    if args.max_train_windows:
        print(f"  ! training subsampled to {args.max_train_windows} windows: smoke test, NOT a reproduction")
    print(f"-> {out}\n", flush=True)

    for s in subjects:
        todo = [(m, k, sp) for m in models for k, sp in zip(recipe["split_ids"], recipe["splits"])
                if not os.path.exists(os.path.join(out, "folds", f"{m}_s{s}_k{k}.npz"))]
        if not todo:
            continue
        X, y, rep = sota.paper_windows(s, recipe, args.lowpass_hz, args.data_dir)
        for m, k, (train_reps, test_reps) in todo:
            run_fold(m, s, k, X, y, rep, train_reps, test_reps, lut, K, recipe, args,
                     os.path.join(out, "folds", f"{m}_s{s}_k{k}.npz"))
        aggregate(out, recipe)
    aggregate(out, recipe, show=True)


if __name__ == "__main__":
    main()
