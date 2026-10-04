"""STEP 5 - Robustness: how fast does each model degrade under simulated
real-world problems? Runs LOSO; each fold's model is trained ONCE on clean
data, then tested on the unseen subject under every perturbation level.

Perturbations (see semg/perturb.py):
    noise    Gaussian sensor noise, SNR 30 -> 0 dB
    fatigue  amplitude gain 1.25x -> 3x (with channel-to-channel spread)
    rotation forearm ring rotated by 1-4 electrode positions (electrode shift)
    dropout  1-4 random channels lost (electrode failure)

Examples
    python run_robustness.py --models LDA,kNN5,SVM,RF
    python run_robustness.py --models CNN1D,LSTM
    python run_robustness.py --models RF,CNN1D --augment --out results/robustness_augmented
"""
import os

import numpy as np
import pandas as pd

from semg import cli, perturb, plots, protocols
from semg.data import build_dataset

CLEAN_LEVEL = {"noise": "", "fatigue": "1.0", "rotation": "0", "dropout": "0"}


def make_test_sets(grid, seed):
    sets = {"clean": lambda X: X}
    for kind, levels in grid.items():
        for i, lev in enumerate(levels):
            if str(lev) == CLEAN_LEVEL[kind] or lev is None:
                continue
            sets[f"{kind}:{lev}"] = (lambda X, k=kind, l=lev, s=seed + 1000 * i:
                                     perturb.apply(X, k, l, seed=s))
    return sets


def main():
    p = cli.base_parser(__doc__)
    p.add_argument("--test-subjects", default="all")
    p.add_argument("--kinds", default="noise,fatigue,rotation,dropout")
    args = p.parse_args()
    subjects, models, gestures, cfg = cli.parse_common(args)
    test_subjects = subjects if args.test_subjects == "all" else [int(s) for s in args.test_subjects.split(",")]
    out = cli.out_dir(args, "robustness")
    grid = {k: perturb.DEFAULT_GRID[k] for k in args.kinds.split(",")}

    data = build_dataset(subjects, gestures, args.window, args.step, seed=args.seed, data_dir=args.data_dir)
    df, _, _ = protocols.run_loso(data, test_subjects, models, gestures, cfg, out,
                                  test_sets=make_test_sets(grid, args.seed))

    # copy each clean result into every perturbation family as its level-0 point
    clean = df[df.condition == "clean"]
    curves = [df[df.condition != "clean"]]
    for kind in grid:
        c = clean.copy()
        c["condition"], c["level"] = kind, CLEAN_LEVEL[kind]
        curves.append(c)
    long = pd.concat(curves, ignore_index=True)
    long["level"] = long["level"].astype(str).replace("nan", "")
    long.to_csv(os.path.join(out, "robustness_long.csv"), index=False)

    table = long.groupby(["condition", "level", "model"])["macro_f1"].mean().unstack("model").round(3)
    table.to_csv(os.path.join(out, "robustness_table.csv"))
    print("\nMean LOSO macro-F1 by perturbation\n", table.to_string())

    # relative retention at the harshest level: score / clean score
    harsh = {k: str(v[-1]) for k, v in grid.items()}
    ret = []
    for kind, lev in harsh.items():
        for m in models:
            base = clean[clean.model == m].set_index("subject")["macro_f1"]
            hit = long[(long.condition == kind) & (long.level == lev) & (long.model == m)].set_index("subject")["macro_f1"]
            ret.append({"condition": kind, "level": lev, "model": m,
                        "retention_mean": float(np.mean(hit / base.reindex(hit.index)))})
    pd.DataFrame(ret).to_csv(os.path.join(out, "retention_harshest.csv"), index=False)

    plots.degradation_curves(long, os.path.join(out, "degradation_curves.png"))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
