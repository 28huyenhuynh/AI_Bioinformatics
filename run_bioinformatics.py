"""STEP 4 - Muscle-channel (bioinformatics) analysis.

Outputs (results/bioinformatics/):
    channel_annotation.csv          channel -> electrode site / muscle
    activation_heatmap.png          gesture x channel mean activation (population)
    variability_heatmap.png         inter-subject coefficient of variation
    channel_importance.csv/.png     RF importance + drop-one-channel LOSO
    backward_elimination.csv/.png   accuracy vs number of channels kept
    atypicality_vs_loso.png         subject atypicality vs LOSO score (needs run_loso.py first)
    correlations.txt                Spearman correlations with p-values

Example
    python run_bioinformatics.py --loso-results results/loso_subset8 --loso-model RF
"""
import argparse
import os

import numpy as np
import pandas as pd

from semg import bioinfo, features, plots
from semg import config as C
from semg.data import build_dataset, list_subjects, resolve_gestures
from semg.protocols import encoder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gestures", default="subset8")
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--loso-results", default=os.path.join(C.RESULTS_DIR, "loso_subset8"))
    p.add_argument("--loso-model", default="RF", help="whose LOSO scores to correlate with atypicality")
    p.add_argument("--importance-model", default="LDA", help="fast model used for drop-channel / elimination")
    p.add_argument("--skip-elimination", action="store_true")
    p.add_argument("--folds", default="all",
                   help="test subjects used for drop-channel / elimination LOSO (e.g. 1,5,9 to run faster)")
    p.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "bioinformatics"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    subjects = list_subjects(args.data_dir)
    folds = subjects if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    gestures = resolve_gestures(args.gestures)
    moves = [g for g in gestures if g != 0]  # profiles of the movements themselves
    report = []

    pd.DataFrame({"channel": list(C.CHANNEL_ANNOTATION), "site": list(C.CHANNEL_ANNOTATION.values())}) \
        .to_csv(os.path.join(args.out, "channel_annotation.csv"), index=False)

    # 1-2. activation profiles and variability -------------------------------
    print("Computing activation tensor (subjects x gestures x channels) ...")
    A = bioinfo.activation_tensor(subjects, gestures, args.data_dir)
    np.save(os.path.join(args.out, "activation_tensor.npy"), A)
    names = [f"{g}: {C.gesture_name(g)}" for g in gestures]
    chan = [f"{c}\n{'FDS' if c == 'ch9' else 'EDS' if c == 'ch10' else 'ring'}" for c in C.CHANNEL_NAMES]
    mean_profile = np.nanmean(A, axis=0)
    plots.heatmap(mean_profile, names, chan, os.path.join(args.out, "activation_heatmap.png"),
                  "Mean activation (fraction of subject's 99th pct)", cbar_label="relative amplitude")
    cv = bioinfo.coefficient_of_variation(A)
    plots.heatmap(cv, names, chan, os.path.join(args.out, "variability_heatmap.png"),
                  "Inter-subject variability (coefficient of variation)", cmap="magma", cbar_label="CV")
    pd.DataFrame(mean_profile, index=names, columns=C.CHANNEL_NAMES).to_csv(os.path.join(args.out, "activation_profile.csv"))
    pd.DataFrame(cv, index=names, columns=C.CHANNEL_NAMES).to_csv(os.path.join(args.out, "variability_cv.csv"))

    # sanity check with anatomy: wrist flexion should load FDS (ch9) more than
    # EDS (ch10), wrist extension the opposite.
    if 25 in gestures and 26 in gestures:
        f, e = gestures.index(25), gestures.index(26)
        report.append("Anatomical sanity check (population mean):")
        report.append(f"  wrist flexion   ch9(FDS)={mean_profile[f, 8]:.3f}  ch10(EDS)={mean_profile[f, 9]:.3f}")
        report.append(f"  wrist extension ch9(FDS)={mean_profile[e, 8]:.3f}  ch10(EDS)={mean_profile[e, 9]:.3f}")

    # 3. channel importance --------------------------------------------------
    print("Building feature matrix ...")
    data = build_dataset(subjects, gestures, verbose=False, data_dir=args.data_dir)
    _, lut = encoder(gestures)
    F = features.extract(data["X"], "all")
    fch = features.feature_channel_index(features.feature_names("all"))
    y = lut[data["y"]]

    print("Random-forest channel importance ...")
    rf_imp = bioinfo.rf_channel_importance(F, y, fch)
    print(f"Drop-one-channel LOSO with {args.importance_model} ...")
    full, drops = bioinfo.drop_one_channel(F, y, data["subject"], fch, folds, args.importance_model)
    imp = pd.DataFrame({"channel": C.CHANNEL_NAMES, "site": list(C.CHANNEL_ANNOTATION.values()),
                        "rf_importance": rf_imp, f"loso_f1_drop_without_channel_{args.importance_model}": drops,
                        "mean_cv_across_gestures": np.nanmean(cv[[gestures.index(g) for g in moves]], axis=0)})
    imp.to_csv(os.path.join(args.out, "channel_importance.csv"), index=False)
    plots.heatmap(np.vstack([rf_imp / rf_imp.max(), drops / max(np.abs(drops).max(), 1e-9)]),
                  ["RF importance (scaled)", "F1 drop when removed (scaled)"], C.CHANNEL_NAMES,
                  os.path.join(args.out, "channel_importance.png"), "Channel importance", cmap="Blues")
    report.append(f"\nLOSO macro-F1 with all channels ({args.importance_model}): {full:.4f}")
    r, pv = bioinfo.correlate(imp["mean_cv_across_gestures"], drops)
    report.append(f"Spearman(channel variability CV, channel importance drop) = {r:.3f} (p={pv:.3f}, n=10 channels)")

    # 4. minimum channel subset ----------------------------------------------
    if not args.skip_elimination:
        print(f"Greedy backward elimination with {args.importance_model} (LOSO) ...")
        path = bioinfo.backward_elimination(F, y, data["subject"], fch, folds, args.importance_model)
        el = pd.DataFrame([{"n_channels": n, "loso_macro_f1": s, "channels": ",".join(f"ch{c + 1}" for c in k)}
                           for n, s, k in path])
        el.to_csv(os.path.join(args.out, "backward_elimination.csv"), index=False)
        plots.line_plot(el["n_channels"], {args.importance_model: el["loso_macro_f1"]},
                        os.path.join(args.out, "backward_elimination.png"), "Channels kept",
                        "LOSO macro-F1", "Minimum muscle-channel subset")
        best = el["loso_macro_f1"].max()
        smallest = el[el["loso_macro_f1"] >= 0.95 * best].sort_values("n_channels").iloc[0]
        report.append(f"Smallest subset within 95% of best F1: {smallest['n_channels']} channels "
                      f"({smallest['channels']}) F1={smallest['loso_macro_f1']:.4f}")

    # 5. atypicality vs generalisation ---------------------------------------
    aty = bioinfo.subject_atypicality(A[:, [gestures.index(g) for g in moves]])
    pd.DataFrame({"subject": subjects, "atypicality": aty}).to_csv(os.path.join(args.out, "subject_atypicality.csv"), index=False)
    loso_csv = os.path.join(args.loso_results, "per_subject.csv")
    if os.path.exists(loso_csv):
        lo = pd.read_csv(loso_csv)
        if "condition" in lo:
            lo = lo[lo.condition == "clean"]
        lo = lo[lo.model == args.loso_model].set_index("subject")["macro_f1"]
        ok = [s for s in subjects if s in lo.index]
        a = np.array([aty[subjects.index(s)] for s in ok])
        r, pv = bioinfo.correlate(a, lo.loc[ok].to_numpy())
        report.append(f"\nSpearman(subject atypicality, LOSO macro-F1 of {args.loso_model}) = {r:.3f} (p={pv:.4f}, n={len(ok)})")
        plots.scatter(a, lo.loc[ok].to_numpy(), [f"S{s}" for s in ok], os.path.join(args.out, "atypicality_vs_loso.png"),
                      "Subject atypicality (1 - r with population profile)", f"LOSO macro-F1 ({args.loso_model})",
                      f"Atypical subjects generalise worse?  Spearman r={r:.2f}, p={pv:.3f}")
    else:
        report.append(f"\n(no LOSO results at {loso_csv}; run run_loso.py first for the atypicality correlation)")

    text = "\n".join(report)
    print("\n" + text)
    with open(os.path.join(args.out, "correlations.txt"), "w") as fh:
        fh.write(text + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
