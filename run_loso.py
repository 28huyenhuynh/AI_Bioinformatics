"""STEP 2 - Main study: leave-one-subject-out (LOSO) cross-subject evaluation.

For each of the 27 subjects: train on the other 26, test on the unseen one.
Normalisation and early stopping only ever see the 26 training subjects.

Examples
    python run_loso.py --models LDA,kNN3,kNN5,SVM,RF
    python run_loso.py --models CNN1D,LSTM,CNNLSTM --verbose
    python run_loso.py --models CNN1D --augment --out results/loso_augmented
"""
import os

from semg import cli, plots, protocols, stats
from semg.data import build_dataset


def main():
    p = cli.base_parser(__doc__)
    p.add_argument("--test-subjects", default="all", help="restrict which folds run (training still uses all others)")
    args = p.parse_args()
    subjects, models, gestures, cfg = cli.parse_common(args)
    test_subjects = subjects if args.test_subjects == "all" else [int(s) for s in args.test_subjects.split(",")]
    out = cli.out_dir(args, f"loso_{args.gestures if args.gestures in ('all', 'subset8') else 'custom'}")

    print(f"Loading windows for {len(subjects)} subjects ...")
    data = build_dataset(subjects, gestures, args.window, args.step, seed=args.seed, data_dir=args.data_dir)
    print(f"Total windows: {len(data['y'])}  | classes: {len(gestures)} | models {models}\n-> {out}")

    df, cms, classes = protocols.run_loso(data, test_subjects, models, gestures, cfg, out)

    summary = stats.summary_table(df)
    summary.to_csv(os.path.join(out, "summary.csv"))
    print("\nLOSO mean ± SD across test subjects\n", summary.to_string())
    plots.per_subject_bars(df, os.path.join(out, "per_subject_accuracy.png"), "accuracy",
                           "LOSO accuracy on each unseen subject")
    plots.per_subject_bars(df, os.path.join(out, "per_subject_f1.png"), "macro_f1",
                           "LOSO macro-F1 on each unseen subject")
    plots.model_boxplot(df, os.path.join(out, "model_boxplot.png"), "macro_f1", "LOSO macro-F1 across subjects")
    for m, cm in cms.items():
        plots.confusion_plot(cm, classes, os.path.join(out, f"confusion_{m}.png"), f"{m} - LOSO (all folds pooled)")
    print(f"\nSaved results and figures to {out}\nNext: python run_stats.py --results {out}")


if __name__ == "__main__":
    main()
