"""STEP 1 - Reproduction: within-subject classification on NinaPro DB1.

Trains one model per subject on repetitions 1,3,4,6,8,9,10 and tests on 2,5,7
(the protocol of Atzori et al. 2014 and most DB1 papers). Use this to check
that your pipeline reproduces published numbers BEFORE moving to LOSO.

Examples
    # classical reproduction on all 52 movements + rest
    python run_within_subject.py --gestures all --models LDA,kNN5,SVM,RF
    # CNN reproduction (needs PyTorch)
    python run_within_subject.py --gestures all --models CNN2D,CNN1D
"""
import os

from semg import cli, plots, protocols, stats


def main():
    p = cli.base_parser(__doc__)
    args = p.parse_args()
    subjects, models, gestures, cfg = cli.parse_common(args)
    out = cli.out_dir(args, f"within_subject_{args.gestures if args.gestures in ('all', 'subset8') else 'custom'}")
    print(f"Within-subject | {len(subjects)} subjects | {len(gestures)} classes | models {models}\n-> {out}")

    df, cms, classes = protocols.run_within_subject(subjects, models, gestures, cfg, args.window, args.step, out)

    summary = stats.summary_table(df)
    summary.to_csv(os.path.join(out, "summary.csv"))
    print("\nMean ± SD across subjects\n", summary.to_string())
    plots.per_subject_bars(df, os.path.join(out, "per_subject_accuracy.png"), "accuracy",
                           "Within-subject accuracy per subject")
    plots.model_boxplot(df, os.path.join(out, "model_boxplot.png"), "macro_f1")
    for m, cm in cms.items():
        plots.confusion_plot(cm, classes, os.path.join(out, f"confusion_{m}.png"), f"{m} - within-subject")
    print(f"\nSaved results and figures to {out}")


if __name__ == "__main__":
    main()
