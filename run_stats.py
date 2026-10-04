"""STEP 3 - Statistics: pairwise Wilcoxon signed-rank tests between models.

Reads per_subject.csv from one or more result folders (e.g. classical LOSO and
deep LOSO run separately) and compares every pair of models on the same
subjects, with Holm correction.

Example
    python run_stats.py --results results/loso_subset8 results/loso_deep --metric macro_f1
"""
import argparse
import os

import pandas as pd

from semg import stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True, help="folders containing per_subject.csv")
    p.add_argument("--metric", default="macro_f1", choices=["accuracy", "balanced_accuracy", "macro_f1"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    frames = []
    for d in args.results:
        df = pd.read_csv(os.path.join(d, "per_subject.csv"))
        if "condition" in df:
            df = df[df.condition == "clean"]
        frames.append(df)
    df = pd.concat(frames).drop_duplicates(subset=["model", "subject"], keep="last")

    out = args.out or args.results[0]
    summary = stats.summary_table(df)
    tests = stats.pairwise_wilcoxon(df, args.metric)
    summary.to_csv(os.path.join(out, "stats_summary.csv"))
    tests.to_csv(os.path.join(out, f"wilcoxon_{args.metric}.csv"), index=False)

    pd.set_option("display.width", 160)
    print("Mean ± SD across subjects\n", summary.to_string(), "\n")
    print(f"Pairwise Wilcoxon signed-rank on {args.metric} (Holm-corrected)\n",
          tests.round(4).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
