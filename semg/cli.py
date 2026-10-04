"""Command-line options shared by the run_*.py scripts."""
import argparse
import os

from . import config as C
from .data import list_subjects, resolve_gestures


def base_parser(description):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", default="LDA,kNN5,SVM,RF",
                   help=f"comma list from {C.CLASSICAL_MODELS + C.DEEP_MODELS}")
    p.add_argument("--gestures", default="subset8", help="'subset8', 'all', or comma list of global labels")
    p.add_argument("--subjects", default="all", help="'all' or comma list, e.g. 1,2,3")
    p.add_argument("--window", type=int, default=C.WINDOW, help="window length in samples (100 Hz)")
    p.add_argument("--step", type=int, default=C.STEP, help="window hop in samples")
    p.add_argument("--feature-set", default="all", choices=["td", "mdwt", "all"])
    p.add_argument("--epochs", type=int, default=40, help="max epochs for deep models")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true", help="train with rotation/noise/gain augmentation")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--out", default=None, help="output folder (default under results/)")
    p.add_argument("--verbose", action="store_true", help="print deep-learning epochs")
    return p


def parse_common(args):
    subjects = list_subjects(args.data_dir) if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in C.CLASSICAL_MODELS + C.DEEP_MODELS:
            raise SystemExit(f"Unknown model '{m}'")
    if any(m in C.DEEP_MODELS for m in models):
        from .deep import TORCH_OK
        if not TORCH_OK:
            raise SystemExit("Deep models requested but PyTorch is not installed: pip install torch")
    cfg = dict(feature_set=args.feature_set, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
               augment=args.augment, seed=args.seed, verbose=args.verbose)
    return subjects, models, resolve_gestures(args.gestures), cfg


def out_dir(args, default_name):
    d = args.out or os.path.join(C.RESULTS_DIR, default_name)
    os.makedirs(d, exist_ok=True)
    return d
