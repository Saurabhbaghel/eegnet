"""
Entry point for EEGNet experiments.

Usage
-----
# Within-subject  (trains one model per subject in TRAIN_DIR)
python main.py --mode within

# Cross-subject LOSO  (leave-one-subject-out over all subjects)
python main.py --mode cross

# Cross-subject, specific held-out subjects
python main.py --mode cross --test-subjects S001 S010 S023

# Override any default from config.py
python main.py --mode within --epochs 100 --batch-size 32 --lr 5e-4

Assumes your EEGNet model class is defined in eegnet.py and exported as EEGNet.
"""

import argparse
import random
import numpy as np
import torch

from config import (
    TRAIN_DIR, TEST_DIR,
    N_CHANNELS, N_CLASSES,
    TARGET_SFREQ, TMIN, TMAX,
    BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    SEED,
)
from data_loader import load_all, load_subject_epochs, list_subjects
from dataset import build_loaders
from splits import within_subject_split, cross_subject_split, leave_one_subject_out
from train import train, test

# ---- Import your EEGNet model -------------------------------------------
# Adjust the import if your file/class name differs.
from eegnet import EEGNet


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------
def make_model(n_samples: int) -> EEGNet:
    """
    Instantiate EEGNet with dimensions matching the loaded data.
    Adjust constructor arguments to match your EEGNet implementation.

    n_samples = int((TMAX - TMIN) * TARGET_SFREQ)  e.g. 2.0 * 128 = 256
    """
    return EEGNet(
        n_classes=N_CLASSES,
        n_channels=N_CHANNELS,
        n_samples=n_samples,
    )


# -----------------------------------------------------------------------
# Experiment modes
# -----------------------------------------------------------------------

def run_within_subject(args):
    """Train and evaluate one model per subject (from TRAIN_DIR)."""
    subjects = list_subjects(TRAIN_DIR)
    print(f"\n{'='*60}")
    print(f"WITHIN-SUBJECT experiment  ({len(subjects)} subjects)")
    print(f"{'='*60}")

    results = {}
    for subject in subjects:
        print(f"\n>>> Subject: {subject}")
        X, y = load_subject_epochs(TRAIN_DIR, subject)
        n_samples = X.shape[-1]

        splits = within_subject_split(X, y)
        X_train, y_train, X_val, y_val, X_test, y_test = splits

        train_loader, val_loader, test_loader = build_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=args.batch_size,
        )

        model = make_model(n_samples)
        train(
            model, train_loader, val_loader,
            checkpoint_name=f"{subject}_within.pt",
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=args.patience,
        )
        acc, _, _ = test(model, test_loader)
        results[subject] = acc

    # Summary
    accs = list(results.values())
    print(f"\n{'='*60}")
    print(f"Within-subject results:")
    for s, a in results.items():
        print(f"  {s}: {a:.4f}")
    print(f"  Mean ± Std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    return results


def run_cross_subject(args):
    """
    Cross-subject experiment.
    If --test-subjects is given, hold out exactly those subjects.
    Otherwise run full Leave-One-Subject-Out (LOSO) evaluation.
    """
    # Load everything from TRAIN_DIR; also load TEST_DIR if it exists and
    # is different, then merge.
    all_data = load_all(TRAIN_DIR)

    import os
    if os.path.isdir(TEST_DIR) and TEST_DIR != TRAIN_DIR:
        test_data = load_all(TEST_DIR)
        # Merge; TEST_DIR subjects take precedence on collision
        all_data = {**all_data, **test_data}

    if args.test_subjects:
        # Single held-out group
        test_subjects = args.test_subjects
        print(f"\n{'='*60}")
        print(f"CROSS-SUBJECT (fixed hold-out): test = {test_subjects}")
        print(f"{'='*60}")

        splits = cross_subject_split(all_data, test_subjects)
        X_train, y_train, X_val, y_val, X_test, y_test = splits
        n_samples = X_train.shape[-1]

        train_loader, val_loader, test_loader = build_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=args.batch_size,
        )
        model = make_model(n_samples)
        train(
            model, train_loader, val_loader,
            checkpoint_name="cross_fixed.pt",
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=args.patience,
        )
        acc, _, _ = test(model, test_loader)
        return {tuple(test_subjects): acc}

    else:
        # Full LOSO
        print(f"\n{'='*60}")
        print(f"CROSS-SUBJECT  LOSO  ({len(all_data)} subjects)")
        print(f"{'='*60}")

        results = {}
        for subject, splits in leave_one_subject_out(all_data):
            print(f"\n>>> Held-out subject: {subject}")
            X_train, y_train, X_val, y_val, X_test, y_test = splits
            n_samples = X_train.shape[-1]

            train_loader, val_loader, test_loader = build_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=args.batch_size,
            )
            model = make_model(n_samples)
            train(
                model, train_loader, val_loader,
                checkpoint_name=f"{subject}_loso.pt",
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
                patience=args.patience,
            )
            acc, _, _ = test(model, test_loader)
            results[subject] = acc

        accs = list(results.values())
        print(f"\n{'='*60}")
        print(f"LOSO results:")
        for s, a in results.items():
            print(f"  {s}: {a:.4f}")
        print(f"  Mean ± Std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        return results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EEGNet training pipeline")
    p.add_argument("--mode", choices=["within", "cross"], required=True,
                   help="'within' for within-subject, 'cross' for cross-subject LOSO")
    p.add_argument("--test-subjects", nargs="+", default=None,
                   metavar="S",
                   help="(cross mode only) specific subject IDs to hold out. "
                        "Defaults to full LOSO if not given.")
    p.add_argument("--epochs",       type=int,   default=MAX_EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--patience",     type=int,   default=EARLY_STOP_PATIENCE)
    p.add_argument("--seed",         type=int,   default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    if args.mode == "within":
        run_within_subject(args)
    else:
        run_cross_subject(args)
