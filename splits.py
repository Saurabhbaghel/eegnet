"""
Two experimental splits used in the EEGNet paper:

  within_subject_split(X, y)
      Splits one subject's data into train / val / test.
      Typical use: evaluate how well the model learns for a single person.

  cross_subject_split(all_data, test_subjects)
      Train on all subjects except those in test_subjects; evaluate on them.
      Typical use: test generalisation to unseen individuals.

Both functions return plain numpy arrays so they slot into build_loaders()
without further changes.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from config import SEED


# -----------------------------------------------------------------------
# 1. Within-subject split
# -----------------------------------------------------------------------

def within_subject_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    # remainder goes to test  (= 1 - train_ratio - val_ratio, default 0.15)
) -> tuple:
    """
    Stratified chronological split of a single subject's epochs.

    Stratified so every class is proportionally represented in each split.
    Trials are NOT shuffled before splitting to respect temporal ordering
    (avoids leakage from overlapping time windows if you ever switch to
    sliding-window epoching).

    Parameters
    ----------
    X            : (n_trials, n_channels, n_samples)
    y            : (n_trials,)
    train_ratio  : fraction of data used for training
    val_ratio    : fraction of data used for validation

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1.0, \
        "train_ratio + val_ratio must be < 1.0 (remainder becomes test set)"

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split off the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=SEED,
        shuffle=True,
    )

    # Then split the remaining data into train / val
    relative_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        stratify=y_trainval,
        random_state=SEED,
        shuffle=True,
    )

    _print_split_summary("Within-subject", y_train, y_val, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------------------------------------------------
# 2. Cross-subject split
# -----------------------------------------------------------------------

def cross_subject_split(
    all_data: dict,
    test_subjects: list,
    val_ratio: float = 0.15,
) -> tuple:
    """
    Pool all subjects, hold out `test_subjects` for testing,
    and carve a validation set from the remaining training subjects.

    Parameters
    ----------
    all_data      : {"S001": (X, y), "S002": (X, y), ...}
    test_subjects : list of subject IDs to hold out, e.g. ["S001", "S010"]
    val_ratio     : fraction of training subjects' data used for validation

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    test_set  = set(test_subjects)
    train_subs = [s for s in all_data if s not in test_set]

    if not train_subs:
        raise ValueError("No training subjects remain after holding out test_subjects.")
    if not test_set.intersection(all_data):
        raise ValueError(f"None of {test_subjects} found in all_data keys.")

    # Concatenate test subjects
    X_test = np.concatenate([all_data[s][0] for s in test_subjects if s in all_data])
    y_test = np.concatenate([all_data[s][1] for s in test_subjects if s in all_data])

    # Concatenate all training subjects
    X_all_train = np.concatenate([all_data[s][0] for s in train_subs])
    y_all_train = np.concatenate([all_data[s][1] for s in train_subs])

    # Carve out a validation set from the training pool
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_train, y_all_train,
        test_size=val_ratio,
        stratify=y_all_train,
        random_state=SEED,
        shuffle=True,
    )

    print(f"\nCross-subject split: "
          f"train subjects={train_subs}, test subjects={test_subjects}")
    _print_split_summary("Cross-subject", y_train, y_val, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def leave_one_subject_out(all_data: dict, val_ratio: float = 0.15):
    """
    Generator that yields one (subject, splits) at a time for LOSO evaluation.

    Usage
    -----
        for subject, splits in leave_one_subject_out(all_data):
            X_train, y_train, X_val, y_val, X_test, y_test = splits
            ...

    This is the standard cross-subject evaluation protocol in the EEGNet paper.
    """
    subjects = list(all_data.keys())
    for subject in subjects:
        splits = cross_subject_split(all_data, test_subjects=[subject],
                                     val_ratio=val_ratio)
        yield subject, splits


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------

def _print_split_summary(tag, y_train, y_val, y_test):
    print(f"\n[{tag}] Split sizes — "
          f"train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  {split_name} class distribution: {dist}")
