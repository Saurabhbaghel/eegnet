"""
Loads raw .edf recordings from your local folder structure and extracts
labeled epochs per subject.

Layout assumed:
    <root>/S001/S001R04.edf
    <root>/S001/S001R06.edf
    ...

Public API
----------
    list_subjects(root)                -> ["S001", "S002", ...]
    load_subject_epochs(root, subject) -> (X, y)
        X : np.ndarray  (n_trials, n_channels, n_samples)
        y : np.ndarray  (n_trials,)
    load_all(root)                     -> {"S001": (X, y), ...}
"""

import os
import glob
import numpy as np
import mne

from config import RUN_GROUPS, LABEL_MAP, TMIN, TMAX
from preprocessing import preprocess_raw

mne.set_log_level("ERROR")


# -----------------------------------------------------------------------
# Subject discovery
# -----------------------------------------------------------------------

def list_subjects(root: str) -> list[str]:
    """Return sorted subject IDs (folder names) found directly under `root`."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    subjects = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    return sorted(subjects)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _glob_run_files(subject_dir: str, run_numbers: list[int]) -> list[str]:
    """
    Return sorted .edf paths for the requested run numbers inside subject_dir.
    Matches filenames like S001R04.edf  (zero-padded 2-digit run number).
    """
    paths = []
    for run in run_numbers:
        pattern = os.path.join(subject_dir, f"*R{run:02d}.edf")
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def _epochs_from_edf(edf_path: str, group: str):
    """
    Load one .edf file, preprocess, and slice labeled epochs from it.

    Returns
    -------
    X : np.ndarray  (n_valid_trials, n_channels, n_samples)  or empty
    y : np.ndarray  (n_valid_trials,)                        or empty
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw = preprocess_raw(raw)

    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Keep only T1 / T2 annotations (rest/T0 is not a motor imagery class)
    wanted = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
    if not wanted:
        return np.empty((0,)), np.empty((0,))

    epochs = mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,       # no baseline: we normalise per-trial in preprocessing
        preload=True,
        verbose=False,
    )

    inv = {v: k for k, v in wanted.items()}   # code -> "T1" / "T2"
    data   = epochs.get_data()                 # (n_epochs, n_ch, n_times)
    codes  = epochs.events[:, -1]

    X_list, y_list = [], []
    for trial, code in zip(data, codes):
        label = LABEL_MAP.get((group, inv[code]))
        if label is None:
            continue
        X_list.append(trial)
        y_list.append(label)

    if not X_list:
        return np.empty((0,)), np.empty((0,))

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_subject_epochs(root: str, subject: str):
    """
    Load all imagery-run epochs for a single subject and concatenate them.

    Parameters
    ----------
    root    : path to the split directory (TRAIN_DIR or TEST_DIR)
    subject : folder name, e.g. "S001"

    Returns
    -------
    X : np.ndarray  shape (n_trials, n_channels, n_samples)
    y : np.ndarray  shape (n_trials,)   integer class labels
    """
    subject_dir = os.path.join(root, subject)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    all_X, all_y = [], []

    for group, run_numbers in RUN_GROUPS.items():
        edf_paths = _glob_run_files(subject_dir, run_numbers)

        if not edf_paths:
            print(f"  [warn] {subject} – no files found for group '{group}' "
                  f"(runs {run_numbers}) in {subject_dir}")
            continue

        for edf_path in edf_paths:
            X, y = _epochs_from_edf(edf_path, group)
            if y.ndim == 0 or len(y) == 0:
                continue
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        raise RuntimeError(
            f"No usable epochs found for {subject}. "
            "Check that your run files match the RUN_GROUPS in config.py."
        )

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def load_all(root: str) -> dict:
    """
    Load every subject found under `root`.

    Returns
    -------
    dict  { "S001": (X, y), "S002": (X, y), ... }
    """
    dataset = {}
    subjects = list_subjects(root)
    print(f"Found {len(subjects)} subjects under {root}")

    for subject in subjects:
        try:
            X, y = load_subject_epochs(root, subject)
            dataset[subject] = (X, y)
            print(f"  {subject}: {X.shape[0]} trials, "
                  f"X shape {X.shape}, classes {np.unique(y).tolist()}")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  [warn] skipping {subject}: {e}")

    return dataset


# -----------------------------------------------------------------------
# Quick smoke-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from config import TRAIN_DIR

    subs = list_subjects(TRAIN_DIR)
    print(f"\nFirst subject: {subs[0]}")
    X, y = load_subject_epochs(TRAIN_DIR, subs[0])
    print(f"X: {X.shape}  y: {y.shape}  labels: {np.unique(y)}")
