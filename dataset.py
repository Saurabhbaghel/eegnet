"""
PyTorch Dataset that wraps numpy arrays into the 4-D tensor shape
that EEGNet expects:  (batch, 1, n_channels, n_samples)

Helpers
-------
build_loaders()        - normalise + wrap numpy arrays into DataLoaders
block_kfold_loaders()  - block-wise k-fold CV generator (default k=4)
                         2 blocks train | 1 block val | 1 block test
                         Preserves temporal order — no leakage between
                         adjacent trials.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import normalize_epochs
from config import BATCH_SIZE, SEED


class EEGDataset(Dataset):
    """
    Wraps (X, y) numpy arrays for use with PyTorch DataLoader.

    Parameters
    ----------
    X : np.ndarray  (n_trials, n_channels, n_samples)   raw or pre-normalised
    y : np.ndarray  (n_trials,)                          integer class labels
    normalise : bool  if True, applies per-trial z-score before returning tensors
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, normalise: bool = True):
        if normalise:
            X = normalize_epochs(X)

        # EEGNet input shape: (1, n_channels, n_samples)  — single "image" channel
        X = X[:, np.newaxis, :, :]               # (n_trials, 1, n_ch, n_samples)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray  = None,
    y_test:  np.ndarray  = None,
    batch_size: int      = BATCH_SIZE,
) -> tuple:
    """
    Normalise arrays and wrap them in DataLoaders.

    Returns (train_loader, val_loader)  or
            (train_loader, val_loader, test_loader)  if test arrays are given.
    """
    train_ds = EEGDataset(X_train, y_train, normalise=True)
    val_ds   = EEGDataset(X_val,   y_val,   normalise=True)

    g = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    if X_test is not None and y_test is not None:
        test_ds     = EEGDataset(X_test, y_test, normalise=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def block_kfold_loaders(
    X:          np.ndarray,
    y:          np.ndarray,
    k:          int = 4,
    batch_size: int = BATCH_SIZE,
):
    """
    Block-wise k-fold cross-validation generator for EEG data.

    Trials are split into k contiguous blocks preserving temporal order.
    On each fold the roles of the blocks rotate:

        k=4 example
        ┌────────┬────────┬────────┬────────┐
        │ Fold 1 │ train  │ train  │  val   │  test  │  ← blocks [0,1,2,3]
        │ Fold 2 │ train  │ train  │  val   │  test  │  ← blocks [1,2,3,0]
        │ Fold 3 │ train  │ train  │  val   │  test  │  ← blocks [2,3,0,1]
        │ Fold 4 │ train  │ train  │  val   │  test  │  ← blocks [3,0,1,2]
        └────────┴────────┴────────┴────────┘

    The last block in the rotation is always the test block, the
    second-to-last is always the validation block, and the remaining
    k-2 blocks are concatenated for training.

    For k=4 this gives the 2-train / 1-val / 1-test split you asked for.
    For other values of k the same rule applies (k-2 train blocks).

    Parameters
    ----------
    X          : np.ndarray  (n_trials, n_channels, n_samples)
                 Trials must be in temporal order (as returned by
                 load_subject_epochs / load_all).
    y          : np.ndarray  (n_trials,)
    k          : number of folds / blocks  (default 4)
    batch_size : DataLoader batch size

    Yields
    ------
    fold_idx   : int  0-indexed fold number
    loaders    : tuple  (train_loader, val_loader, test_loader)
    meta       : dict with keys
                     'train_idx', 'val_idx', 'test_idx'  — numpy index arrays
                     'train_size', 'val_size', 'test_size'

    Example
    -------
    >>> for fold, (train_dl, val_dl, test_dl), meta in block_kfold_loaders(X, y):
    ...     model = make_model(X.shape[-1])
    ...     train(model, train_dl, val_dl, checkpoint_name=f"fold{fold}.pt")
    ...     acc, _, _ = test(model, test_dl)
    """
    n = len(y)
    if k < 3:
        raise ValueError(f"k must be >= 3 (need at least 1 train + 1 val + 1 test block), got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds number of trials ({n})")

    # Split trial indices into k roughly equal contiguous blocks
    # np.array_split handles uneven splits gracefully (earlier blocks get +1 trial)
    all_indices = np.arange(n)
    blocks      = np.array_split(all_indices, k)   # list of k index arrays

    print(f"\nBlock k-fold CV: k={k}, total trials={n}")
    print(f"Block sizes: {[len(b) for b in blocks]}")
    print(f"Assignment per fold: {k-2} train block(s), 1 val block, 1 test block\n")

    for fold in range(k):
        # Rotate the block order so a different block lands in each role
        # rotation: [fold, fold+1, ..., fold+k-1]  (all mod k)
        rotation = [blocks[(fold + i) % k] for i in range(k)]

        # Last block  → test
        # Second-last → val
        # All others  → train (concatenated)
        test_idx  = rotation[-1]
        val_idx   = rotation[-2]
        train_idx = np.concatenate(rotation[:-2])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # Shuffle only the training set (val/test stay in temporal order)
        g = torch.Generator().manual_seed(SEED + fold)
        train_ds = EEGDataset(X_train, y_train, normalise=True)
        val_ds   = EEGDataset(X_val,   y_val,   normalise=True)
        test_ds  = EEGDataset(X_test,  y_test,  normalise=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, generator=g, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        meta = {
            "train_idx":  train_idx,
            "val_idx":    val_idx,
            "test_idx":   test_idx,
            "train_size": len(train_idx),
            "val_size":   len(val_idx),
            "test_size":  len(test_idx),
        }

        print(f"  Fold {fold + 1}/{k} — "
              f"train: {meta['train_size']} trials, "
              f"val: {meta['val_size']} trials, "
              f"test: {meta['test_size']} trials")

        yield fold, (train_loader, val_loader, test_loader), meta