"""
Signal preprocessing matching the EEGNet paper's SMR pipeline:
  1. Band-pass filter  (4–40 Hz, zero-phase FIR)
  2. Resample          (160 Hz → 128 Hz)
  3. Per-trial standardisation  (zero-mean, unit-variance across time, per channel)

preprocess_raw()  operates on an MNE Raw object (called inside data_loader).
normalize_epochs() operates on a numpy array after epoching (called in dataset.py).
"""

import numpy as np
import mne

from config import L_FREQ, H_FREQ, RAW_SFREQ, TARGET_SFREQ


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Apply band-pass filter and resample to a loaded MNE Raw object.
    Mutates `raw` in-place and also returns it for chaining.

    Parameters
    ----------
    raw : mne.io.BaseRaw  (must already be preloaded)

    Returns
    -------
    raw : same object, filtered and resampled
    """
    # ---- 1. Band-pass filter ------------------------------------------------
    # method='fir' + phase='zero' gives zero-phase filtering (no causal delay).
    # filter_length='auto' lets MNE pick a sensible FIR order.
    raw.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        method="fir",
        fir_window="hamming",
        phase="zero",
        verbose=False,
    )

    # ---- 2. Resample --------------------------------------------------------
    # Only resample if the file's native rate differs from the target.
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 0.5:
        raw.resample(TARGET_SFREQ, verbose=False)

    return raw


def normalize_epochs(X: np.ndarray) -> np.ndarray:
    """
    Per-trial, per-channel z-score normalisation.

    The EEGNet paper does not apply a baseline correction; instead each trial
    is standardised so that the temporal mean is 0 and std is 1 per channel.
    This removes DC offsets and amplitude differences across channels/subjects.

    Parameters
    ----------
    X : np.ndarray  shape (n_trials, n_channels, n_samples)

    Returns
    -------
    X_norm : np.ndarray  same shape, float32
    """
    X = X.astype(np.float32)
    # mean and std over the time axis, keeping dims for broadcasting
    mu  = X.mean(axis=-1, keepdims=True)          # (n_trials, n_ch, 1)
    std = X.std(axis=-1, keepdims=True) + 1e-8    # avoid divide-by-zero
    return (X - mu) / std
