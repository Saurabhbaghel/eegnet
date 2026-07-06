"""
Central configuration for the EEGNet pipeline.

Dataset layout:
    data/train/S001/S001R04.edf
    data/train/S001/S001R06.edf
    ...
    data/test/S088/S088R04.edf

Dataset: PhysioNet EEG Motor Movement/Imagery (EEGBCI) accessed via local files.
"""
import os

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
DATA_ROOT      = "data/MNE-eegbci-data/files/eegmmidb/1.0.0"
TRAIN_DIR      = os.path.join(DATA_ROOT, "train")
TEST_DIR       = os.path.join(DATA_ROOT, "test")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Signal parameters  (match EEGNet paper's SMR setup)
# -----------------------------------------------------------------------
RAW_SFREQ    = 160.0   # native PhysioNet EEGBCI sampling rate
TARGET_SFREQ = 128.0   # resample target
L_FREQ       = 4.0     # band-pass low  edge (Hz)
H_FREQ       = 40.0    # band-pass high edge (Hz)

# Epoch window relative to event onset.
# 0–2 s  →  256 samples @ 128 Hz  (EEGNet paper default)
TMIN, TMAX = 0.0, 2.0

N_CHANNELS = 64        # PhysioNet EEGBCI is a 64-channel cap

# -----------------------------------------------------------------------
# Task / label configuration
# -----------------------------------------------------------------------
# PhysioNet run numbers:
#   R03/07/11 – motor EXECUTION  left fist (T1) vs right fist (T2)
#   R04/08/12 – motor IMAGERY    left fist (T1) vs right fist (T2)
#   R05/09/13 – motor EXECUTION  both fists (T1) vs both feet (T2)
#   R06/10/14 – motor IMAGERY    both fists (T1) vs both feet (T2)
#
# 4-class imagery task used in the EEGNet paper:
RUN_GROUPS = {
    "LR": [4, 8, 12],   # imagery: left fist vs right fist
    "FF": [6, 10, 14],  # imagery: both fists vs both feet
}

# (group, annotation_name) -> class index
LABEL_MAP = {
    ("FF", "T1"): 0,  # both fists
    ("FF", "T2"): 1,  # both feet
}
CLASS_NAMES = ["both_fists", "both_feet"]
N_CLASSES   = len(CLASS_NAMES)

# -----------------------------------------------------------------------
# Training defaults
# -----------------------------------------------------------------------
BATCH_SIZE          = 64
LR                  = 1e-3
WEIGHT_DECAY        = 0.0
MAX_EPOCHS          = 200
EARLY_STOP_PATIENCE = 20
SEED                = 42
