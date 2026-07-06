import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci


def load_data(subjects, runs):
    raws = []
    for subj in subjects:
        raw_fnames = eegbci.load_data(subj, runs)
        raw = concatenate_raws(
                [read_raw_edf(f, preload=True) for f in raw_fnames]
                )
        raws.append(raw)
    return raws

# Use only these subjects and runs
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]   # 8 subjects for training
TEST_SUBJECTS = [9, 10]     # 2 subjects for testing
RUNS = [6, 10, 14]      # Motor imagery: left and right


