import mne
import glob
import os
import re
import numpy as np

def import_FACED(data_directory):
    """
    FACED: Finer-grained Affective Computing EEG Dataset
    https://www.synapse.org/Synapse:syn50614821
    """
    eeg_data = []
    print('Reading FACED data')
    # cohorts 1 and 2 have different placement for EEG channels; just take cohort 1
    valid_subj_ids_re = r'sub0([0-5][0-9]|60)'
    subjects_to_consider = [
        f for f in glob.glob(os.path.join(data_directory, "**/data.bdf"), recursive=True) if _filepath_regex(f, valid_subj_ids_re)
    ]
    for file in subjects_to_consider:
        try:
            raw_data = mne.io.read_raw_bdf(file, preload=True)
            eeg_data.append(raw_data)
        except e as Exception:
            raise e
    input_data = np.stack(eeg_data, axis=0)
    return input_data

def _filepath_regex(filepath, regex):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if re.search(regex, filepath):
        return True
    else:
        return False