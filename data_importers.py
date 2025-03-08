import pickle
import glob
import os
import re
import numpy as np

def import_FACED(data_directory):
    """
    FACED: Finer-grained Affective Computing EEG Dataset
    https://www.synapse.org/Synapse:syn50614821
    Use the Processed_data.zip file
    """
    eeg_data = []
    print('Reading FACED data ...')
    # cohorts 1 and 2 have different placement for EEG channels; just take cohort 1
    valid_subj_ids_re = r'sub0([0-5][0-9]|60)'
    subjects_to_consider = [
        f for f in glob.glob(os.path.join(data_directory, "**/sub*.pkl"), recursive=True) if _filepath_regex(f, valid_subj_ids_re)
    ]
    for file in subjects_to_consider:
        print(file)
        with open(file, "rb") as data:
            try:
                # only use EEG recording for 10th video (45 sec, In Bruges, PED, Negative, Sadness)
                eeg_data.append(pickle.load(data)[9])
            except e as Exception:
                raise e
    # final shape should be (61, 32, 7500)
    return np.stack(eeg_data, axis=0)

def _filepath_regex(filepath, regex):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if re.search(regex, filepath):
        return True
    else:
        return False