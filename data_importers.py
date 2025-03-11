import pickle
import glob
import os
import re
import numpy as np
import mne

def import_single_subject(data_directory):
    """
    OOPS. this is 32-channels also
    https://figshare.com/articles/dataset/_b_EEG_dataset_of_repeated_measurements_from_single-individual_b_/24877770/3?file=45140680
    from this paper: https://www.nature.com/articles/s41597-024-03241-z#Sec6
    Use the EEG/EEG_Cleaned_data directory
    Citation:
      - Wang, Guangjun; JIa, Shuyong; Liu, Qi (2023). EEG dataset of repeated measurements from single-individual. figshare. Dataset.
        https://doi.org/10.6084/m9.figshare.24877770.v3
    """
    eeg_data = []
    print('Reading Single Subject data (https://doi.org/10.1038/s41597-024-03241-z) ...')
    files_to_consider = [ # .fdt files get automatically handled by data loader if present
        f for f in glob.glob(os.path.join(data_directory, "**/*.set"), recursive=True)
    ]
    for file in files_to_consider:
        print(file)
        raw = mne.io.read_epochs_eeglab(file)
        data = raw.get_data(copy=False)
        eeg_data.extend(data)
    return np.stack(eeg_data, axis=0)


def import_FACED(data_directory):
    """
    FACED: Finer-grained Affective Computing EEG Dataset
    https://www.synapse.org/Synapse:syn50614821
    Use the Processed_data.zip file
    !!! Qiskit Bug currently prevents use of this dataset without dimensionality reduction due to
    !!! use of 32 channels causing overflow issue: See https://github.com/Qiskit/qiskit/issues/13974
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