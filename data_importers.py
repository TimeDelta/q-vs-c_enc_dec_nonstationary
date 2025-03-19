import pickle
import glob
import os
import re
import numpy as np
import mne

import os
import numpy as np
import re
import os
import numpy as np
import hashlib

def compute_series_hash(series):
    """
    SHA-256 hash for a numpy array
    """
    return hashlib.sha256(series.tobytes()).hexdigest()

def import_generated(generated_datasets_dir, train_ratio=0.75, seed=42):
    """
    Load training and validation partitions separately for each dataset. For each series file in a
    dataset, compute its hash. If the hash is in the set of grid series hashes (loaded from
    'grid_series_hashes.npy'), force that series into the validation set. The remaining series are
    randomly split into training and validation.


    Returns:
      dict: Mapping each dataset index (parsed from directory name) to a tuple
            (training_series, validation_series), with each series as a numpy array.
    """
    np.random.seed(seed)
    partitions = {}

    dataset_dirs = [d for d in os.listdir(generated_datasets_dir) if os.path.isdir(os.path.join(generated_datasets_dir, d))]
    dataset_dirs.sort()

    grid_hashes = [
        compute_series_hash(np.load(os.path.join(generated_datasets_dir, f)))
        for f in os.listdir(generated_datasets_dir) if f.startswith("series_cell_")
    ]

    for dataset_dir in dataset_dirs:
        full_path = os.path.join(generated_datasets_dir, dataset_dir)
        series_files = [f for f in os.listdir(full_path) if f.endswith('.npy')]
        series_files.sort()
        all_indices = np.arange(len(series_files))

        forced_indices = []
        non_forced_indices = []
        # Load each file, compute its hash, and partition accordingly.
        series_hashes = {}
        for idx, f in enumerate(series_files):
            series = np.load(os.path.join(full_path, f))
            s_hash = compute_series_hash(series)
            series_hashes[idx] = s_hash
            if s_hash in grid_hashes:
                forced_indices.append(idx)
            else:
                non_forced_indices.append(idx)

        non_forced_indices = np.array(non_forced_indices, dtype=int)
        np.random.shuffle(non_forced_indices)
        num_train = int(np.floor(train_ratio * len(non_forced_indices)))
        train_indices = non_forced_indices[:num_train]
        val_indices = non_forced_indices[num_train:]

        final_val_indices = np.concatenate([np.array(forced_indices, dtype=int), val_indices])

        training_series = [np.load(os.path.join(full_path, series_files[i])) for i in train_indices]
        validation_series = [np.load(os.path.join(full_path, series_files[i])) for i in final_val_indices]

        dataset_index = int(dataset_dir.split('_')[-1])
        partitions[dataset_index] = (training_series, validation_series)

    return partitions

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