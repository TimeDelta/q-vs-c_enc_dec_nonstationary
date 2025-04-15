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

def import_generated(generated_datasets_dir, train_ratio=2.0/3.0, seed=42):
    """
    Load training and validation partitions separately for each dataset.
    For each series file in a dataset, compute its hash. If the hash is in the set of grid
    series hashes (loaded from files starting with 'series_cell_'), force that series into
    the validation set. The remaining series are randomly split into training and validation.

    Only datasets that have at least one series_i.npy file whose hash matches one of the grid_hashes
    are included in the final partitions.

    At the end, adjust each dataset so that the validation partition has the same size,
    by moving series from training to validation as needed.

    Returns:
      dict: Mapping each dataset index (parsed from directory name) to a tuple
            (training_series, validation_series), where each series is stored as a tuple
            (series_index, numpy array). Downstream code can extract the series data via tuple[1].
    """
    print('Importing generated data from ' + generated_datasets_dir)
    np.random.seed(seed)
    partitions = {}

    dataset_dirs = [d for d in os.listdir(generated_datasets_dir) if os.path.isdir(os.path.join(generated_datasets_dir, d))]
    dataset_dirs.sort()

    series_num_pattern = re.compile(r'series_(\d+)\.npy$')

    grid_hashes = [
        compute_series_hash(np.load(os.path.join(generated_datasets_dir, f)))
        for f in os.listdir(generated_datasets_dir) if f.startswith("series_cell_")
    ]

    for dataset_dir in dataset_dirs:
        full_path = os.path.join(generated_datasets_dir, dataset_dir)

        series_files = []
        for fname in os.listdir(full_path):
            if fname.endswith('.npy'):
                match = series_num_pattern.search(fname)
                if match:
                    series_index = int(match.group(1))
                    series_files.append((series_index, fname))
                else:
                    raise Exception('Unable to parse series num from: ' + fname)

        series_files.sort(key=lambda x: x[0])
        series_dict = {index: fname for index, fname in series_files}

        forced_indices = [] # must be in validation (grid hash match)
        non_forced_indices = [] # will be randomly partitioned

        # compute hash for each series file
        for index, fname in series_files:
            series = np.load(os.path.join(full_path, fname))
            s_hash = compute_series_hash(series)
            if s_hash in grid_hashes:
                forced_indices.append(index)
            else:
                non_forced_indices.append(index)

        # only include this dataset if at least one series hash matches a grid hash
        if not forced_indices:
            continue

        # Randomly split non-forced series into training and validation partitions
        non_forced_indices = np.array(non_forced_indices, dtype=int)
        np.random.shuffle(non_forced_indices)
        desired_num_train = train_ratio * (len(forced_indices) + len(non_forced_indices))
        if len(non_forced_indices) > desired_num_train:
            num_train = int(desired_num_train)
        else:
            num_train = len(non_forced_indices)
        train_indices = sorted(non_forced_indices[:num_train])
        val_indices = sorted(non_forced_indices[num_train:])

        # Combine forced validation indices with non-forced validation indices
        final_val_indices = sorted(np.concatenate([np.array(forced_indices, dtype=int), val_indices]))

        training_series = [(i, np.load(os.path.join(full_path, series_dict[i]))) for i in train_indices]
        validation_series = [(i, np.load(os.path.join(full_path, series_dict[i]))) for i in final_val_indices]

        dataset_index = int(dataset_dir.split('_')[-1])
        partitions[dataset_index] = (training_series, validation_series)

    # equalize validation partition sizes across datasets
    max_validation_size = max(len(val) for (_, val) in partitions.values())
    print('  Ensuring validation partitions are all of size', max_validation_size)
    for key, (train_series, val_series) in partitions.items():
        missing = max_validation_size - len(val_series)
        if missing > 0:
            # shuffle to avoid introducing bias from complexity pattern in data generation
            np.random.shuffle(train_series)
            moved = train_series[:missing]
            train_series = train_series[missing:]
            val_series = val_series + moved
            partitions[key] = (train_series, val_series)
        print(f'  Dataset {key} final validation series indices: {[idx for idx, _ in val_series]}')

    return partitions

def _filepath_regex(filepath, regex):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if re.search(regex, filepath):
        return True
    else:
        return False