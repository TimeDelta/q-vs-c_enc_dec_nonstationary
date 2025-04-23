import os
import json
import glob

from data_generation import generate_data
from analysis import MODEL_TYPES
from data_importers import import_generated

import argparse
parser = argparse.ArgumentParser(
    description="Run this experiment."
)
parser.add_argument("data_directory", type=str, nargs='?', default='generated_datasets', help="Directory in which to store the generated data.")
args = parser.parse_args()

dataset_partitions = import_generated(args.data_directory)

num_features = len(next(iter(dataset_partitions.values()))[0][0][1][0])
pytorch_models = len(glob.glob(os.path.join(args.data_directory, '*.pth')))
qiskit_models = len(glob.glob(os.path.join(args.data_directory, '*.qpy')))
num_required_models = len(dataset_partitions) * len(MODEL_TYPES)
num_missing_models = num_required_models - (pytorch_models + qiskit_models)
print(num_missing_models, f'models ({num_missing_models/num_required_models}%) left')
