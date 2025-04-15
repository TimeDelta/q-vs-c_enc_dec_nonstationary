import os

from data_generation import generate_data
from optimize_hyperparams import get_best_config
from training import train_and_analyze_bottlenecks
from analysis import run_analysis, MODEL_TYPES
from data_importers import import_generated

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Run this experiment."
    )
    parser.add_argument("data_directory", type=str, nargs='?', default='generated_datasets', help="Directory in which to store the generated data.")
    args = parser.parse_args()

    if not os.path.exists(args.data_directory):
        generate_data(args.data_directory)
    else:
        print('!! Skipping data generation due to existing specified directory')
    dataset_partitions = import_generated(args.data_directory)

    best_config_path = os.path.join(args.data_directory, 'best_config.json')
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as file:
            best_config = json.load(file)
        print('Loaded best config from hyperparameter optimization')
    else:
        best_config = get_best_config(dataset_partitions)
        with open(best_config_path, 'w') as file:
            json.dump(best_config, file, indent=2)

    num_features = len(next(iter(dataset_partitions.values()))[0][0][1][0])
    pytorch_models = len(glob.glob(os.path.join(args.data_directory, '*.pth')))
    qiskit_models = len(glob.glob(os.path.join(args.data_directory, '*.qpy')))
    num_required_models = len(dataset_partitions) * len(MODEL_TYPES)
    num_missing_models = num_required_models - (pytorch_models + qiskit_models)
    if num_missing_models > 0:
        if pytorch_models > 0 or qiskit_models > 0:
            print(f'Missing {num_missing_models} models')
        train_and_analyze_bottlenecks(args.data_directory, dataset_partitions, num_features, num_epochs, best_config)
    run_analysis(dataset_partitions, args.data_directory)
    print('Experiment complete')
