from data_generation import generate_data
from optimize_hyperparams import get_best_config
from training import train_and_analyze_bottlenecks
from analysis import run_analysis

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Run this experiment."
    )
    parser.add_argument("--data_directory", type=str, default='generated_datasets', help="Directory in which to store the generated data.")
    args = parser.parse_args()

    generate_data(args.data_directory)
    dataset_partitions = import_generated(args.data_directory)
    best_config = get_best_config(dataset_partitions)
    num_features = len(next(iter(dataset_partitions.values()))[0][0][1][0])
    train_and_analyze_bottlenecks(args.data_directory, dataset_partitions, num_features, num_epochs, best_config)
    run_analysis(dataset_partitions, args.data_directory)
