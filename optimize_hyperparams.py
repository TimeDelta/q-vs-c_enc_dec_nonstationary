import numpy as np
import math

from training import *
from loss import *
from models import ENTANGLEMENT_OPTIONS, ENTANGLEMENT_GATES, ROTATION_GATES
from analysis import check_for_overfitting, MODEL_TYPES

MAX_NUM_BLOCKS = 1 # per encoder AND per decoder

def sample_hyperparameters(num_features):
    return {
        'bottleneck_size': num_features // 2,
        'num_blocks': np.random.randint(1, MAX_NUM_BLOCKS+1), # per encoder AND per decoder
        'learning_rate': 10 ** np.random.uniform(-3, -1),
        'max_penalty_weight': 2.0,
        'entanglement_topology': 'circular', # np.random.choice(ENTANGLEMENT_OPTIONS),
        'entanglement_gate': np.random.choice(ENTANGLEMENT_GATES),
        'embedding_gate': np.random.choice(ROTATION_GATES),
        'block_gate': np.random.choice(ROTATION_GATES),
    }

def get_loss(data, model_type, config, allocated_epochs):
    print(config)
    np.random.seed(12984)

    training = data[0]
    validation = data[1]
    num_features = len(training[0][1][0])

    print(f'Training {model_type.upper()} for {allocated_epochs}')
    is_recurrent = 'r' in model_type
    autoregressive = 'te' in model_type # transition encoder

    if model_type.startswith('c'):
        model = ClassicalEncoderDecoder(num_features, config, is_recurrent)
        trash_penalty_fn = classical_trash_penalty
    elif model_type.startswith('q'):
        model = QuantumEncoderDecoder(num_features, config, is_recurrent)
        trash_penalty_fn = trash_qubit_penalty
    else:
        raise Exception('Unexpected model type: ' + model_type)

    if autoregressive:
        loss_fn = autoencoder_cost_function(trash_penalty_fn)
    else:
        loss_fn = autoregressive_cost_function(trash_penalty_fn)

    trained_model, cost_history, validation_costs = \
        train_adam(training, validation, loss_fn, config, model, allocated_epochs)

    print(' ', validation_costs)
    if allocated_epochs >= 10:
        mean_training_costs = np.array(cost_history[-1]) / len(training)
        mean_validation_costs = np.sum(validation_costs[:,1:], axis=0) / len(validation)
        if check_for_overfitting(mean_training_costs, mean_validation_costs, threshold=.5): # throw away any configs that lead to obvious overfitting
            return float('inf')
    # scale the cost by the % of max num of blocks used
    # TODO: incorporate bottleneck size
    scale = float(config['num_blocks']) / float(MAX_NUM_BLOCKS)
    return sum(cost_history[-1]) * scale

def hyperband_search(data, max_training_epochs=16, reduction_factor=2):
    """
    Parameters:
      - type: "QAE" or "QTE" (case-insensitive)
      - max_training_epochs: maximum number of epochs allocated to any configuration
      - reduction_factor: factor by which successive configuration evals are reduced each round (eta)

    Returns:
      optimal_config, optimal_loss
    """
    print(f'Performing hyperband search over training hyperparameters')
    max_bracket = int(np.floor(np.log(max_training_epochs) / np.log(reduction_factor)))
    total_budget = (max_bracket + 1) * max_training_epochs
    optimal_config = None
    optimal_loss = float('inf')

    print('  num series in training:', len(data[0]))
    num_features = len(data[0][0][1][0])

    for bracket in reversed(range(max_bracket + 1)):
        initial_num_configs = int(np.ceil(total_budget / max_training_epochs / (bracket + 1) * reduction_factor ** bracket))
        initial_allocated_epochs = max_training_epochs * reduction_factor ** (-bracket)

        print(f"Bracket {max_bracket-bracket}: Starting with {initial_num_configs} configurations, each with {initial_allocated_epochs} epochs.")
        configs = [sample_hyperparameters(num_features) for _ in range(initial_num_configs)]
        print('Created configs')

        # successive reduction in num configs
        for round_index in range(bracket + 1):
            num_configs_this_round = int(np.floor(initial_num_configs * reduction_factor ** (-round_index)))
            epochs_this_round = int(initial_allocated_epochs * reduction_factor ** (round_index))

            print('Round', round_index, '-', num_configs_this_round, 'configs for', epochs_this_round, ' epochs this round')
            round_losses = [np.mean([get_loss(data, m, config, epochs_this_round) for m in MODEL_TYPES]) for config in configs]
            print(f"  Round {round_index}: {epochs_this_round} epochs; best loss = {min(round_losses):.4f}")

            for i, loss in enumerate(round_losses):
                if loss < optimal_loss:
                    optimal_loss = loss
                    optimal_config = configs[i]

            best_indices = np.argsort(round_losses)[:num_configs_this_round]
            configs = [configs[i] for i in best_indices]

    return optimal_config, optimal_loss

def get_best_config(dataset_partitions, max_training_epochs=16, reduction_factor=4):
    input_data = [[], []]
    for _, (training, validation) in sorted(dataset_partitions.items()):
        # take one training and one validation series from each dataset
        t = np.random.randint(0, len(training))
        input_data[0].append(training[t])
        v = np.random.randint(0, len(validation))
        input_data[1].append(validation[v])

    best_config, best_loss = hyperband_search(input_data, max_training_epochs, reduction_factor)
    print("\nBest hyperparameter configuration found:")
    print(best_config)
    print("With estimated loss:", best_loss)
    return best_config

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Find single optimal hyperparameter config to use across all model types in this experiment."
    )
    parser.add_argument("data_directory", type=str, nargs='?', default='generated_datasets', help="Path to the directory containing the training data.")
    parser.add_argument("--reduction_factor", type=int, default=4, help="Factor by which successive configuration evals are reduced each round.")
    parser.add_argument("--max_training_epochs", type=int, default=16, help="Maximum number of epochs allocated to any configuration.")
    args = parser.parse_args()

    from data_importers import import_generated
    best_config = get_best_config(import_generated(args.data_directory, args.max_training_epochs, args.reduction_factor))
    best_config_path = os.path.join(args.data_directory, 'best_config.json')
    with open(best_config_path, 'w') as file:
        json.dump(best_config, file, indent=2)
