import numpy as np
import math
from training import train_adam, qae_cost_function, qte_cost_function, ENTANGLEMENT_OPTIONS, ENTANGLEMENT_GATES

MAX_NUM_BLOCKS = 1 # per encoder AND per decoder

def sample_hyperparameters(num_qubits):
    return {
        'bottleneck_size': 2,
        'num_blocks': np.random.randint(1, MAX_NUM_BLOCKS+1), # per encoder AND per decoder
        'learning_rate': 10 ** np.random.uniform(-4, -1),
        'penalty_weight': .75,
        'entanglement_topology': np.random.choice(ENTANGLEMENT_OPTIONS),
        'entanglement_gate': np.random.choice(ENTANGLEMENT_GATES),
    }

def get_loss(data, type, config, allocated_epochs):
    print(config)
    np.random.seed(12984)
    if type.lower() == 'qae':
        trained_params, cost_history, validation_cost, _, _, _ = \
            train_adam(data[0], data[1], qae_cost_function, config, allocated_epochs)
    elif type.lower() == 'qte':
        trained_params, cost_history, validation_cost, _, _, _ = \
            train_adam(data[0], data[1], qte_cost_function, config, allocated_epochs)
    else:
        raise Exception('Unknown type: ' + type)
    if allocated_epochs >= 10:
        overfit_cost = abs(validation_cost - cost_history[-1]) / max(validation_cost, cost_history[-1])
        if overfit_cost > 1./3.: # throw away any configs that lead to obvious overfitting
            return float('inf')
    # scale the cost by the % of max num of blocks used
    # TODO: incorporate bottleneck size
    scale = float(config['num_blocks']) / float(MAX_NUM_BLOCKS)
    return cost_history[-1] * scale

def hyperband_search(data, type, max_training_epochs=100, reduction_factor=3):
    """
    Parameters:
      - type: "QAE" or "QTE" (case-insensitive)
      - max_training_epochs: maximum number of epochs allocated to any configuration
      - reduction_factor: factor by which successive configuration evals are reduced each round (eta)

    Returns:
      optimal_config, optimal_loss
    """
    print(f'Performing hyperband search over {type.upper()} training hyperparameters')
    max_bracket = int(np.floor(np.log(max_training_epochs) / np.log(reduction_factor)))
    total_budget = (max_bracket + 1) * max_training_epochs
    optimal_config = None
    optimal_loss = float('inf')

    print('num series in training:', len(data[0]))
    num_qubits = len(data[0][0])

    for bracket in reversed(range(max_bracket + 1)):
        initial_num_configs = int(np.ceil(total_budget / max_training_epochs / (bracket + 1) * reduction_factor ** bracket))
        initial_allocated_epochs = max_training_epochs * reduction_factor ** (-bracket)

        print(f"Bracket {max_bracket-bracket}: Starting with {initial_num_configs} configurations, each with {initial_allocated_epochs} epochs.")
        configs = [sample_hyperparameters(num_qubits) for _ in range(initial_num_configs)]
        print('Created configs')

        # successive reduction in num configs
        for round_index in range(bracket + 1):
            num_configs_this_round = int(np.floor(initial_num_configs * reduction_factor ** (-round_index)))
            epochs_this_round = int(initial_allocated_epochs * reduction_factor ** (round_index))

            print('Round', round_index, '-', num_configs_this_round, 'configs for', epochs_this_round, 'this round')
            round_losses = [get_loss(data, type, config, epochs_this_round) for config in configs]
            print(f"  Round {round_index}: {epochs_this_round} epochs; best loss = {min(round_losses):.4f}")

            for i, loss in enumerate(round_losses):
                if loss < optimal_loss:
                    optimal_loss = loss
                    optimal_config = configs[i]

            best_indices = np.argsort(round_losses)[:num_configs_this_round]
            configs = [configs[i] for i in best_indices]

    return optimal_config, optimal_loss

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimize the hyperparameters of the QAE or QTE training for this experiment."
    )
    parser.add_argument("data_directory", type=str, help="Path to the directory containing the training data.")
    parser.add_argument("--type", type=str, default='qte', help="QAE or QTE (case-insensitive)")
    parser.add_argument("--reduction_factor", type=int, default=3, help="Factor by which successive configuration evals are reduced each round.")
    parser.add_argument("--max_training_epochs", type=int, default=27, help="Maximum number of epochs allocated to any configuration.")
    args = parser.parse_args()

    from data_importers import import_generated
    dataset_partitions = import_generated(args.data_directory)
    input_data = [[], []]
    for _, (training, validation) in sorted(dataset_partitions.items()):
        # take one training and one validation series from each dataset
        t = np.random.randint(0, len(training))
        input_data[0].append(training[t])
        v = np.random.randint(0, len(validation))
        input_data[1].append(validation[v])

    best_config, best_loss = hyperband_search(input_data, args.type, args.max_training_epochs, args.reduction_factor)
    print("\nBest hyperparameter configuration found:")
    print(best_config)
    print("With estimated loss:", best_loss)
