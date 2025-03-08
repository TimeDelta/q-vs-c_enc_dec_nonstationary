import numpy as np
import math
from common import ENTANGLEMENT_OPTIONS, ENTANGLEMENT_GATES, EMBEDDING_ROTATION_GATES
from qte_training import train_qte_adam
from qae_training import train_qae_adam

MAX_NUM_BLOCKS = 5 # per 1/2 of QTE

def sample_hyperparameters(num_qubits):
    return {
        'bottleneck_size': np.random.randint(1, num_qubits),
        'num_blocks': np.random.randint(1, MAX_NUM_BLOCKS+1), # per 1/2 of QTE
        'learning_rate': 10 ** np.random.uniform(-4, -1),
        'penalty_weight': 10 ** np.random.uniform(-4, 0),
        'entanglement_topology': np.random.choice(ENTANGLEMENT_OPTIONS),
        'entanglement_gate': np.random.choice(ENTANGLEMENT_GATES),
        'embedding_gate': np.random.choice(EMBEDDING_ROTATION_GATES),
    }

def get_training_loss(data, type, config, allocated_epochs):
    if type.lower() == 'qae':
        trained_params, cost_history = train_qae_adam(input_data, config, allocated_epochs)
    elif type.lower() == 'qte':
        trained_params, cost_history = train_qte_adam(input_data, config, allocated_epochs)
    else:
        raise Exception('Unknown type: ' + type)
    return 3./4 * cost_history[-1] + config['num_blocks'] / MAX_NUM_BLOCKS / 4.

def hyperband_search(data, type, max_training_epochs=100, reduction_factor=3):
    """
    Parameters:
      - type: "QAE" or "QTE" (case-insensitive)
      - max_training_epochs: maximum number of epochs allocated to any configuration
      - reduction_factor: factor by which successive configuration evals are reduced each round (eta)

    Returns:
      optimal_config, optimal_loss
    """
    max_bracket = int(np.floor(np.log(max_training_epochs) / np.log(reduction_factor)))
    total_budget = (max_bracket + 1) * max_training_epochs
    optimal_config = None
    optimal_loss = float('inf')

    num_qubits = len(data[0])

    for bracket in reversed(range(max_bracket + 1)):
        initial_num_configs = int(np.ceil(total_budget / max_training_epochs / (bracket + 1) * reduction_factor ** bracket))
        initial_allocated_epochs = max_training_epochs * reduction_factor ** (-bracket)

        print(f"Bracket {bracket}: Starting with {initial_num_configs} configurations, each with {initial_allocated_epochs} epochs.")
        configs = [sample_hyperparameters(num_qubits) for _ in range(initial_num_configs)]

        # successive reduction in num configs
        for round_index in range(bracket + 1):
            num_configs_this_round = int(np.floor(initial_num_configs * reduction_factor ** (-round_index)))
            epochs_this_round = int(initial_allocated_epochs * reduction_factor ** (round_index))

            round_losses = [get_training_loss(data, type, config, epochs_this_round) for config in configs]
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
    parser.add_argument("--type", type=str, default='qae', help="QAE or QTE (case-insensitive)")
    parser.add_argument("--reduction_factor", type=int, default=3, help="Factor by which successive configuration evals are reduced each round.")
    parser.add_argument("--max_training_epochs", type=int, default=100, help="Maximum number of epochs allocated to any configuration.")
    parser.add_argument("--dataset", type=str, defaul='FACED')
    args = parser.parse_args()
    # input_data = np.array([[[.5, 1., 1.5, 2.],[.8, .8, 1.3, 2.]], [[1,1,1,1],[2,3,1,4]])

    if args.dataset == 'FACED':
        from data_importers import import_FACED
        input_data = import_FACED(args.data_directory)
    else:
        raise Exception('Unknown dataset type: ' + args.dataset)

    best_config, best_loss = hyperband_search(input_data, args.type, args.max_training_epochs, args.reduction_factor)
    print("\nBest hyperparameter configuration found:")
    print(best_config)
    print("With estimated loss:", best_loss)
