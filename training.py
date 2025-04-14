import numpy as np
from qiskit.quantum_info import partial_trace, DensityMatrix

from loss import *
from models import *

def adam_update(params, gradients, moment1, moment2, t, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    moment1 = beta1 * moment1 + (1 - beta1) * gradients
    moment2 = beta2 * moment2 + (1 - beta2) * (gradients ** 2)
    bias_corrected_moment1 = moment1 / (1 - beta1 ** t)
    bias_corrected_moment2 = moment2 / (1 - beta2 ** t)
    new_params = params - lr * bias_corrected_moment1 / (np.sqrt(bias_corrected_moment2) + epsilon)
    return new_params, moment1, moment2

def train_adam(training_data, validation_data, cost_function, config, model, num_epochs=100):
    """
    Train the QTE by minimizing the cost function using ADAM. Note that the QTE will only enforce
    the bottleneck via the cost function. This is done in order to balance efficiency w/ added
    flexibility for which qubits get thrown away.

    Parameters:
      - cost_function: the function to use for calculating the cost

      - config: dict containing additional hyperparameters:
            num_blocks:              # [entanglement layer, rotation layer] repetitions per 1/2 of QTE
            entanglement_topology:   for all entanglement layers
            entanglement_gate:       options are ['CX', 'CZ', 'RZX']
            learning_rate:
            bottleneck_size:         number of qubits for the latent space
            penalty_weight:          weight for the bottleneck penalty term
      - num_epochs: number of training iterations

    Returns trained_model, cost_history, validation_costs
    """
    cost_history = []
    gradient_width = 1e-4

    learning_rate = float(config['learning_rate'])
    max_penalty_weight = float(config.get('max_penalty_weight', 1.0))

    param_values = np.random.uniform(-np.pi, np.pi, size=len(model.trainable_params))
    moment1 = np.zeros_like(param_values)
    moment2 = np.zeros_like(param_values)

    print('  created untrained model')

    previous_param_values = param_values.copy()
    penalty_weight = max_penalty_weight # for testing w/ num_epochs == 0
    for t in range(1, num_epochs + 1):
        penalty_weight = max_penalty_weight * t / num_epochs
        print(f'  Epoch {t} (trash penalty weight: {penalty_weight})')
        param_dict = {param: value for param, value in zip(model.trainable_params, param_values)}
        model.set_params(param_dict)

        print('    calculating initial cost')
        # avoid also calculating the loss for (current param - epsilon) and use single
        # initial cost evaluation for all parameters to speed up training
        initial_costs = cost_function(training_data, model, penalty_weight)
        cost_history.append(initial_costs)
        initial_cost = np.sum(initial_costs)
        print('     ', initial_cost)

        gradients = np.zeros_like(param_values)
        # progressively increase the probability that the model will have to deal with it's own noise from the previous time step
        for j in range(len(param_values)):
            print('    calculating gradient for param ' + str(j+1) + ' / ' + str(len(param_values)) + ' = ' + str((j)/len(param_values)*100) + '% done')
            params_eps = param_values.copy()
            params_eps[j] += gradient_width
            param_dict = {param: value for param, value in zip(model.trainable_params, params_eps)}
            model.set_params(param_dict)
            perturbed_costs = cost_function(training_data, model, penalty_weight)
            perturbed_cost = perturbed_costs[0]
            for cost in perturbed_costs:
                perturbed_cost += cost
            gradients[j] = (perturbed_cost - initial_cost) / gradient_width

        previous_param_values = param_values.copy()
        param_values, moment1, moment2 = adam_update(param_values, gradients, moment1, moment2, t, learning_rate)
        print(f'    Min param update: {np.min(param_values-previous_param_values)}')
        print(f'    Mean param update: {np.mean(param_values-previous_param_values)}')
        print(f'    Std dev param update: {np.std(param_values-previous_param_values)}')
        print(f'    Median param update: {np.median(param_values-previous_param_values)}')
        print(f'    Max param update: {np.max(param_values-previous_param_values)}')

    print('  calculating validation costs')
    param_dict = {param: value for param, value in zip(model.trainable_params, param_values)}
    model.set_params(param_dict)
    validation_costs = []
    for (i, series) in validation_data:
        series_costs = [i]
        for c in cost_function([(i, series)], model, penalty_weight):
            series_costs.append(c)
        validation_costs.append(series_costs)

    return model, cost_history, validation_costs

if __name__ == '__main__':
    from qiskit import qpy
    import os
    import argparse
    import matplotlib.pyplot as plt

    from data_importers import import_generated
    from analysis import *

    parser = argparse.ArgumentParser(
        description="Train both a quantum and a classical version of: autoregressive Encoder/Decoder and Auto-Encoder (w/ and w/o time step) over each relevant dataset."
    )
    parser.add_argument("data_directory", type=str, help="Path to the directory containing the generated data.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix to use for every saved file name in this run.")
    parser.add_argument("--type_filter", type=str, default=None, help="Only train model types that contain the provided string")
    parser.add_argument("--seed", type=int, default=89266583, help="Seed value to set before creation of each model.")
    args = parser.parse_args()

    run_prefix = args.prefix if args.prefix else ''
    model_types = [m for m in MODEL_TYPES if args.type_filter in m]
    dataset_partitions = import_generated(args.data_directory)
    num_epochs = 2

    def save(dataset_metrics, metric_desc):
        print(f'  {metric_desc}:')
        metric_desc = metric_desc.lower().replace(' ', '_')
        fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_{metric_desc}.npy')
        print('    Shape (number series, number of values[+1 for prepending series index]):', dataset_metrics.shape)
        np.save(fname, dataset_metrics)
        print('    Saved', fname)

    def save_trash_indices_histogram(trash_indices, num_features):
        fig, ax = plt.subplots()
        plt.hist(trash_indices, bins=range(num_features + 1), align='left')
        plt.xlabel('Trash Feature Index')
        plt.ylabel('Frequency')
        plt.title('Trash Feature Index Selection Histogram')
        ax.set_xticks(range(num_features))
        hist_save_path = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_trash_feature_histogram.png')
        plt.savefig(hist_save_path)
        print(f'Saved trash feature histogram to {hist_save_path}')

    for d_i, (training, validation) in sorted(dataset_partitions.items()):
        num_features = len(training[0][1][0])
        bottleneck_size = num_features // 2
        config = {
            'bottleneck_size': bottleneck_size,
            'num_blocks': 1,
            'learning_rate': 0.08,
            'max_penalty_weight': 2.0,
            'entanglement_topology': 'full',
            'entanglement_gate': 'cz',
            'embedding_gate': 'rz',
        }
        for model_type in model_types:
            np.random.seed(args.seed)
            print('Training ' + model_type.upper() + ' for dataset ' + str(d_i))
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
                train_adam(training, validation, loss_fn, config, model, num_epochs)

            print('  Training cost history:', cost_history)
            cost_history = np.array(cost_history)
            fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_cost_history.npy')
            np.save(fname, cost_history)
            print('  Saved cost history')
            print('  Validation cost per series:', validation_costs)
            print(validation_costs)
            validation_costs = np.array(validation_costs)
            fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_validation_costs.npy')
            np.save(fname, validation_costs)
            print('  Saved validation cost per series')

            if num_epochs > 0: # avoid index out of range while testing
                mean_training_costs = cost_history[-1] / len(training)
                mean_validation_costs = np.sum(validation_costs[:,1:], axis=0) / len(validation)
                check_for_overfitting(mean_training_costs, mean_validation_costs)

            # === Model metric computations ===
            all_trash_indices = []
            dataset_enc_differential_entropies = []
            dataset_bottleneck_lzcs = []
            dataset_bottleneck_hes = []
            dataset_bottleneck_hfds = []
            if model_type.startswith('q'):
                def extract_marginal_features(bottlenecks):
                    # One feature per qubit: the marginal probability of |0>
                    features = []
                    for dm in bottlenecks:
                        num_qubits = dm.num_qubits
                        qubit_features = []
                        for i in range(num_qubits):
                            trace_indices = list(range(num_qubits))
                            trace_indices.remove(i)
                            reduced_dm = partial_trace(dm, trace_indices)
                            p0 = np.real(reduced_dm.data[0, 0])
                            qubit_features.append(p0)
                        features.append(qubit_features)
                    return np.array(features)

                def extract_bloch_z_features(bottlenecks):
                    # One feature per qubit: the expectation value of the Pauli Z operator.
                    # For a single-qubit density matrix ρ, <Z> = Tr(ρ * Z)
                    Z = np.array([[1, 0], [0, -1]])
                    features = []
                    for dm in bottlenecks:
                        num_qubits = dm.num_qubits
                        qubit_features = []
                        for i in range(num_qubits):
                            trace_indices = list(range(num_qubits))
                            trace_indices.remove(i)
                            reduced_dm = partial_trace(dm, trace_indices)
                            exp_z = np.real(np.trace(reduced_dm.data @ Z))
                            qubit_features.append(exp_z)
                        features.append(qubit_features)
                    return np.array(features)
                dataset_enc_entangle_entropies = []
                dataset_enc_vn_entropies = []
                for (s_i, series) in validation:
                    enc_entangle_entropies = []
                    enc_vn_entropies = []
                    bottlenecks = []

                    num_examples = len(series)
                    if autoregressive:
                        num_examples -= 1
                    for t in range(num_examples):
                        input_state = model.prepare_state(series[t])
                        initial_dm = DensityMatrix(input_state)
                        bottleneck_dm, _ = model.forward(initial_dm)
                        bottlenecks.append(bottleneck_dm)

                        all_trash_indices.extend(model.get_trash_indices(bottleneck_dm))

                        enc_entangle_entropies.append(entanglement_entropy(bottleneck_dm))
                        enc_vn_entropies.append(von_neumann_entropy(bottleneck_dm))

                    dataset_enc_entangle_entropies.append(np.concatenate(([s_i], enc_entangle_entropies)))
                    dataset_enc_vn_entropies.append(np.concatenate(([s_i], enc_vn_entropies)))

                    bottlenecks_features = extract_marginal_features(bottlenecks)

                    diff_entropies = multimodal_differential_entropy_per_feature(bottlenecks_features)
                    dataset_enc_differential_entropies.append([s_i, np.sum(diff_entropies)])

                    dataset_bottleneck_lzcs.append([[s_i, lempel_ziv_complexity_continuous(bottlenecks_features)]])
                    dataset_bottleneck_hes.append(np.concatenate(([s_i], hurst_exponent(bottlenecks_features))))
                    dataset_bottleneck_hfds.append(np.concatenate(([s_i], higuchi_fractal_dimension(bottlenecks_features))))

                save(np.array(dataset_enc_entangle_entropies), 'Bottleneck entanglement entropy')
                save(np.array(dataset_enc_vn_entropies), 'Bottleneck full VN entropy')

                fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_trained_model.qpy')
                with open(fname, 'wb') as file:
                    qpy.dump(trained_model.full_circuit, file)
            elif model_type.startswith('c'):
                all_trash_indices = []
                for (s_i, series) in validation:
                    enc_differential_entropies = []
                    num_examples = len(series)
                    if autoregressive:
                        num_examples -= 1
                    bottlenecks = []
                    for t in range(num_examples):
                        input_state = model.prepare_state(series[t])
                        bottleneck_state, _ = model.forward(input_state)
                        bottlenecks.append(bottleneck_state)

                        all_trash_indices.extend(model.get_trash_indices(bottleneck_state))
                    bottlenecks = np.array(bottlenecks)
                    diff_entropies = multimodal_differential_entropy_per_feature(bottlenecks)
                    dataset_enc_differential_entropies.append([s_i, np.sum(diff_entropies)])
                    dataset_bottleneck_lzcs.append([[s_i, lempel_ziv_complexity_continuous(bottlenecks)]])
                    dataset_bottleneck_hes.append(np.concatenate(([s_i], hurst_exponent(bottlenecks))))
                    dataset_bottleneck_hfds.append(np.concatenate(([s_i], higuchi_fractal_dimension(bottlenecks))))

                fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_trained_model.pth')
                torch.save(model, fname)

            save(np.array(dataset_bottleneck_lzcs), 'Bottleneck LZC')
            save(np.array(dataset_bottleneck_hes), 'Bottleneck HE')
            save(np.array(dataset_bottleneck_hfds), 'Bottleneck HFD')
            save(np.array(dataset_enc_differential_entropies), 'Bottleneck Differential Entropy')
            print(f"Saved trained model to {fname}")
            save_trash_indices_histogram(all_trash_indices, model.num_features)
