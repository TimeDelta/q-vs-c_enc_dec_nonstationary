import os
import json

import numpy as np

from models import *
from data_importers import import_generated
from analysis import MODEL_TYPES

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

datasets_dir = './generated_datasets'
dataset_partitions = import_generated(datasets_dir)
num_features = len(next(iter(dataset_partitions.values()))[0][0][1][0])
best_config_path = os.path.join(datasets_dir, 'best_config.json')
if os.path.exists(best_config_path):
    with open(best_config_path, 'r') as file:
        config = json.load(file)
    print('Loaded best config from hyperparameter optimization')
else:
    config = {
        'bottleneck_size': num_features // 2,
        'num_blocks': 1,
        'learning_rate': 0.021450664374153845,
        'max_penalty_weight': 2.0,
        'entanglement_topology': 'circular',
        'entanglement_gate': 'cz',
        'embedding_gate': 'rz',
        'block_gate': 'rz',
    }
    # raise Exception('Need config')
for d_i, (_, validation) in dataset_partitions.items():
    for model_type in MODEL_TYPES:
        is_recurrent = 'r' in model_type
        autoregressive = 'te' in model_type # transition encoder
        if model_type.startswith('c'):
            model = ClassicalEncoderDecoder(num_features, config, is_recurrent)
        elif model_type.startswith('q'):
            model = QuantumEncoderDecoder(num_features, config, is_recurrent)
        else:
            raise Exception('Unexpected model type: ' + model_type)
        try:
            model.load(f'{datasets_dir}/dataset{d_i}_{model_type}_trained_model')
            print(f'Loaded {model_type} for dataset {d_i}')
        except Exception as e:
            print(f'Failed to load {model_type} model for dataset {d_i}: {e}')
            continue
        all_bottlenecks = []
        all_z_bottlenecks = []
        all_marginal_bottlenecks = []
        for (s_i, series) in validation:
            bottlenecks = []
            for state in series:
                bottleneck, prediction = model.forward(model.prepare_state(state))
                bottlenecks.append(bottleneck)
            if model_type.startswith('q'):
                all_marginal_bottlenecks.append(np.concatenate(([[s_i for _ in range(num_features)]], extract_marginal_features(bottlenecks))))
                all_z_bottlenecks.append(np.concatenate(([[s_i for _ in range(num_features)]], extract_bloch_z_features(bottlenecks))))
            else:
                all_bottlenecks.append(np.concatenate(([[s_i for _ in range(num_features)]], bottlenecks)))
        if model_type.startswith('q'):
            fname = os.path.join(datasets_dir, f'dataset{d_i}_{model_type}_marginal_bottlenecks.npy')
            np.save(fname, np.array(all_marginal_bottlenecks))
            fname = os.path.join(datasets_dir, f'dataset{d_i}_{model_type}_z_bottlenecks.npy')
            np.save(fname, np.array(all_z_bottlenecks))
        else:
            fname = os.path.join(datasets_dir, f'dataset{d_i}_{model_type}_bottlenecks.npy')
            np.save(fname, np.array(all_bottlenecks))

