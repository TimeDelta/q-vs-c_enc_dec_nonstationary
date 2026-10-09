"""Models and full-budget, validation-only selection for the CPU pilot."""

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from models import ClassicalEncoderDecoder, QuantumEncoderDecoder
from optimizers import adam_update
from pilot.initialization import feature_neutral_parameters


def aligned_targets(observations, objective):
    return observations[:, 1:] if objective == 'forecast' else observations[:, :-1]


class LegacyEncoder:
    def __init__(self, name, config, initialization_seed):
        self.name = name
        self.objective = 'forecast' if name.split('_')[0].endswith('te') else 'reconstruction'
        self.quantum = name.startswith('q')
        self.learning_rate = config['learning_rate']
        self.config = dict(config)
        self.config['feature_encoding'] = 'bounded_ry'
        self.config['enforce_bottleneck'] = True
        if name.endswith('_noent'):
            self.config['entanglement_topology'] = 'none'
        base_model_name = name.split('_')[0]
        recurrent = 'r' in base_model_name
        torch.manual_seed(initialization_seed)
        if self.quantum:
            self.model = QuantumEncoderDecoder(config['num_features'], self.config, recurrent)
        else:
            self.model = ClassicalEncoderDecoder(config['num_features'], self.config, recurrent).double()
        self.initialization_seed = initialization_seed
        self.initialization_mode = config.get('quantum_initialization', 'near_zero') if self.quantum else 'near_zero'
        if self.initialization_mode not in ('near_zero', 'feature_neutral'):
            raise ValueError('Unknown quantum initialization mode')
        self.initialization_metadata = {}
        self.parameter_handles = self.model.trainable_params
        generator = np.random.default_rng(initialization_seed)
        self.parameter_values = generator.uniform(-0.15, 0.15, len(self.parameter_handles))
        if recurrent:
            self.parameter_values[0] = 0.0
        if self.initialization_mode == 'feature_neutral':
            self.parameter_values, self.initialization_metadata = feature_neutral_parameters(
                self.model, self.parameter_handles, self.parameter_values)
        self.set_parameter_values(self.parameter_values)
        self.prepared_sequences = {}
        self.state_preparations = 0
        self.loss_evaluations = 0
        self.parameter_count = len(self.parameter_handles)
        self.latent_dimension = config['bottleneck_size']
        self.initial_parameter_values = self.parameter_values.copy()

    def set_parameter_values(self, parameter_values):
        self.parameter_values = np.asarray(parameter_values).copy()
        self.model.set_params(dict(zip(self.parameter_handles, self.parameter_values)))

    def snapshot(self):
        if self.quantum:
            return self.parameter_values.copy()
        return copy.deepcopy(self.model.state_dict())

    def restore(self, snapshot):
        if self.quantum:
            self.set_parameter_values(snapshot)
        else:
            self.model.load_state_dict(snapshot)
        self.model.reset_hidden_state()

    def sequence_outputs(self, sequence):
        self.model.reset_hidden_state()
        if self.quantum:
            sequence_key = hashlib.sha256(sequence.tobytes()).hexdigest()
            if sequence_key not in self.prepared_sequences:
                self.prepared_sequences[sequence_key] = [self.model.prepare_state(features) for features in sequence]
                self.state_preparations += len(sequence)
            inputs = self.prepared_sequences[sequence_key]
        else:
            inputs = [self.model.prepare_state(features) for features in sequence]
        latent_features = []
        predictions = []
        with torch.no_grad():
            for input_state in inputs:
                bottleneck_state, predicted_state = self.model.forward(input_state)
                latent_features.append(self.model.latent_features(bottleneck_state))
                predictions.append(self.model.collapse_state(predicted_state) if self.quantum else predicted_state)
        return np.asarray(latent_features), np.asarray(predictions)

    def native_mse(self, observations):
        self.loss_evaluations += 1
        predictions = np.stack([self.sequence_outputs(sequence)[1][:-1] for sequence in observations])
        return float(np.mean((predictions - aligned_targets(observations, self.objective))**2))

    def torch_loss(self, observations):
        sequence_losses = []
        for sequence in observations:
            self.model.reset_hidden_state()
            outputs = [self.model.forward_tensor(self.model.prepare_state(features))[1] for features in sequence[:-1]]
            predictions = torch.stack(outputs)
            target_features = sequence[1:] if self.objective == 'forecast' else sequence[:-1]
            targets = torch.as_tensor(target_features, dtype=torch.float64)
            sequence_losses.append(torch.mean((predictions - targets)**2))
        return torch.stack(sequence_losses).mean()

    def train(self, training, validation, epochs, gradient_width):
        history = []
        best_validation_mse = np.inf
        best_snapshot = None
        if self.quantum:
            first_moment = np.zeros(self.parameter_count)
            second_moment = np.zeros(self.parameter_count)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(1, epochs + 1):
            if self.quantum:
                current_parameters = self.parameter_values.copy()
                current_loss = self.native_mse(training)
                gradients = np.empty(self.parameter_count)
                for parameter_index in range(self.parameter_count):
                    perturbed_parameters = current_parameters.copy()
                    perturbed_parameters[parameter_index] += gradient_width
                    self.set_parameter_values(perturbed_parameters)
                    gradients[parameter_index] = (self.native_mse(training) - current_loss) / gradient_width
                updated_parameters, first_moment, second_moment = adam_update(
                    current_parameters, gradients, first_moment, second_moment, epoch, self.learning_rate
                )
                self.set_parameter_values(updated_parameters)
                gradient_norm = float(np.linalg.norm(gradients))
            else:
                optimizer.zero_grad()
                loss = self.torch_loss(training)
                loss.backward()
                gradient_norm = float(torch.sqrt(sum((parameter.grad**2).sum() for parameter in self.model.parameters())).item())
                optimizer.step()
            training_mse = self.native_mse(training)
            validation_mse = self.native_mse(validation)
            if not np.isfinite(training_mse + validation_mse + gradient_norm):
                raise FloatingPointError('Nonfinite training diagnostics for ' + self.name)
            history.append(dict(epoch=epoch, train_mse=training_mse,
                                validation_mse=validation_mse, gradient_norm=gradient_norm))
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_snapshot = self.snapshot()
        self.restore(best_snapshot)
        return history, best_validation_mse

    def save(self, checkpoint_prefix):
        if self.quantum:
            self.model.save(str(checkpoint_prefix))
        else:
            torch.save(dict(protocol_version=2, model_name=self.name, config=self.config,
                            state_dict=self.model.state_dict()), str(checkpoint_prefix) + '.pt')


class EncoderNetwork(nn.Module):
    def __init__(self, feature_count, latent_dimension, hidden_width, recurrent):
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.encoder = nn.GRU(feature_count, latent_dimension, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Linear(feature_count, hidden_width), nn.Tanh(),
                                         nn.Linear(hidden_width, latent_dimension), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(latent_dimension, hidden_width), nn.Tanh(),
                                     nn.Linear(hidden_width, feature_count), nn.Tanh())

    def forward(self, observations):
        latent_features = self.encoder(observations)[0] if self.recurrent else self.encoder(observations)
        return latent_features, self.decoder(latent_features)


class NeuralEncoder:
    def __init__(self, name, config, initialization_seed):
        self.name = name
        self.config = config
        self.objective = 'forecast' if name.endswith('te') else 'reconstruction'
        self.quantum = False
        self.learning_rate = config['learning_rate']
        self.latent_dimension = config['bottleneck_size']
        torch.manual_seed(initialization_seed)
        self.network = EncoderNetwork(config['num_features'], self.latent_dimension,
                                      config['hidden_width'], recurrent=name.startswith('gru')).double()
        self.parameter_count = sum(parameter.numel() for parameter in self.network.parameters())
        self.loss_evaluations = 0
        self.state_preparations = 0

    def sequence_outputs(self, sequence):
        self.network.eval()
        with torch.no_grad():
            latent_features, predictions = self.network(torch.as_tensor(sequence[None], dtype=torch.float64))
        return latent_features[0].numpy(), predictions[0].numpy()

    def native_mse(self, observations):
        self.loss_evaluations += 1
        self.network.eval()
        with torch.no_grad():
            _, predictions = self.network(torch.as_tensor(observations[:, :-1], dtype=torch.float64))
            targets = torch.as_tensor(aligned_targets(observations, self.objective), dtype=torch.float64)
            return float(torch.mean((predictions - targets)**2).item())

    def train(self, training, validation, epochs, gradient_width):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        training_inputs = torch.as_tensor(training[:, :-1], dtype=torch.float64)
        training_targets = torch.as_tensor(aligned_targets(training, self.objective), dtype=torch.float64)
        best_snapshot = None
        best_validation_mse = np.inf
        history = []
        for epoch in range(1, epochs + 1):
            self.network.train()
            optimizer.zero_grad()
            _, predictions = self.network(training_inputs)
            loss = torch.mean((predictions - training_targets)**2)
            loss.backward()
            gradient_norm = float(torch.sqrt(sum((parameter.grad**2).sum() for parameter in self.network.parameters())).item())
            optimizer.step()
            training_mse = self.native_mse(training)
            validation_mse = self.native_mse(validation)
            if not np.isfinite(training_mse + validation_mse + gradient_norm):
                raise FloatingPointError('Nonfinite training diagnostics for ' + self.name)
            history.append(dict(epoch=epoch, train_mse=training_mse,
                                validation_mse=validation_mse, gradient_norm=gradient_norm))
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_snapshot = copy.deepcopy(self.network.state_dict())
        self.network.load_state_dict(best_snapshot)
        return history, best_validation_mse

    def save(self, checkpoint_prefix):
        torch.save(dict(protocol_version=2, model_name=self.name, config=self.config,
                        state_dict=self.network.state_dict()), str(checkpoint_prefix) + '.pt')


class LinearEncoder:
    def __init__(self, name, config, initialization_seed, ridge_penalty=0.001):
        self.name = name
        self.objective = 'forecast' if name in ('reduced_rank', 'persistence') else 'reconstruction'
        self.quantum = False
        self.latent_dimension = config['num_features'] if name == 'persistence' else config['bottleneck_size']
        self.initialization_seed = initialization_seed
        self.ridge_penalty = ridge_penalty
        self.parameter_count = 0
        self.loss_evaluations = 0
        self.state_preparations = 0

    def fit(self, observations):
        inputs = observations[:, :-1].reshape(-1, observations.shape[-1])
        targets = observations[:, 1:].reshape(-1, observations.shape[-1])
        self.input_mean = inputs.mean(axis=0)
        centered_inputs = inputs - self.input_mean
        if self.name == 'persistence':
            self.input_mean = np.zeros(inputs.shape[1])
            self.encoder_matrix = np.eye(inputs.shape[1])
            self.decoder_matrix = np.eye(inputs.shape[1])
            self.target_mean = np.zeros(inputs.shape[1])
        elif self.name == 'pca':
            _, _, right_vectors = np.linalg.svd(centered_inputs, full_matrices=False)
            self.encoder_matrix = right_vectors[:self.latent_dimension].T
            self.decoder_matrix = self.encoder_matrix.T
            self.target_mean = self.input_mean
        elif self.name == 'random_linear':
            generator = np.random.default_rng(self.initialization_seed)
            random_matrix, _ = np.linalg.qr(generator.normal(size=(inputs.shape[1], self.latent_dimension)))
            self.encoder_matrix = random_matrix
            self.decoder_matrix = random_matrix.T
            self.target_mean = self.input_mean
        else:
            self.target_mean = targets.mean(axis=0)
            covariance = centered_inputs.T @ centered_inputs / len(inputs)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance + self.ridge_penalty * np.eye(inputs.shape[1]))
            inverse_square_root = (eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T
            cross_covariance = centered_inputs.T @ (targets - self.target_mean) / len(inputs)
            left_vectors, singular_values, right_vectors = np.linalg.svd(inverse_square_root @ cross_covariance, full_matrices=False)
            self.encoder_matrix = inverse_square_root @ left_vectors[:, :self.latent_dimension]
            self.decoder_matrix = singular_values[:self.latent_dimension, None] * right_vectors[:self.latent_dimension]

    def sequence_outputs(self, sequence):
        latent_features = (sequence - self.input_mean) @ self.encoder_matrix
        return latent_features, latent_features @ self.decoder_matrix + self.target_mean

    def native_mse(self, observations):
        self.loss_evaluations += 1
        predictions = np.stack([self.sequence_outputs(sequence)[1][:-1] for sequence in observations])
        return float(np.mean((predictions - aligned_targets(observations, self.objective))**2))

    def save(self, checkpoint_prefix):
        np.savez(str(checkpoint_prefix) + '.npz', input_mean=self.input_mean,
                 target_mean=self.target_mean, encoder_matrix=self.encoder_matrix,
                 decoder_matrix=self.decoder_matrix, ridge_penalty=self.ridge_penalty)


def build_model(name, config, initialization_seed, ridge_penalty=0.001):
    if name in ('pca', 'reduced_rank', 'random_linear', 'persistence'):
        return LinearEncoder(name, config, initialization_seed, ridge_penalty)
    if name.startswith(('mlp', 'gru')):
        return NeuralEncoder(name, config, initialization_seed)
    return LegacyEncoder(name, config, initialization_seed)


def fit_selected_model(name, config, training, validation, initialization_seed):
    """Only training and validation arrays are accepted; test data is unavailable here."""
    candidate_records = []
    selected_model = None
    selected_validation_mse = np.inf
    if name == 'reduced_rank':
        candidates = config['native_ridge_penalties']
    elif name in ('pca', 'random_linear', 'persistence'):
        candidates = [0.001]
    else:
        candidates = config['learning_rates']
    started_at = time.perf_counter()
    for candidate_index, candidate_value in enumerate(candidates):
        candidate_config = dict(config, learning_rate=float(candidate_value))
        candidate_model = build_model(name, candidate_config, initialization_seed, candidate_value)
        initial_training_mse = candidate_model.native_mse(training) if not isinstance(candidate_model, LinearEncoder) else None
        if isinstance(candidate_model, LinearEncoder):
            candidate_model.fit(training)
            validation_mse = candidate_model.native_mse(validation)
            history = []
        else:
            history, validation_mse = candidate_model.train(
                training, validation, config['epochs'], config['gradient_width']
            )
        candidate_records.append(dict(candidate_index=candidate_index, candidate_value=candidate_value,
                                      initial_training_mse=initial_training_mse,
                                      best_validation_mse=validation_mse, history=history,
                                      loss_evaluations=candidate_model.loss_evaluations,
                                      state_preparations=candidate_model.state_preparations))
        if validation_mse < selected_validation_mse:
            selected_model = candidate_model
            selected_validation_mse = validation_mse
    return selected_model, dict(candidates=candidate_records, selected_validation_mse=selected_validation_mse,
                                fit_seconds=time.perf_counter() - started_at,
                                parameter_count=selected_model.parameter_count,
                                initialization_mode=getattr(selected_model, 'initialization_mode', 'default'),
                                initialization_metadata=getattr(selected_model, 'initialization_metadata', {}),
                                optimizer='closed_form' if isinstance(selected_model, LinearEncoder) else (
                                    'adam_forward_difference' if selected_model.quantum else 'adam_autograd'))
