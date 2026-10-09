"""Controlled bounded observations with independently varied regime factors."""

from dataclasses import asdict, dataclass, replace
import hashlib

import numpy as np


@dataclass(frozen=True)
class RegimeProfile:
    mean_shift: float = 0.15
    state_standard_deviation: float = 0.35
    variance_ratio: float = 1.5
    persistence_low: float = 0.35
    persistence_high: float = 0.75
    observation_noise: float = 0.04


def generate_sequences(sequence_count, sequence_length, mixing_matrix, profile, random_generator):
    latent_dimension = mixing_matrix.shape[1]
    feature_count = mixing_matrix.shape[0]
    boundaries = np.array([0, sequence_length // 3, 2 * sequence_length // 3, sequence_length])
    observations = np.empty((sequence_count, sequence_length, feature_count))
    latent_states = np.empty((sequence_count, sequence_length, latent_dimension))
    for sequence_index in range(sequence_count):
        previous_state = random_generator.normal(0, profile.state_standard_deviation, latent_dimension)
        for timestep in range(sequence_length):
            regime_index = int(np.searchsorted(boundaries[1:-1], timestep, side='right'))
            sign = (-1) ** regime_index
            state_mean = sign * profile.mean_shift * np.linspace(0.5, 1.0, latent_dimension)
            state_sd = profile.state_standard_deviation * (profile.variance_ratio if regime_index == 1 else 1)
            persistence = profile.persistence_high if regime_index == 1 else profile.persistence_low
            # This scaling holds the stationary variance fixed when persistence varies.
            innovation = np.sqrt(1 - persistence**2) * state_sd * random_generator.normal(size=latent_dimension)
            current_state = persistence * previous_state + (1 - persistence) * state_mean + innovation
            measurement_noise = profile.observation_noise * random_generator.normal(size=feature_count)
            observations[sequence_index, timestep] = np.tanh(mixing_matrix @ current_state + measurement_noise)
            latent_states[sequence_index, timestep] = current_state
            previous_state = current_state
    return observations, latent_states, boundaries[1:-1]


def partition_hash(observations):
    return hashlib.sha256(np.ascontiguousarray(observations).tobytes()).hexdigest()


def make_dataset(config, data_seed, *, include_test=True):
    if config['sequence_length'] < 32:
        raise ValueError('sequence_length must be at least 32')
    feature_count = config['num_features']
    latent_dimension = config['generating_latent_dimension']
    if not 1 <= latent_dimension <= feature_count:
        raise ValueError('Invalid generating latent dimension')
    mixing_generator = np.random.default_rng(np.random.SeedSequence([data_seed, 0]))
    mixing_matrix, _ = np.linalg.qr(mixing_generator.normal(size=(feature_count, latent_dimension)))
    base_profile = RegimeProfile()
    profiles = {
        'train': base_profile,
        'validation': base_profile,
        'test_id': base_profile,
        'test_mean_shift': replace(base_profile, mean_shift=0.8),
        'test_variance_shift': replace(base_profile, variance_ratio=3.0),
        'test_persistence_shift': replace(base_profile, persistence_low=-0.35, persistence_high=0.95),
        'test_noise_shift': replace(base_profile, observation_noise=0.25),
        'test_combined_shift': replace(base_profile, mean_shift=0.8, variance_ratio=3.0,
                                       persistence_low=-0.35, persistence_high=0.95, observation_noise=0.25),
    }
    if not include_test:
        profiles = {key: value for key, value in profiles.items() if key in ('train', 'validation')}
    partitions = {}
    metadata = {}
    for split_index, (partition_name, profile) in enumerate(profiles.items(), start=1):
        split_generator = np.random.default_rng(np.random.SeedSequence([data_seed, split_index]))
        sequence_count = config['train_sequences'] if partition_name == 'train' else (
            config['validation_sequences'] if partition_name == 'validation' else config['test_sequences'])
        if sequence_count < 1:
            raise ValueError('Every partition needs at least one independent sequence')
        observations, latent_states, changepoints = generate_sequences(
            sequence_count, config['sequence_length'], mixing_matrix, profile, split_generator
        )
        partitions[partition_name] = observations
        metadata[partition_name] = {
            'profile': asdict(profile), 'sha256': partition_hash(observations),
            'sequence_count': sequence_count, 'changepoints': changepoints.tolist(),
            'seed_components': [data_seed, split_index],
        }
        partitions[partition_name + '_generating_states'] = latent_states
    return partitions, dict(partitions=metadata, mixing_matrix=mixing_matrix.tolist())
