"""Temporal descriptors and training-only linear probes.

These descriptors are finite-sample summaries, not MDL codelengths or proofs of
information preservation. Quantum descriptors use retained Z readouts only.
"""

import numpy as np
import antropy


def temporal_descriptors(sequence):
    sequence = np.asarray(sequence, dtype=float)
    if sequence.ndim != 2 or len(sequence) < 2 or not np.all(np.isfinite(sequence)):
        raise ValueError('Expected a finite time-by-feature sequence with at least two samples')
    permutation_entropies = []
    binary_complexities = []
    lag_correlations = []
    for feature in sequence.T:
        feature_entropies = []
        for scale in (1, 2, 4):
            coarse_length = len(feature) // scale
            if coarse_length - 2 < 30:
                continue
            coarse_feature = feature[:coarse_length * scale].reshape(coarse_length, scale).mean(axis=1)
            entropy = antropy.perm_entropy(coarse_feature, order=3, delay=1, normalize=True)
            feature_entropies.append(float(entropy))
        permutation_entropies.append(float(np.mean(feature_entropies)) if feature_entropies else np.nan)
        binary_symbols = (feature > np.median(feature)).astype(np.uint8)
        phrase_count = antropy.lziv_complexity(binary_symbols, normalize=False)
        binary_complexities.append(float(phrase_count * np.log2(len(feature)) / len(feature)))
        if np.std(feature[:-1]) < 1e-12 or np.std(feature[1:]) < 1e-12:
            lag_correlations.append(0.0)
        else:
            lag_correlations.append(float(np.corrcoef(feature[:-1], feature[1:])[0, 1]))
    return dict(permutation_entropy=float(np.mean(permutation_entropies)),
                binary_lz76=float(np.mean(binary_complexities)),
                lag1_correlation=float(np.mean(lag_correlations)))


def descriptor_distance(first_descriptors, second_descriptors):
    """Exploratory squared distance; lag correlation is rescaled to [0, 1]."""
    differences = [first_descriptors[key] - second_descriptors[key] for key in first_descriptors]
    differences[-1] /= 2
    return float(np.mean(np.square(differences)))


def fit_ridge_probe(latent_inputs, targets, ridge_penalty):
    input_mean = latent_inputs.mean(axis=0)
    target_mean = targets.mean(axis=0)
    centered_inputs = latent_inputs - input_mean
    centered_targets = targets - target_mean
    covariance = centered_inputs.T @ centered_inputs
    coefficient_matrix = np.linalg.solve(
        covariance + ridge_penalty * len(latent_inputs) * np.eye(covariance.shape[0]),
        centered_inputs.T @ centered_targets,
    )
    intercept = target_mean - input_mean @ coefficient_matrix
    return coefficient_matrix, intercept


def predict_probe(latent_inputs, probe):
    return latent_inputs @ probe[0] + probe[1]


def choose_probe(train_latents, train_targets, validation_latents, validation_targets, penalties):
    candidates = []
    for penalty in penalties:
        probe = fit_ridge_probe(train_latents, train_targets, penalty)
        validation_mse = float(np.mean((predict_probe(validation_latents, probe) - validation_targets)**2))
        candidates.append((validation_mse, penalty, probe))
    best_validation_mse, best_penalty, best_probe = min(candidates, key=lambda candidate: candidate[0])
    return best_probe, dict(ridge_penalty=best_penalty, validation_mse=best_validation_mse)


def transformation_controls(latent_sequence, probe, random_generator):
    latent_dimension = latent_sequence.shape[1]
    rotation_matrix, _ = np.linalg.qr(random_generator.normal(size=(latent_dimension, latent_dimension)))
    scaled_sequence = 3 * latent_sequence
    rotated_sequence = latent_sequence @ rotation_matrix
    equivalent_probe = (rotation_matrix.T @ probe[0], probe[1])
    original_predictions = predict_probe(latent_sequence, probe)
    equivalent_predictions = predict_probe(rotated_sequence, equivalent_probe)
    original_descriptors = temporal_descriptors(latent_sequence)
    shuffled_sequence = latent_sequence[random_generator.permutation(len(latent_sequence))]
    return dict(
        scale_descriptor_change=descriptor_distance(original_descriptors, temporal_descriptors(scaled_sequence)),
        rotation_descriptor_change=descriptor_distance(original_descriptors, temporal_descriptors(rotated_sequence)),
        shuffled_descriptor_change=descriptor_distance(original_descriptors, temporal_descriptors(shuffled_sequence)),
        rotation_prediction_max_difference=float(np.max(np.abs(original_predictions - equivalent_predictions))),
    )
