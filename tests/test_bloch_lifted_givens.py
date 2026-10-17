"""Checks for the bounded-RY lifted Givens counterpart of the quantum encoder/decoder."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models import BlochLiftedGivensEncoderDecoder, bounded_ry_bloch_vectors
from pilot.descriptor_study import restore_model
from pilot.models import build_model, fit_selected_model
from pilot.run import MODEL_NAMES, validate_config


class BlochLiftedGivensTests(unittest.TestCase):
    def setUp(self):
        repository = Path(__file__).resolve().parents[1]
        self.config = json.loads((repository/'configs/cpu_smoke.json').read_text())
        self.config.update(learning_rate=.02, epochs=1, learning_rates=[.02], quantum_initialization='near_zero')
        self.observation_sequence = np.random.default_rng(5).uniform(-.95, .95, (12, 4))

    def test_lift_is_the_bounded_ry_bloch_vector(self):
        features = torch.tensor([-1., -.3, 0., .6, 1.], dtype=torch.float64)
        expected_bloch_vectors = torch.stack((torch.sqrt(1 - features**2), torch.zeros_like(features), features), dim=-1)
        lifted_bloch_vectors = bounded_ry_bloch_vectors(features)
        torch.testing.assert_close(lifted_bloch_vectors, expected_bloch_vectors)
        torch.testing.assert_close(torch.linalg.norm(lifted_bloch_vectors, dim=-1), torch.ones_like(features))
        model = BlochLiftedGivensEncoderDecoder(4, self.config).double()
        with self.assertRaisesRegex(ValueError, r'\[-1, 1\]'):
            model.prepare_state(np.array([0., 1.1, 0., 0.]))

    def test_without_coupling_matches_quantum_model_without_entanglement(self):
        cases = (('qte_noent', 'cte_rylift_noent', {}),
                 ('qae_noent', 'cae_rylift_noent', {}),
                 ('qrte_noent', 'crte_rylift_noent', {}),
                 ('qte_noent', 'cte_rylift_noent', {'num_blocks': 2}),
                 ('qte_noent', 'cte_rylift_noent', {'quantum_rotation_family': 'ry'}),
                 ('qte_noent', 'cte_rylift_noent', {'quantum_initialization': 'feature_neutral'}))
        for quantum_name, classical_name, overrides in cases:
            with self.subTest(quantum=quantum_name, classical=classical_name, **overrides):
                config = dict(self.config, **overrides)
                quantum_model = build_model(quantum_name, config, 303)
                classical_model = build_model(classical_name, config, 303)
                self.assertEqual([handle.name for handle in classical_model.parameter_handles],
                                 [parameter.name for parameter in quantum_model.parameter_handles])
                np.testing.assert_allclose(classical_model.parameter_values, quantum_model.parameter_values,
                                           rtol=0, atol=1e-12)
                quantum_latents, quantum_predictions = quantum_model.sequence_outputs(self.observation_sequence)
                classical_latents, classical_predictions = classical_model.sequence_outputs(self.observation_sequence)
                np.testing.assert_allclose(classical_predictions, quantum_predictions, rtol=0, atol=1e-12)
                np.testing.assert_allclose(classical_latents, quantum_latents, rtol=0, atol=1e-12)

    def test_coupling_differs_from_entanglement_and_is_the_only_difference(self):
        quantum_model = build_model('qte', self.config, 303)
        classical_model = build_model('cte_rylift', self.config, 303)
        np.testing.assert_allclose(classical_model.parameter_values, quantum_model.parameter_values, rtol=0, atol=1e-12)
        _, quantum_predictions = quantum_model.sequence_outputs(self.observation_sequence)
        _, classical_predictions = classical_model.sequence_outputs(self.observation_sequence)
        self.assertGreater(np.max(np.abs(classical_predictions - quantum_predictions)), 1e-3)

    def test_coupling_is_parameter_free_orthogonal_and_leaves_z_unchanged(self):
        model = BlochLiftedGivensEncoderDecoder(4, dict(self.config, entanglement_topology='circular')).double()
        self.assertEqual(model.coupling_edges, [(0, 1), (1, 2), (2, 3), (3, 0)])
        coupling_matrix = model.transverse_coupling_matrix
        torch.testing.assert_close(coupling_matrix @ coupling_matrix.T, torch.eye(4, dtype=torch.float64))
        self.assertEqual(len(model.trainable_params), 2*model.num_blocks*4)
        self.assertNotIn('transverse_coupling_matrix', model.state_dict())
        unit_bloch_vectors = model.prepare_state(self.observation_sequence)
        coupled_bloch_vectors = model.couple_units(unit_bloch_vectors)
        torch.testing.assert_close(coupled_bloch_vectors[..., 2], unit_bloch_vectors[..., 2])

    def test_zero_angles_without_bottleneck_reproduce_the_input(self):
        model = BlochLiftedGivensEncoderDecoder(4, dict(self.config, enforce_bottleneck=False)).double()
        model.set_params({handle: 0.0 for handle in model.trainable_params})
        _, predictions = model.forward(model.prepare_state(self.observation_sequence))
        np.testing.assert_allclose(predictions, self.observation_sequence, rtol=0, atol=1e-12)

    def test_discarded_units_reset_to_the_zero_state(self):
        model = BlochLiftedGivensEncoderDecoder(4, self.config).double()
        unit_bloch_vectors = model.prepare_state(self.observation_sequence)
        compressed = model.compress_bottleneck(unit_bloch_vectors)
        torch.testing.assert_close(compressed[..., :2, :], unit_bloch_vectors[..., :2, :])
        expected_reset_units = torch.zeros_like(compressed[..., 2:, :])
        expected_reset_units[..., 2] = 1.0
        torch.testing.assert_close(compressed[..., 2:, :], expected_reset_units)

    def test_autograd_gradient_matches_central_differences(self):
        model = build_model('crte_rylift', self.config, 303)
        training_sequences = self.observation_sequence[None]
        model.model.zero_grad()
        model.torch_loss(training_sequences).backward()
        autograd_gradient = np.concatenate([parameter.grad.detach().numpy().ravel()
                                            for parameter in model.model.parameters()])
        finite_difference_width = 1e-6
        original_values = model.parameter_values.copy()
        central_difference_gradient = np.empty_like(original_values)
        for parameter_index in range(len(original_values)):
            shifted_values = original_values.copy()
            shifted_values[parameter_index] += finite_difference_width
            model.set_parameter_values(shifted_values)
            loss_above = model.torch_loss(training_sequences).item()
            shifted_values[parameter_index] -= 2*finite_difference_width
            model.set_parameter_values(shifted_values)
            loss_below = model.torch_loss(training_sequences).item()
            central_difference_gradient[parameter_index] = (loss_above - loss_below)/(2*finite_difference_width)
        model.set_parameter_values(original_values)
        np.testing.assert_allclose(autograd_gradient, central_difference_gradient, rtol=0, atol=1e-8)

    def test_pilot_selection_save_and_restore(self):
        lifted_model_names = ('cae_rylift', 'crae_rylift', 'cte_rylift', 'crte_rylift', 'cae_rylift_noent', 'cte_rylift_noent')
        self.assertTrue(set(lifted_model_names) <= set(MODEL_NAMES))
        validate_config(dict(self.config, models=list(lifted_model_names)))
        training = np.random.default_rng(11).uniform(-.9, .9, (2, 16, 4))
        validation = np.random.default_rng(12).uniform(-.9, .9, (1, 16, 4))
        selected_model, metadata = fit_selected_model('cte_rylift', dict(self.config, epochs=2), training, validation, 303)
        self.assertEqual(metadata['optimizer'], 'adam_autograd')
        self.assertEqual(metadata['parameter_count'], 8)
        self.assertEqual(metadata['control_metadata']['coupling_angle'], np.pi/4)
        _, selected_predictions = selected_model.sequence_outputs(validation[0])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_prefix = Path(directory)/'cte_rylift'
            selected_model.save(checkpoint_prefix)
            restored_model = restore_model('cte_rylift', self.config, 303, checkpoint_prefix)
        _, restored_predictions = restored_model.sequence_outputs(validation[0])
        np.testing.assert_allclose(restored_predictions, selected_predictions, rtol=0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
