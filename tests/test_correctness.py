"""Regressions for experimental validity, not historical numerical results."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from qiskit.quantum_info import DensityMatrix, partial_trace

from analysis import symbol_entropy, quantize_signal_bayesian_block_feature_bins
from loss import main_cost_function, cost_function_for_model_type
from models import ClassicalEncoderDecoder, QuantumEncoderDecoder
from optimize_hyperparams import hyperband_search


class IdentityModel:
    bottleneck_size = 1
    def reset_hidden_state(self):
        pass
    def prepare_state(self, features):
        return features
    def forward(self, features):
        return features, features


class LossTests(unittest.TestCase):
    def test_objective_dispatch(self):
        series = [(0, np.array([[0.0], [1.0], [2.0]]))]
        for model_type in ('qae', 'qrae', 'cae', 'crae', 'qte', 'qrte', 'cte', 'crte'):
            loss_function = cost_function_for_model_type(model_type, lambda state, size: 0.0)
            self.assertEqual(loss_function(series, IdentityModel(), 0)[0],
                             1.0 if model_type.endswith('te') else 0.0)

    def test_loss_is_order_independent_and_timestep_weighted(self):
        class ZeroModel(IdentityModel):
            def forward(self, features):
                return features, np.zeros_like(features)
        sequences = [(0, np.ones((2, 1))), (1, np.full((4, 1), 2.0))]
        expected_mse = (2 * 1 + 4 * 4) / 6
        for ordering in (sequences, sequences[::-1]):
            self.assertEqual(main_cost_function(ordering, ZeroModel(), lambda state, size: 0, 0)[0], expected_mse)

    def test_invalid_partitions_fail(self):
        for partition in ([], [(0, np.empty((0, 2)))], [(0, np.array([[np.nan]]))]):
            with self.assertRaises(ValueError):
                main_cost_function(partition, IdentityModel(), lambda state, size: 0)


class ModelTests(unittest.TestCase):
    def quantum_model(self, recurrent=False, bottleneck=1, encoding='bounded_ry'):
        config = dict(bottleneck_size=bottleneck, num_blocks=1,
                      entanglement_topology='none', feature_encoding=encoding)
        model = QuantumEncoderDecoder(2, config, recurrent)
        model.set_params({parameter: 0.0 for parameter in model.trainable_params})
        return model

    def test_quantum_encoding_readout_roundtrip(self):
        for encoding, features in [('bounded_ry', np.array([-0.7, 0.3])),
                                   ('arctan_ry', np.array([-10.0, 2.0]))]:
            model = self.quantum_model(bottleneck=2, encoding=encoding)
            np.testing.assert_allclose(model.collapse_state(model.prepare_state(features)), features, atol=1e-10)

    def test_discarded_quantum_input_cannot_reach_decoder(self):
        model = self.quantum_model()
        outputs = []
        for features in (np.array([0.4, -0.8]), np.array([0.4, 0.8])):
            _, prediction = model.forward(model.prepare_state(features))
            outputs.append(prediction.data)
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)

    def test_reset_preserves_retained_entangled_mixed_state(self):
        model = self.quantum_model()
        bell_state = DensityMatrix(np.array([1, 0, 0, 1]) / np.sqrt(2))
        reset_state = model.compress_bottleneck(bell_state)
        np.testing.assert_allclose(partial_trace(reset_state, [1]).data, np.eye(2) / 2, atol=1e-12)
        self.assertAlmostEqual(float(np.real(reset_state.trace())), 1.0)
        self.assertAlmostEqual(float(np.real(reset_state.purity())), 0.5)

    def test_recurrent_quantum_states_are_valid_and_resettable(self):
        model = self.quantum_model(recurrent=True)
        first_features = np.array([-0.8, 0.2])
        _, first_prediction = model.forward(model.prepare_state(first_features))
        for features in [np.array([0.8, -0.2]), np.zeros(2)]:
            bottleneck, prediction = model.forward(model.prepare_state(features))
            self.assertTrue(bottleneck.is_valid())
            self.assertTrue(prediction.is_valid())
        model.reset_hidden_state()
        _, repeat_prediction = model.forward(model.prepare_state(first_features))
        np.testing.assert_allclose(first_prediction.data, repeat_prediction.data, atol=1e-12)

    def test_mixed_state_readout_does_not_choose_an_eigenvector(self):
        model = self.quantum_model(bottleneck=2)
        maximally_mixed_state = DensityMatrix(np.eye(4) / 4)
        np.testing.assert_allclose(model.collapse_state(maximally_mixed_state), [0, 0], atol=1e-12)

    def test_classical_compression_and_autograd(self):
        model = ClassicalEncoderDecoder(2, dict(bottleneck_size=1), is_recurrent=True).double()
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.zero_()
        input_features = torch.tensor([0.4, 0.8], dtype=torch.float64, requires_grad=True)
        _, prediction = model.forward_tensor(input_features)
        np.testing.assert_allclose(prediction.detach().numpy(), [0.4, 0.0])
        prediction.sum().backward()
        np.testing.assert_allclose(input_features.grad.numpy(), [1.0, 0.0])
        self.assertEqual(model.hidden_state[1].item(), 0.0)

    def test_classical_checkpoint_protocol_and_replay(self):
        config = dict(bottleneck_size=1)
        model = ClassicalEncoderDecoder(2, config).double()
        features = np.array([0.2, -0.4])
        _, expected = model.forward(model.prepare_state(features))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory) / 'model')
            model.save(checkpoint_path)
            restored = ClassicalEncoderDecoder(2, config).double()
            restored.load(checkpoint_path)
            _, actual = restored.forward(restored.prepare_state(features))
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            torch.save(model.state_dict(), checkpoint_path + '.pth')
            with self.assertRaisesRegex(ValueError, 'Historical checkpoints'):
                restored.load(checkpoint_path)

    def test_quantum_checkpoint_roundtrip(self):
        model = self.quantum_model(recurrent=True)
        features = np.array([0.2, -0.4])
        _, expected_prediction = model.forward(model.prepare_state(features))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory) / 'model')
            model.save(checkpoint_path)
            restored_model = self.quantum_model(recurrent=True)
            restored_model.load(checkpoint_path)
            _, actual_prediction = restored_model.forward(restored_model.prepare_state(features))
        np.testing.assert_allclose(expected_prediction.data, actual_prediction.data, atol=1e-12)


class AnalysisTests(unittest.TestCase):
    def test_symbol_entropy_is_label_invariant(self):
        symbols = np.tile([0, 1], 50)
        self.assertEqual(symbol_entropy(symbols, lambda values: values), 1.0)
        self.assertEqual(symbol_entropy(symbols * 1000, lambda values: values), 1.0)
        self.assertEqual(symbol_entropy(np.zeros(10), lambda values: values), 0.0)

    def test_one_dimensional_bayesian_quantization(self):
        symbols = quantize_signal_bayesian_block_feature_bins(np.linspace(-1, 1, 32))
        self.assertEqual(len(symbols), 32)

    def test_hyperband_prunes_and_selects_at_full_budget(self):
        observed_budgets = []
        def fake_loss(data, model_type, config, epochs):
            observed_budgets.append(epochs)
            return 0.0 if epochs == 1 else float(config['candidate'])
        candidate_counter = iter(range(100))
        with patch('optimize_hyperparams.sample_hyperparameters', side_effect=lambda _: dict(candidate=next(candidate_counter))), \
             patch('optimize_hyperparams.get_loss', side_effect=fake_loss), \
             patch('optimize_hyperparams.MODEL_TYPES', ['cae']), contextlib.redirect_stdout(io.StringIO()):
            best_config, best_loss = hyperband_search([[(0, np.ones((4, 2)))], []], 4, 2)
        self.assertEqual(observed_budgets.count(1), 4)
        self.assertEqual(observed_budgets.count(2), 5)
        self.assertEqual(observed_budgets.count(4), 5)
        self.assertEqual(best_config['candidate'], 0)
        self.assertEqual(best_loss, 0.0)


if __name__ == '__main__':
    unittest.main()
