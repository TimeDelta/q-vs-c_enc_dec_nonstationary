import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pilot.data import make_dataset
from pilot.metrics import choose_probe, predict_probe, temporal_descriptors, transformation_controls
from pilot.models import LinearEncoder, build_model, fit_selected_model
from pilot.run import run_pilot, validate_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'cpu_smoke.json'


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())

    def test_partition_reproducibility_independence_and_factor_control(self):
        partitions, metadata = make_dataset(self.config, 17)
        repeated_partitions, repeated_metadata = make_dataset(self.config, 17)
        for partition_name, partition_metadata in metadata['partitions'].items():
            np.testing.assert_array_equal(partitions[partition_name], repeated_partitions[partition_name])
            self.assertTrue(np.all(np.abs(partitions[partition_name]) <= 1))
        self.assertEqual(metadata, repeated_metadata)
        hashes = [partition['sha256'] for partition in metadata['partitions'].values()]
        self.assertEqual(len(hashes), len(set(hashes)))
        original_profile = metadata['partitions']['train']['profile']
        shifted_profile = metadata['partitions']['test_mean_shift']['profile']
        changed_factors = [key for key in original_profile if original_profile[key] != shifted_profile[key]]
        self.assertEqual(changed_factors, ['mean_shift'])

    def test_training_validation_generation_reserves_tests(self):
        all_partitions, _ = make_dataset(self.config, 17)
        limited_partitions, metadata = make_dataset(self.config, 17, include_test=False)
        self.assertEqual(set(metadata['partitions']), {'train', 'validation'})
        self.assertFalse(any(key.startswith('test_') for key in limited_partitions))
        for key, values in limited_partitions.items():
            np.testing.assert_array_equal(values, all_partitions[key])

    def test_feature_neutral_initialization_preserves_encoder_and_parameter_count(self):
        self.config['learning_rate'] = .02
        for name in ('qae', 'qte_noent', 'qrte'):
            ordinary = build_model(name, self.config, 101)
            neutral_config = dict(self.config, quantum_initialization='feature_neutral')
            neutral = build_model(name, neutral_config, 101)
            self.assertEqual(ordinary.parameter_count, neutral.parameter_count)
            changed = neutral.initialization_metadata['changed_parameter_indices']
            for index in range(neutral.parameter_count):
                if index not in changed:
                    self.assertEqual(ordinary.parameter_values[index], neutral.parameter_values[index])
            self.assertLess(neutral.initialization_metadata['center_max_abs_readout'], 1e-8)
            example = np.random.default_rng(12).uniform(-1, 1, (8, 4))
            np.testing.assert_array_equal(ordinary.sequence_outputs(example)[0], neutral.sequence_outputs(example)[0])
            repeated = build_model(name, neutral_config, 101)
            np.testing.assert_array_equal(neutral.parameter_values, repeated.parameter_values)

    def test_validation_control_has_no_test_partitions(self):
        from pilot.validation_control import run_control
        config = dict(self.config, models=['pca', 'reduced_rank'],
                      initialization_modes=['near_zero', 'feature_neutral'],
                      include_untrained_controls=False)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            records = run_control(config, output)
            self.assertEqual(len(records), 2)
            arrays = np.load(output/'data_17/training_validation.npz')
            self.assertFalse(any(key.startswith('test_') for key in arrays.files))
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertIn('test partitions are not generated', manifest['evaluation'])

    def test_linear_probe_uses_selected_training_fit(self):
        generator = np.random.default_rng(3)
        training = generator.normal(size=(200, 2))
        validation = generator.normal(size=(40, 2))
        coefficients = np.array([[2.0, -1.0], [0.5, 3.0]])
        training_targets = training @ coefficients + np.array([1.0, -2.0])
        validation_targets = validation @ coefficients + np.array([1.0, -2.0])
        probe, metadata = choose_probe(training, training_targets, validation, validation_targets, [1e-8, 1.0])
        self.assertEqual(metadata['ridge_penalty'], 1e-8)
        np.testing.assert_allclose(predict_probe(validation, probe), validation_targets, atol=1e-6)

    def test_invertible_rotation_preserves_probe_predictions(self):
        generator = np.random.default_rng(5)
        latent_sequence = generator.normal(size=(128, 2))
        probe = (np.array([[2.0, 0.5], [-1.0, 3.0]]), np.array([0.3, -0.2]))
        controls = transformation_controls(latent_sequence, probe, generator)
        self.assertLess(controls['rotation_prediction_max_difference'], 1e-12)
        self.assertLess(controls['scale_descriptor_change'], 1e-12)

    def test_reduced_rank_baseline_has_enforced_rank(self):
        partitions, _ = make_dataset(self.config, 17)
        model = LinearEncoder('reduced_rank', self.config, 101)
        model.fit(partitions['train'])
        self.assertLessEqual(np.linalg.matrix_rank(model.encoder_matrix @ model.decoder_matrix), self.config['bottleneck_size'])
        self.assertTrue(np.isfinite(model.native_mse(partitions['test_id'])))

    def test_checkpoint_replays_classical_native_predictions(self):
        import torch
        partitions, _ = make_dataset(self.config, 17)
        model, _ = fit_selected_model('crae', self.config, partitions['train'], partitions['validation'], 101)
        expected_predictions = model.sequence_outputs(partitions['test_id'][0])[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            model.save(path)
            checkpoint = torch.load(str(path) + '.pt', weights_only=True)
            restored_model = build_model('crae', checkpoint['config'], 101)
            restored_model.model.load_state_dict(checkpoint['state_dict'])
            actual_predictions = restored_model.sequence_outputs(partitions['test_id'][0])[1]
        np.testing.assert_allclose(expected_predictions, actual_predictions, atol=1e-12)

    def test_runner_writes_finite_records_and_refuses_reuse(self):
        self.config['models'] = ['pca', 'reduced_rank', 'persistence']
        self.config['include_untrained_controls'] = False
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / 'run'
            records = run_pilot(self.config, output_path)
            self.assertEqual(len(records), 3 * 6 * 2)
            self.assertTrue(all(np.isfinite(record['probe_mse']) for record in records))
            self.assertTrue((output_path / 'manifest.json').exists())
            self.assertTrue((output_path / 'sequence_metrics.csv').exists())
            with self.assertRaises(FileExistsError):
                run_pilot(self.config, output_path)


if __name__ == '__main__':
    unittest.main()
