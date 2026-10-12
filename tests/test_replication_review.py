import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace

from pilot.data import make_dataset
from pilot.fresh_study import run_study, verify_selection_lock
from pilot.models import LinearEncoder, LegacyEncoder, build_model
from pilot.preflight import preflight
from pilot.replication_report import primary_contrasts, optimization_diagnostics, report, verify_run
from pilot.run import source_manifest, validate_config
from pilot.transform_controls import SharedReadoutEncoder

REPOSITORY = Path(__file__).resolve().parents[1]


class ReviewGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((REPOSITORY/'configs/transformation_replication.json').read_text())

    def test_rejects_checkpoint_collisions_nonfinite_gradient_width_and_reused_seeds(self):
        for changes in ({'models': ['cte', 'cte']}, {'gradient_width': float('nan')},
                        {'gradient_width': float('inf')}, {'num_blocks': 0},
                        {'generating_latent_dimension': 5}, {'data_seeds': [97]},
                        {'execution_workers': True}, {'include_untrained_controls': 'false'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(dict(self.config, **changes))

    def test_preflight_has_no_dataset_generation_or_fitting(self):
        with (patch('pilot.data.make_dataset', side_effect=AssertionError('Dataset generated')),
             patch.object(LinearEncoder, 'fit', side_effect=AssertionError('Reference fitted')),
             patch.object(LegacyEncoder, 'train', side_effect=AssertionError('Model fitted')),
             patch.object(SharedReadoutEncoder, 'train', side_effect=AssertionError('Shared model fitted'))):
            result = preflight(self.config, require_clean_source=False)
        self.assertEqual(result['generated_datasets'], 0)
        self.assertEqual(result['fitted_models'], 0)
        self.assertEqual(result['planned_model_cases'], 624)
        self.assertEqual(result['planned_sequence_task_rows'], 29952)
        self.assertEqual(result['planned_gradient_updates'], 61440)
        self.assertEqual(result['planned_constant_rows'], 384)
        self.assertEqual(result['native_calibrations_verified'], 12)

    def test_primary_aggregation_averages_all_initializations_and_weights_conditions_equally(self):
        config = dict(data_seeds=[1], training_seeds=[10, 20],
                      primary_comparison=['qte_shared', 'cte_shared'], primary_partitions=['a', 'b'])
        records = []
        for initialization, first_values in [(10, [2., 10.]), (20, [6., 14.])]:
            for partition, value, sequence_count in [('a', first_values[0], 1), ('b', first_values[1], 3)]:
                for sequence in range(sequence_count):
                    for model, score in [('qte_shared', value), ('cte_shared', 3.)]:
                        records.append(dict(data_seed=1, training_seed=initialization, model=model,
                                            partition=partition, sequence_index=sequence, task='forecast', native_mse=score))
        result = primary_contrasts(records, config)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['first_model_mse'], 8.)
        self.assertEqual(result[0]['difference'], 5.)
        self.assertEqual([row['difference'] for row in result[0]['initialization_contrasts']], [3., 7.])

    def test_diagnostics_flag_continuing_training_and_late_validation_minimum(self):
        history = [dict(epoch=index+1, train_mse=value, validation_mse=value)
                   for index, value in enumerate([1., .9, .6, .5])]
        selections = [dict(data_seed=1, training_seed=2, model='cte_shared', selected_candidate_index=0,
                           candidates=[dict(candidate_index=0, candidate_value=.02, history=history)])]
        result = optimization_diagnostics(selections, dict(diagnostic_window_epochs=2,
                    diagnostic_relative_improvement_threshold=.01, epochs=4))
        self.assertTrue(result[0]['training_still_improving'])
        self.assertTrue(result[0]['best_validation_in_final_window'])
        self.assertTrue(result[0]['selected_candidate_on_grid_boundary'])

    def test_all_models_are_causal_when_future_observations_change(self):
        observations = np.random.default_rng(97).uniform(-.8, .8, (2, 8, 4))
        modified_sequence = observations[0].copy()
        modified_sequence[4:] *= -1
        for name in self.config['models']:
            model = build_model(name, dict(self.config, learning_rate=.02), 303)
            if isinstance(model, LinearEncoder):
                model.fit(observations)
            original_outputs = model.sequence_outputs(observations[0])
            modified_outputs = model.sequence_outputs(modified_sequence)
            for expected, actual in zip(original_outputs, modified_outputs):
                np.testing.assert_allclose(expected[:4], actual[:4], rtol=0, atol=1e-13)

    def test_four_qubit_discard_preserves_complete_retained_marginal(self):
        model = build_model('qte', dict(self.config, learning_rate=.02), 303).model
        random_generator = np.random.default_rng(97)
        statevector = random_generator.normal(size=16)+1j*random_generator.normal(size=16)
        density = DensityMatrix(statevector/np.linalg.norm(statevector))
        expected_retained = partial_trace(density, [2, 3])
        compressed = model.compress_bottleneck(density)
        np.testing.assert_allclose(partial_trace(compressed, [2, 3]).data, expected_retained.data, atol=1e-14)
        replacement_trash = np.diag([1., 0., 0., 0.])
        np.testing.assert_allclose(compressed.data, np.kron(replacement_trash, expected_retained.data), atol=1e-14)
        self.assertTrue(compressed.is_valid())
        self.assertLess(float(np.real(expected_retained.purity())), 1.)


class ReviewedRunnerTests(unittest.TestCase):
    def small_config(self):
        config = json.loads((REPOSITORY/'configs/transformation_replication.json').read_text())
        config.update(data_seeds=[89, 97], training_seeds=[303], excluded_data_seeds=[], epochs=2,
                      preflight_required=False,
                      diagnostic_window_epochs=1, learning_rates=[.02], native_ridge_penalties=[.001],
                      probe_ridge_penalties=[.001], sequence_length=64, train_sequences=1,
                      validation_sequences=1, test_sequences=1, execution_workers=1,
                      models=['cte_shared', 'qte_shared', 'encoding_ridge', 'encoding_reduced_rank', 'persistence'])
        return config

    def fixture_manifest(self, config):
        # Test fixtures exercise reporting while the reviewed source has local edits.
        return dict(source_manifest(config), working_tree_dirty=False)

    def test_corrupt_saved_checkpoint_stops_before_any_test_generation(self):
        config = self.small_config()
        config.update(data_seeds=[97], models=['pca'], include_untrained_controls=False)
        original_save = LinearEncoder.save
        def corrupt_save(model, prefix):
            original_save(model, prefix)
            with np.load(str(prefix)+'.npz') as checkpoint:
                values = {key: checkpoint[key].copy() for key in checkpoint.files}
            values['decoder_matrix'] = np.zeros_like(values['decoder_matrix'])
            np.savez(str(prefix)+'.npz', **values)
        generation_modes = []
        def checked_generation(config, seed, *, include_test=True):
            generation_modes.append(include_test)
            return make_dataset(config, seed, include_test=include_test)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(LinearEncoder, 'save', corrupt_save), patch('pilot.fresh_study.make_dataset', checked_generation):
                with self.assertRaisesRegex(RuntimeError, 'replay differs'):
                    run_study(config, Path(directory)/'run', require_clean_source=False)
        self.assertEqual(generation_modes, [False])

    def test_source_change_during_fitting_stops_before_tests(self):
        config = self.small_config()
        config.update(data_seeds=[97], models=['pca'], include_untrained_controls=False)
        original_manifest = self.fixture_manifest(config)
        changed_manifest = dict(original_manifest, git_commit='changed')
        generation_modes = []
        def checked_generation(config, seed, *, include_test=True):
            generation_modes.append(include_test)
            return make_dataset(config, seed, include_test=include_test)
        with tempfile.TemporaryDirectory() as directory:
            with patch('pilot.fresh_study.source_manifest', side_effect=[original_manifest, changed_manifest]), \
                 patch('pilot.fresh_study.make_dataset', checked_generation):
                with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                    run_study(config, Path(directory)/'run', require_clean_source=False)
        self.assertEqual(generation_modes, [False])

    def test_complete_report_replays_controls_and_rejects_score_and_lock_tampering(self):
        config = self.small_config()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'run'
            with patch('pilot.fresh_study.source_manifest', side_effect=self.fixture_manifest):
                run_study(config, source)
            verified = report(source, Path(directory)/'report')
            self.assertEqual(verified['model_cases'], 14)
            self.assertEqual(verified['sequence_task_rows'], 168)
            self.assertLess(verified['maximum_score_replay_difference'], 1e-10)
            metric_file = source/'sequence_metrics.csv'
            original_text = metric_file.read_text()
            with metric_file.open() as stream:
                reader = csv.DictReader(stream)
                fieldnames, rows = reader.fieldnames, list(reader)
            rows[0]['native_mse'] = str(float(rows[0]['native_mse'])+.1)
            with metric_file.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'test scores'):
                verify_run(source)
            metric_file.write_text(original_text)
            lock_file = source/'selection_lock.json'
            changed_lock = json.loads(lock_file.read_text())
            changed_lock['test_partitions_generated'] = True
            lock_file.write_text(json.dumps(changed_lock))
            with self.assertRaisesRegex(RuntimeError, 'lock file changed'):
                verify_selection_lock(source)

    def test_shared_control_parallel_fitting_matches_sequential_results(self):
        config = self.small_config()
        config.update(models=['cte_shared', 'qte_shared'], epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            sequential = run_study(config, Path(directory)/'sequential', require_clean_source=False)
            parallel = run_study(dict(config, execution_workers=2), Path(directory)/'parallel', require_clean_source=False)
        self.assertEqual(sequential, parallel)


if __name__ == '__main__':
    unittest.main()
