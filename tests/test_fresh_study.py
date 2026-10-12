import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pilot.data import make_dataset
from pilot.fresh_study import run_study, verify_selection_lock
from pilot.models import build_model
from pilot.run import evaluate_model, fit_evaluation_probes


class FreshStudyTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'configs/fresh_seed_forecasting.json').read_text())
        self.config.update(data_seeds=[97],models=['pca','reduced_rank'],sequence_length=64,test_sequences=1,
                           include_untrained_controls=False,execution_workers=1)

    def test_all_selections_lock_before_test_generation_and_remain_unchanged(self):
        generation_modes = []
        def checked_generation(config, seed, *, include_test=True):
            generation_modes.append(include_test)
            if include_test:
                lock = verify_selection_lock(output)
                self.assertEqual(lock['model_cases'],2)
                self.assertFalse(lock['test_partitions_generated'])
            return make_dataset(config,seed,include_test=include_test)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            with patch('pilot.fresh_study.make_dataset',side_effect=checked_generation):
                records = run_study(self.config,output,require_clean_source=False)
            self.assertEqual(generation_modes,[False,True])
            self.assertEqual(len(records),24)
            verify_selection_lock(output)
            locked_file = output/'constant_forecast_predictions.json'
            locked_file.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Selection lock changed'):
                verify_selection_lock(output)

    def test_test_evaluation_uses_locked_coefficients_without_probe_selection(self):
        partitions,_ = make_dataset(self.config,97)
        model = build_model('pca',self.config,303)
        model.fit(partitions['train'])
        probes,metadata = fit_evaluation_probes(model,partitions['train'],partitions['validation'],self.config)
        before = {task:(probe[0].copy(),probe[1].copy()) for task,probe in probes.items()}
        records = []
        with tempfile.TemporaryDirectory() as directory:
            with patch('pilot.run.fit_evaluation_probes',side_effect=AssertionError('Probe refit after lock')):
                evaluate_model(model,partitions,self.config,97,303,records,Path(directory),
                               locked_probes=probes,locked_probe_metadata=metadata)
        self.assertEqual(len(records),12)
        for task in probes:
            for original,current in zip(before[task],probes[task]):
                np.testing.assert_array_equal(original,current)

class WorkerSerializationTests(unittest.TestCase):
    def test_quantum_worker_transfer_preserves_latent_and_native_predictions(self):
        import pickle
        config = json.loads((Path(__file__).resolve().parents[1]/'configs/fresh_seed_forecasting.json').read_text())
        config['learning_rate']=.02
        example=np.random.default_rng(97).uniform(-.4,.4,(8,4))
        for name in ['qte','qte_noent']:
            model=build_model(name,config,303)
            expected=model.sequence_outputs(example)
            transferred=pickle.loads(pickle.dumps(model))
            observed=transferred.sequence_outputs(example)
            for original,current in zip(expected,observed):
                np.testing.assert_array_equal(original,current)


class ConcurrentStudyTests(unittest.TestCase):
    def test_parallel_fitting_preserves_sequential_test_outcomes(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/fresh_seed_forecasting.json').read_text())
        config.update(data_seeds=[89,97],models=['qte_noent','cte'],sequence_length=64,epochs=1,
                      learning_rates=[.02],test_sequences=1,include_untrained_controls=False)
        with tempfile.TemporaryDirectory() as directory:
            sequential=run_study(dict(config,execution_workers=1),Path(directory)/'sequential',require_clean_source=False)
            concurrent=run_study(dict(config,execution_workers=2),Path(directory)/'concurrent',require_clean_source=False)
            self.assertEqual(sequential,concurrent)


if __name__=='__main__':
    unittest.main()
