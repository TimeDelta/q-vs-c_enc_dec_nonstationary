import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pilot.descriptor_study import nested_predictions, restore_model, ridge_predictions
from pilot.models import build_model, LinearEncoder


class DescriptorStudyTests(unittest.TestCase):
    def test_heldout_targets_cannot_change_predictions_or_penalty_selection(self):
        generator = np.random.default_rng(97)
        features = generator.normal(size=(24,3))
        groups = np.repeat(np.arange(6),4)
        targets = features[:,0]+generator.normal(size=24)*.2
        expected,audit = nested_predictions(features,targets,groups,[.01,.1,1.0])
        changed = targets.copy()
        changed[groups==2] += 1000
        observed,new_audit = nested_predictions(features,changed,groups,[.01,.1,1.0])
        np.testing.assert_array_equal(expected[groups==2],observed[groups==2])
        self.assertEqual(audit[2],new_audit[2])
        for outer in audit:
            self.assertNotIn(outer['heldout_data_seed'],outer['training_data_seeds'])
            for inner in outer['inner_folds']:
                self.assertNotIn(outer['heldout_data_seed'],inner['training_data_seeds'])
                self.assertNotIn(inner['heldout_data_seed'],inner['training_data_seeds'])

    def test_training_only_scaling_and_unpenalized_intercept(self):
        training = np.array([[0.0],[1.0],[2.0]])
        predictions = ridge_predictions(training,np.array([4.0,5.0,6.0]),np.array([[3.0],[1000.0]]),1.0)
        # Fitting-row scaling gives slope 1/2 and intercept 4.5.
        np.testing.assert_allclose(predictions,[6.0,504.5],rtol=0,atol=1e-12)

    def test_checkpoint_families_preserve_inference_outputs(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'configs/fresh_seed_forecasting.json').read_text())
        config['learning_rate'] = .02
        example = np.random.default_rng(97).uniform(-.4,.4,(64,4))
        with tempfile.TemporaryDirectory() as directory:
            for base_name in ('qte','qte_noent','cte','mlp_te','gru_te','pca','reduced_rank','random_linear','persistence'):
                model = build_model(base_name,config,303)
                if isinstance(model,LinearEncoder):
                    model.fit(example[None])
                else:
                    model.name = 'untrained_'+base_name
                prefix = Path(directory)/model.name
                expected = model.sequence_outputs(example)
                model.save(prefix)
                restored = restore_model(model.name,config,303,prefix)
                for original,current in zip(expected,restored.sequence_outputs(example)):
                    np.testing.assert_allclose(original,current,rtol=0,atol=1e-14)


if __name__=='__main__':
    unittest.main()
