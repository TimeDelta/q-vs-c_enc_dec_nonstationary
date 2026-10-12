import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from pilot.descriptor_study import restore_model
from pilot.models import build_model, fit_selected_model, LinearEncoder
from pilot.transform_controls import bounded_ry_features, CONTROL_MODEL_NAMES


class TransformationControlTests(unittest.TestCase):
    def setUp(self):
        self.repository = Path(__file__).resolve().parents[1]
        self.config = json.loads((self.repository/'configs/cpu_smoke.json').read_text())
        self.config.update(learning_rate=.02,epochs=1,learning_rates=[.02],quantum_initialization='near_zero')
        self.observations = np.random.default_rng(97).uniform(-.8,.8,(2,8,4))

    def test_explicit_basis_spans_native_and_retained_circuit_readouts(self):
        grid = np.array(list(itertools.product((-1.,0.,1.),repeat=4)))
        interpolation_features = bounded_ry_features(grid)
        self.assertEqual(interpolation_features.shape,(81,81))
        heldout = np.random.default_rng(89).uniform(-.95,.95,(17,4))
        for name in ('qte','qte_noent','qte_ry','qte_ry_noent','qte_shared'):
            model = build_model(name,self.config,303)
            parameters = np.random.default_rng(101).uniform(-.7,.7,model.parameter_count)
            model.set_parameter_values(parameters)
            latent,predictions = model.sequence_outputs(grid)
            coefficients = np.linalg.solve(interpolation_features,np.column_stack((latent,predictions)))
            heldout_latent,heldout_predictions = model.sequence_outputs(heldout)
            np.testing.assert_allclose(bounded_ry_features(heldout)@coefficients,
                                       np.column_stack((heldout_latent,heldout_predictions)),rtol=0,atol=2e-12)
        with self.assertRaisesRegex(ValueError,r'\[-1, 1\]'):
            bounded_ry_features(np.array([[1.1]]))

    def test_affine_givens_starts_as_original_and_trains_extra_parameters(self):
        original = build_model('cte',self.config,303)
        extended = build_model('cte_affine',self.config,303)
        for expected,actual in zip(original.sequence_outputs(self.observations[0]),extended.sequence_outputs(self.observations[0])):
            np.testing.assert_allclose(expected,actual,rtol=0,atol=1e-15)
        self.assertEqual(original.parameter_count,8)
        self.assertEqual(extended.parameter_count,14)
        extended.torch_loss(self.observations).backward()
        self.assertGreater(torch.linalg.vector_norm(extended.model.decoder_gains.grad).item(),1e-6)
        self.assertGreater(torch.linalg.vector_norm(extended.model.decoder_offsets.grad).item(),1e-6)
        extended.train(self.observations,self.observations,1,1e-5)
        self.assertFalse(torch.equal(extended.model.decoder_gains,torch.ones(2,dtype=torch.float64)))

    def test_shared_heads_match_and_never_access_native_decoder(self):
        classical = build_model('cte_shared',self.config,303)
        quantum = build_model('qte_shared',self.config,303)
        self.assertEqual(classical.parameter_count,16)
        self.assertEqual(quantum.parameter_count,16)
        np.testing.assert_array_equal(classical.head_coefficients,quantum.head_coefficients)
        np.testing.assert_array_equal(classical.head_intercept,quantum.head_intercept)
        self.assertEqual(classical.control_metadata['initial_head_sha256'],quantum.control_metadata['initial_head_sha256'])
        for model in (classical,quantum):
            with patch.object(model.encoder.model,'forward',side_effect=AssertionError('Native decoder access')):
                latent,predictions = model.sequence_outputs(self.observations[0])
            np.testing.assert_array_equal(predictions,latent@model.head_coefficients+model.head_intercept)
            self.assertEqual(latent.shape,(8,2))

    def test_shared_gradients_agree_with_independent_central_differences(self):
        for name in ('cte_shared','qte_shared'):
            model = build_model(name,self.config,303)
            expected = model.gradients(self.observations,1e-6)
            parameters = model.parameter_values.copy()
            actual = []
            for index in range(model.parameter_count):
                plus,minus = parameters.copy(),parameters.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                model.set_parameter_values(plus)
                upper = model.native_mse(self.observations)
                model.set_parameter_values(minus)
                lower = model.native_mse(self.observations)
                actual.append((upper-lower)/(2e-6))
            model.set_parameter_values(parameters)
            np.testing.assert_allclose(expected,actual,rtol=0,atol=2e-6)
            self.assertGreater(np.linalg.norm(expected[:model.encoder_parameter_count]),1e-5)

    def test_ry_ablation_has_real_amplitudes_and_same_angle_count(self):
        for ordinary_name,ablation_name in [('qte','qte_ry'),('qte_noent','qte_ry_noent')]:
            ordinary = build_model(ordinary_name,self.config,303)
            ablation = build_model(ablation_name,self.config,303)
            self.assertEqual(ordinary.parameter_count,ablation.parameter_count)
            parameters = np.random.default_rng(101).uniform(-.7,.7,ablation.parameter_count)
            ablation.set_parameter_values(parameters)
            state = ablation.model.prepare_state(self.observations[0,0]).evolve(ablation.model.encoder_bound)
            np.testing.assert_allclose(state.data.imag,0,rtol=0,atol=1e-14)
            operations = set(ablation.model.encoder.count_ops())
            self.assertNotIn('rx',operations)
            self.assertNotIn('rz',operations)
            self.assertEqual('cz' in operations,'noent' not in ablation_name)

    def test_every_control_selects_on_validation_and_restores_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in CONTROL_MODEL_NAMES:
                model,metadata = fit_selected_model(name,self.config,self.observations,self.observations,303)
                self.assertTrue(np.isfinite(metadata['selected_validation_mse']))
                prefix = Path(directory)/name
                model.save(prefix)
                restored = restore_model(name,self.config,303,prefix)
                for expected,actual in zip(model.sequence_outputs(self.observations[0]),restored.sequence_outputs(self.observations[0])):
                    np.testing.assert_allclose(expected,actual,rtol=0,atol=1e-14)
                if isinstance(model,LinearEncoder):
                    self.assertEqual(model.objective,'forecast')
                    self.assertEqual(model.latent_dimension,81 if name=='encoding_ridge' else 2)
                    self.assertEqual(metadata['optimizer'],'closed_form')
                elif name.endswith('_shared'):
                    self.assertEqual(metadata['optimizer'],'adam_forward_difference_encoder_analytic_head')

    def test_ry_feature_neutral_initialization_calibrates_without_encoder_changes(self):
        for name in ('qte_ry','qte_ry_noent'):
            ordinary = build_model(name,self.config,303)
            neutral = build_model(name,dict(self.config,quantum_initialization='feature_neutral'),303)
            self.assertLess(neutral.initialization_metadata['center_max_abs_readout'],1e-8)
            np.testing.assert_array_equal(ordinary.parameter_values[:4],neutral.parameter_values[:4])

    def test_default_quantum_family_restores_existing_archived_checkpoint(self):
        configuration = json.loads((self.repository/'docs/fresh_seed_forecasting/config.json').read_text())
        archive = self.repository/'docs/fresh_seed_forecasting/fresh_seed_forecasting_assets.zip'
        checkpoint = 'data_113/training_303/qte.qpy'
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extract(checkpoint,directory)
            model = restore_model('qte',configuration,303,Path(directory)/checkpoint.removesuffix('.qpy'))
            self.assertNotIn('quantum_rotation_family',model.model.checkpoint_config())
            self.assertTrue(np.all(np.isfinite(model.sequence_outputs(self.observations[0])[1])))


if __name__=='__main__':
    unittest.main()
