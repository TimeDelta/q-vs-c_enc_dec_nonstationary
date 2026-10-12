"""Controls for feature encoding, decoder access and transformation restrictions.

The original Givens model remains a separate model. These controls isolate
specific differences rather than assert equal overall expressive capacity.
"""
import hashlib
import json

import numpy as np
import torch
from torch import nn
from qiskit.quantum_info import DensityMatrix

from optimizers import adam_update
from pilot.models import LegacyEncoder, LinearEncoder

CONTROL_MODEL_NAMES = ('encoding_ridge', 'encoding_reduced_rank', 'cte_affine',
                       'cte_shared', 'qte_shared', 'qte_ry', 'qte_ry_noent')


def bounded_ry_features(observations):
    """Tensor products of [1, x_i, sqrt(1-x_i**2)]; last coordinate varies fastest.

For a single bounded RY embedding and data-independent quantum channels,
expectation predictions lie in this feature span. No recurrence or reuploading
is included in that claim. The basis has 3**feature_count columns.
"""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim < 1 or not np.all(np.isfinite(observations)):
        raise ValueError('Expected finite observations with a final feature axis')
    if np.any(np.abs(observations)>1+1e-12):
        raise ValueError('Bounded RY features require observations in [-1, 1]')
    feature_count = observations.shape[-1]
    if not 1 <= feature_count <= 6:
        raise ValueError('Explicit feature controls support one to six input features')
    clipped = np.clip(observations,-1,1)
    products = np.ones(observations.shape[:-1]+(1,))
    for coordinate in range(feature_count):
        coordinate_features = np.stack((np.ones_like(clipped[...,coordinate]),clipped[...,coordinate],
                                       np.sqrt(1-clipped[...,coordinate]**2)),axis=-1)
        products = (products[..., :, None]*coordinate_features[..., None, :]).reshape(
            observations.shape[:-1]+(-1,))
    return products


class EncodingLinearEncoder(LinearEncoder):
    def __init__(self,name,config,initialization_seed,ridge_penalty):
        if not 1 <= config['num_features'] <= 6:
            raise ValueError('Explicit feature controls support one to six input features')
        super().__init__(name,config,initialization_seed,ridge_penalty)
        self.objective = 'forecast'
        self.basis_dimension = 3**config['num_features']
        self.latent_dimension = self.basis_dimension if name=='encoding_ridge' else config['bottleneck_size']
        self.control_metadata = dict(feature_basis='tensor products of 1, x and sqrt(1-x**2)',
                    basis_dimension=self.basis_dimension,readout_dimension=self.latent_dimension,
                    compressed=name=='encoding_reduced_rank',
                    scope='single embedding, nonrecurrent expectation predictions',
                    gradient_parameters=0,coefficient_capacity='not parameter-matched to the circuit')

    def fit(self,observations):
        inputs = bounded_ry_features(observations[:,:-1]).reshape(-1,self.basis_dimension)
        targets = observations[:,1:].reshape(-1,observations.shape[-1])
        self.input_mean = inputs.mean(axis=0)
        self.target_mean = targets.mean(axis=0)
        centered_inputs = inputs-self.input_mean
        covariance = centered_inputs.T@centered_inputs/len(inputs)
        cross_covariance = centered_inputs.T@(targets-self.target_mean)/len(inputs)
        regularized_covariance = covariance+self.ridge_penalty*np.eye(self.basis_dimension)
        if self.name=='encoding_ridge':
            self.encoder_matrix = np.eye(self.basis_dimension)
            self.decoder_matrix = np.linalg.solve(regularized_covariance,cross_covariance)
        else:
            eigenvalues,eigenvectors = np.linalg.eigh(regularized_covariance)
            inverse_square_root = (eigenvectors*(1/np.sqrt(eigenvalues)))@eigenvectors.T
            left_vectors,singular_values,right_vectors = np.linalg.svd(inverse_square_root@cross_covariance,full_matrices=False)
            self.encoder_matrix = inverse_square_root@left_vectors[:,:self.latent_dimension]
            self.decoder_matrix = singular_values[:self.latent_dimension,None]*right_vectors[:self.latent_dimension]

    def sequence_outputs(self,sequence):
        latent_features = (bounded_ry_features(sequence)-self.input_mean)@self.encoder_matrix
        return latent_features,latent_features@self.decoder_matrix+self.target_mean


class GivensAffineEncoder(LegacyEncoder):
    """Original encoder/decoder rotations plus retained-coordinate gains and output bias."""
    def __init__(self,name,config,initialization_seed):
        super().__init__(name,config,initialization_seed)
        original_angle_count = len(self.parameter_handles)
        self.model.decoder_gains = nn.Parameter(torch.ones(self.latent_dimension,dtype=torch.float64))
        self.model.decoder_offsets = nn.Parameter(torch.zeros(config['num_features'],dtype=torch.float64))
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.parameter_handles = self.model.trainable_params
        self.parameter_values = np.array([parameter.detach().item() for parameter in self.parameter_handles])
        self.initial_parameter_values = self.parameter_values.copy()
        self.control_metadata = dict(original_givens_angles=original_angle_count,
                    additional_gains=self.latent_dimension,additional_offsets=config['num_features'],
                    initialization='unit gains and zero offsets reproduce original cte',
                    gains_location='decoder input; latent readouts remain the original encoder coordinates')

    def tensor_outputs(self,observations):
        encoded = observations
        for block in self.model.encoder:
            encoded = block(encoded)
        retained = encoded[...,:self.latent_dimension]
        decoder_input = torch.cat((retained*self.model.decoder_gains,
                                  torch.zeros_like(encoded[...,self.latent_dimension:])),dim=-1)
        predictions = decoder_input
        for block in self.model.decoder:
            predictions = block(predictions)
        return retained,predictions+self.model.decoder_offsets

    def sequence_outputs(self,sequence):
        with torch.no_grad():
            latent,predictions = self.tensor_outputs(torch.as_tensor(sequence,dtype=torch.float64))
        return latent.numpy(),predictions.numpy()

    def torch_loss(self,observations):
        inputs = torch.as_tensor(observations[:,:-1],dtype=torch.float64)
        targets = torch.as_tensor(observations[:,1:],dtype=torch.float64)
        return torch.mean((self.tensor_outputs(inputs)[1]-targets)**2)


class SharedReadoutEncoder:
    """Two real readouts with an identically initialized affine decoder.

Both encoder families use the same forward-difference encoder gradients,
analytic affine-head gradients, Adam rule and validation-only selection.
The quantum native circuit decoder and discarded-qubit state are inaccessible.
"""
    def __init__(self,name,config,initialization_seed):
        self.name = name
        self.base_name = 'qte' if name=='qte_shared' else 'cte'
        self.config = dict(config)
        self.objective = 'forecast'
        self.quantum = self.base_name=='qte'
        self.learning_rate = config['learning_rate']
        self.latent_dimension = config['bottleneck_size']
        self.feature_count = config['num_features']
        self.initialization_seed = initialization_seed
        self.initialization_mode = 'near_zero_encoder_shared_affine_head'
        # Native quantum decoder calibration does not belong to this replaced head.
        self.encoder = LegacyEncoder(self.base_name,dict(config,quantum_initialization='near_zero'),initialization_seed)
        self.encoder_parameter_count = config['num_features']*config['num_blocks']
        self.encoder_handles = self.encoder.parameter_handles[:self.encoder_parameter_count]
        head_generator = np.random.default_rng(np.random.SeedSequence([initialization_seed,441]))
        head_coefficients = head_generator.uniform(-.15,.15,(self.latent_dimension,self.feature_count))
        self.parameter_count = self.encoder_parameter_count+head_coefficients.size+self.feature_count
        self.parameter_values = np.concatenate((self.encoder.initial_parameter_values[:self.encoder_parameter_count],
                                               head_coefficients.ravel(),np.zeros(self.feature_count)))
        self.initial_parameter_values = self.parameter_values.copy()
        self.loss_evaluations = self.state_preparations = 0
        self.set_parameter_values(self.parameter_values)
        self.control_metadata = dict(encoder_parameters=self.encoder_parameter_count,
                    head_parameters=head_coefficients.size+self.feature_count,
                    head='affine on retained real coordinates or single-qubit Z expectations only',
                    initial_head_sha256=hashlib.sha256(self.parameter_values[self.encoder_parameter_count:].tobytes()).hexdigest(),
                    decoder_access='two real readouts; no density-matrix or discarded-subsystem access',
                    optimizer='same forward-difference encoder and analytic affine-head gradients in both families')

    def set_parameter_values(self,values):
        self.parameter_values = np.asarray(values,dtype=float).copy()
        if self.parameter_values.shape!=(self.parameter_count,):
            raise ValueError('Incorrect shared-control parameter vector')
        self.encoder.model.set_params(dict(zip(self.encoder_handles,self.parameter_values[:self.encoder_parameter_count])))
        head_start = self.encoder_parameter_count
        head_end = head_start+self.latent_dimension*self.feature_count
        self.head_coefficients = self.parameter_values[head_start:head_end].reshape(self.latent_dimension,self.feature_count)
        self.head_intercept = self.parameter_values[head_end:]

    def encoder_features(self,sequence):
        if self.quantum:
            sequence_key = hashlib.sha256(np.ascontiguousarray(sequence).tobytes()).hexdigest()
            if sequence_key not in self.encoder.prepared_sequences:
                self.encoder.prepared_sequences[sequence_key] = [self.encoder.model.prepare_state(features) for features in sequence]
                self.state_preparations += len(sequence)
            return np.asarray([self.encoder.model.latent_features(DensityMatrix(state).evolve(self.encoder.model.encoder_bound))
                               for state in self.encoder.prepared_sequences[sequence_key]])
        with torch.no_grad():
            encoded = torch.as_tensor(sequence,dtype=torch.float64)
            for block in self.encoder.model.encoder:
                encoded = block(encoded)
        return encoded.numpy()[...,:self.latent_dimension]

    def sequence_outputs(self,sequence):
        latent = self.encoder_features(sequence)
        return latent,latent@self.head_coefficients+self.head_intercept

    def training_arrays(self,observations):
        latent = np.concatenate([self.encoder_features(sequence[:-1]) for sequence in observations])
        targets = observations[:,1:].reshape(-1,self.feature_count)
        return latent,targets

    def native_mse(self,observations):
        self.loss_evaluations += 1
        latent,targets = self.training_arrays(observations)
        return float(np.mean((latent@self.head_coefficients+self.head_intercept-targets)**2))

    def gradients(self,observations,width):
        current_parameters = self.parameter_values.copy()
        latent,targets = self.training_arrays(observations)
        errors = latent@self.head_coefficients+self.head_intercept-targets
        current_loss = float(np.mean(errors**2))
        gradients = np.empty(self.parameter_count)
        for index in range(self.encoder_parameter_count):
            perturbed = current_parameters.copy()
            perturbed[index] += width
            self.set_parameter_values(perturbed)
            gradients[index] = (self.native_mse(observations)-current_loss)/width
        self.set_parameter_values(current_parameters)
        gradients[self.encoder_parameter_count:] = np.concatenate((
            (2*latent.T@errors/errors.size).ravel(),2*errors.sum(axis=0)/errors.size))
        return gradients

    def train(self,training,validation,epochs,gradient_width):
        history,best_snapshot,best_validation = [],None,np.inf
        first_moment = np.zeros(self.parameter_count)
        second_moment = np.zeros(self.parameter_count)
        for epoch in range(1,epochs+1):
            gradients = self.gradients(training,gradient_width)
            values,first_moment,second_moment = adam_update(self.parameter_values,gradients,first_moment,second_moment,
                                                           epoch,self.learning_rate)
            self.set_parameter_values(values)
            training_mse,validation_mse = self.native_mse(training),self.native_mse(validation)
            if not np.all(np.isfinite([training_mse,validation_mse,*gradients])):
                raise FloatingPointError('Nonfinite shared-readout training')
            history.append(dict(epoch=epoch,train_mse=training_mse,validation_mse=validation_mse,
                                gradient_norm=float(np.linalg.norm(gradients))))
            if validation_mse<best_validation:
                best_validation,best_snapshot = validation_mse,self.parameter_values.copy()
        self.set_parameter_values(best_snapshot)
        return history,best_validation

    def checkpoint_config(self):
        return {key:self.config[key] for key in ('num_features','bottleneck_size','num_blocks',
                                                 'entanglement_topology','entanglement_gate')}

    def save(self,checkpoint_prefix):
        np.savez(str(checkpoint_prefix)+'.npz',protocol_version=2,model_name=self.name,
                 architecture_json=json.dumps(self.checkpoint_config(),sort_keys=True),parameter_values=self.parameter_values)

    def load(self,checkpoint_prefix):
        with np.load(str(checkpoint_prefix)+'.npz') as checkpoint:
            if int(checkpoint['protocol_version'])!=2 or str(checkpoint['model_name'])!=self.name:
                raise ValueError('Shared-control checkpoint name or protocol mismatch')
            if json.loads(str(checkpoint['architecture_json']))!=self.checkpoint_config():
                raise ValueError('Shared-control checkpoint architecture mismatch')
            self.set_parameter_values(checkpoint['parameter_values'])


def build_control(name,config,initialization_seed,ridge_penalty):
    if name in ('encoding_ridge','encoding_reduced_rank'):
        return EncodingLinearEncoder(name,config,initialization_seed,ridge_penalty)
    if name=='cte_affine':
        return GivensAffineEncoder(name,config,initialization_seed)
    if name in ('qte_shared','cte_shared'):
        return SharedReadoutEncoder(name,config,initialization_seed)
    raise ValueError('Unknown transformation control')
