import cma
import numpy as np
import math
import random
import torch
import torch.nn as nn

from analysis import lempel_ziv_complexity_continuous, hurst_exponent, higuchi_fractal_dimension

class SimpleTransformerGenerator(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, output_dim, max_seq_length):
        super(SimpleTransformerGenerator, self).__init__()
        self.model_dim = model_dim
        self.max_seq_length = max_seq_length
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_length, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        x = self.input_proj(x) # (batch_size, seq_length, model_dim)
        pos_emb = self.pos_embedding[:seq_length, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb
        # expects input shape: (seq_length, batch_size, model_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = self.output_proj(x)
        return x

def update_model_params(model, flat_params, param_shapes):
    new_params = []
    idx = 0
    for shape in param_shapes:
        size = np.prod(shape)
        param = flat_params[idx:idx+size].reshape(shape)
        new_params.append(torch.tensor(param, dtype=torch.float32))
        idx += size
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, new_param in zip(state_dict.keys(), new_params):
        new_state_dict[key] = new_param
    model.load_state_dict(new_state_dict)

def get_model_param_shapes(model):
    shapes = []
    for param in model.parameters():
        shapes.append(param.detach().cpu().numpy().shape)
    return shapes

def objective(generated_params, model, param_shapes, length, target_lzc, target_he, target_hfd, target_mean, target_stdev):
    update_model_params(model, generated_params, param_shapes)

    targets = np.array([target_lzc, target_he, target_hfd, target_mean, target_stdev], dtype=np.float32)
    # Reshape to (1, 1, input_dim) and repeat along the time axis to get (1, length, input_dim)
    targets = targets.reshape(1, 1, -1)
    targets = np.repeat(targets, length, axis=1)

    targets_tensor = torch.tensor(targets)

    print('  Generating data for candidate')
    with torch.no_grad():
        generated = model(targets_tensor) # shape: (1, length, output_dim)
    generated_data = generated.squeeze(0).cpu().numpy() # shape: (length, output_dim)

    computed_lzc = lempel_ziv_complexity_continuous(generated_data)

    he_error = np.mean([np.abs(target_he - he) for he in hurst_exponent(generated_data)])
    hfd_error = np.mean([np.abs(target_hfd - hfd) for hfd in higuchi_fractal_dimension(generated_data)])
    mean_error = np.mean([np.abs(target_mean - mean) / max(target_mean, mean) for mean in np.mean(generated_data, axis=0)])
    stdev_error = np.mean([np.abs(target_stdev - stdev) / max(target_stdev, stdev) for stdev in np.std(generated_data, axis=0)])

    total_error = (target_lzc - computed_lzc)**2 + he_error**2 + hfd_error**2 + mean_error**2 + stdev_error**2
    return 1 - math.sqrt(total_error) / 4


input_dim = 5
model_dim = 8
num_layers = 2
num_heads = 2
num_output_features = 4
max_seq_length = 50

model = SimpleTransformerGenerator(input_dim, model_dim, num_layers, num_heads, num_output_features, max_seq_length)
model.eval()
print('initialized model')
param_shapes = get_model_param_shapes(model)
flat_params = np.concatenate([p.flatten() for p in [param.detach().cpu().numpy() for param in model.parameters()]])
print(flat_params.shape)
from cma import sampler
es = cma.CMAEvolutionStrategy(flat_params, .5, {
    'popsize': 20,
    'bounds': [-100, 100],
    'CMA_sampler': sampler.GaussDiagonalSampler # default sampler too slow due to high dimensionality
}) # sigma = .5

iteration = 0
while not es.stop():
    print('Iteration ' + str(iteration))
    target_lzc = random.random() * 100
    target_he = random.random()
    target_hfd = random.random()
    target_mean = random.random() * 100
    target_stdev = random.random() * 100
    length = random.randint(10, max_seq_length)

    solutions = es.ask()
    fitnesses = [objective(s, model, param_shapes, length, target_lzc, target_he, target_hfd, target_mean, target_stdev) for s in solutions]
    es.tell(solutions, fitnesses)
    es.disp()
    iteration += 1

result = es.result()
print("Best solution found:", result[0])
print("Best objective value:", result[1])
