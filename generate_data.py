import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from analysis import lempel_ziv_complexity_continuous, hurst_exponent, higuchi_fractal_dimension

def get_random_fourier_series(num_samples, num_features):
    time_steps = np.linspace(0, 1, num_samples, endpoint=False)
    series = np.zeros((num_samples, num_features))

    for i in range(num_features):
        num_terms = num_samples // 2
        amplitudes = np.random.randn(num_terms)
        phases = np.random.uniform(0, 2 * np.pi, num_terms)
        freqs = np.arange(1, num_terms + 1)[:, np.newaxis] # as column vector

        feature_series = np.sum(amplitudes[:, None] * np.cos(2 * np.pi * freqs * time_steps + phases[:, None]), axis=0)
        # normalize amplitude due to sensitivity of LZC to large variance
        series[:, i] = feature_series / num_terms
    return series

def blend_with_new_block(existing_series, new_block, taper_length):
    """
    linearly blend the last `taper_length` samples of existing_series with the first `taper_length` of new_block
    """
    if existing_series.shape[0] < taper_length:
        raise ValueError("Existing series is too short to blend")

    if isinstance(existing_series, torch.Tensor):
        existing_series = existing_series.detach().cpu().numpy()
    if isinstance(new_block, torch.Tensor):
        new_block = new_block.detach().cpu().numpy()
    overlap_existing = existing_series[-taper_length:]
    overlap_new = new_block[:taper_length]

    weight_existing = np.linspace(1, 0, taper_length)[:, None] # column vector
    weight_new = np.linspace(0, 1, taper_length)[:, None]
    blended_overlap = weight_existing * overlap_existing + weight_new * overlap_new

    blended_series = np.concatenate((existing_series[:-taper_length], blended_overlap, new_block[taper_length:]), axis=0)
    return blended_series

def get_cosine_positional_embeddings(seq_len, input_size):
    """
    Original cosine positional embeddings from "Attention Is All You Need"
    """
    pe = torch.zeros(seq_len, input_size)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, input_size, 2, dtype=torch.float32) * (-math.log(10000.0) / input_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class HierarchicalTransformerGenerator(nn.Module):
    def __init__(self, input_dim, global_latent_dim, num_encoder_layers, num_heads, output_dim, desired_seq_length):
        """
        Hierarchical generator that imposes a common underlying structure to the non-stationarity of the generated signal.
        Don't need to train the model because it is just used as a sequence sampler.
        """
        super(HierarchicalTransformerGenerator, self).__init__()
        self.desired_seq_length = desired_seq_length

        self.global_fc = nn.Linear(input_dim, global_latent_dim)
        positional_embeddings = get_cosine_positional_embeddings(desired_seq_length, desired_seq_length)
        self.register_buffer("time_embedding", positional_embeddings)
        self.input_proj = nn.Linear(global_latent_dim + desired_seq_length, desired_seq_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=desired_seq_length, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_proj = nn.Linear(desired_seq_length, output_dim)

    def forward(self, inputs, seq_length=None):
        """
        Generate sequence of shape (desired_seq_length, output_dim)
        """
        if not seq_length:
            seq_length = self.desired_seq_length

        global_latent = self.global_fc(inputs).unsqueeze(0).expand(seq_length, -1)

        time_emb = self.time_embedding[:seq_length, :]

        # concatenate global latent and time embedding along the feature dimension
        combined = torch.cat([global_latent, time_emb], dim=-1)
        combined = self.input_proj(combined) # shape: (seq_length, model_dim)

        # requires input shape: (seq_length, model_dim)
        combined = combined.transpose(0, 1)
        transformer_out = self.transformer_encoder(combined)
        transformer_out = transformer_out.transpose(0, 1) # back to (seq_length, model_dim)

        return self.output_proj(transformer_out) # shape: (seq_length, output_dim)

input_dim = 5
global_latent_dim = 16
num_layers = 2
num_heads = 2
num_features_per_state = 8
num_series_to_generate = 1000


num_blocks_per_series = 10
num_samples_per_block = 50
num_time_steps_to_taper = num_samples_per_block // 10
generator = HierarchicalTransformerGenerator(
    input_dim, global_latent_dim, num_layers, num_heads, num_features_per_state, num_samples_per_block
)

generated_sequences = []
for i in range(num_series_to_generate):
    print('Generating series ' + str(i + 1))
    series = []
    for _ in range(num_blocks_per_series):
        inputs = torch.randn(input_dim, dtype=torch.float32)
        # have to blend multiple series together to ensure non-stationarity
        new_block = generator.forward(inputs)
        if len(series) > 0:
            series = blend_with_new_block(series, new_block, num_time_steps_to_taper)
        else:
            series = new_block
    generated_sequences.append(series)

print('Calculating complexity metrics for generated sequences ...')
series_metrics = []
for series in generated_sequences:
    metrics = {
        'lzc': lempel_ziv_complexity_continuous(series),
        'he': np.mean(hurst_exponent(series)),
        'hfd': np.mean(higuchi_fractal_dimension(series))
    }
    series_metrics.append((metrics, series))

print('Determining which series to keep ...')
num_bins_per_metric = 10

all_metrics = np.array([[m['lzc']] for m, _ in series_metrics])
lzc_vals = all_metrics[:, 0]

lzc_edges = np.linspace(np.min(lzc_vals), np.max(lzc_vals), num_bins_per_metric + 1)
he_edges = np.linspace(0, 1, num_bins_per_metric + 1)
hfd_edges = np.linspace(0, 1, num_bins_per_metric + 1)

series_metric_grid = {}
for metrics, series in series_metrics:
    lzc, he, hfd = metrics['lzc'], metrics['he'], metrics['hfd']
    lzc_bin = np.digitize(lzc, lzc_edges) - 1
    he_bin = np.digitize(he, he_edges) - 1
    hfd_bin = np.digitize(hfd, hfd_edges) - 1
    metrics_key = (lzc_bin, he_bin, hfd_bin)
    # only store first encountered series per cell
    if metrics_key not in series_metric_grid:
        series_metric_grid[metrics_key] = (metrics, series)
        print(metrics)
        filename = f"series_cell_{metrics_key[0]}_{metrics_key[1]}_{metrics_key[2]}.npy"
        np.save(filename, series)

print('Number of grid cells covered: ', len(series_metric_grid))
print('Saving usable series to disk ...')
