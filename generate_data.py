import numpy as np
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

    overlap_existing = existing_series[-taper_length:]
    overlap_new = new_block[:taper_length]

    weight_existing = np.linspace(1, 0, taper_length)[:, None] # column vector
    weight_new = np.linspace(0, 1, taper_length)[:, None]
    blended_overlap = weight_existing * overlap_existing + weight_new * overlap_new

    blended_series = np.concatenate((existing_series[:-taper_length], blended_overlap, new_block[taper_length:]), axis=0)
    return blended_series

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HierarchicalTransformerGenerator(nn.Module):
    def __init__(self, input_dim, global_latent_dim, model_dim, num_encoder_layers, num_heads, output_dim, desired_seq_length):
        """
        Hierarchical generator that imposes a common underlying structure to the non-stationarity of the generated signal.
        Don't need to train the model because it is just used as a sequence sampler.
        """
        super(HierarchicalTransformerGenerator, self).__init__()
        self.desired_seq_length = desired_seq_length

        self.global_fc = nn.Linear(input_dim, global_latent_dim)
        self.time_embedding = nn.Parameter(torch.randn(desired_seq_length, model_dim))
        self.input_proj = nn.Linear(global_latent_dim + model_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, inputs, seq_length=None):
        """
        Generate sequence of shape (batch_size, desired_seq_length, output_dim)
        """
        if not seq_length:
            seq_length = self.desired_seq_length
        batch_size = inputs.size(0)

        global_latent = self.global_fc(inputs) # shape: (batch_size, global_latent_dim)
        # Expand and repeat to shape: (batch_size, desired_seq_length, global_latent_dim)
        global_latent = global_latent.unsqueeze(1).expand(-1, seq_length, -1)

        time_emb = self.time_embedding[:seq_length, :].unsqueeze(0).expand(batch_size, -1, -1)

        # concatenate global latent and time embedding along the feature dimension
        combined = torch.cat([global_latent, time_emb], dim=-1)
        combined = self.input_proj(combined) # shape: (batch_size, desired_seq_length, model_dim)

        # requires input shape: (desired_seq_length, batch_size, model_dim)
        combined = combined.transpose(0, 1)
        transformer_out = self.transformer_encoder(combined)
        transformer_out = transformer_out.transpose(0, 1) # back to (batch_size, desired_seq_length, model_dim)

        return self.output_proj(transformer_out) # shape: (batch_size, desired_seq_length, output_dim)

    def constrained_decode(self, input_batches, beam_width=5):
        candidates = [([], 0)]

        for t in range(self.desired_seq_length):
            new_candidates = []
            for seq, count in candidates:
                candidate_tokens = self.decode_step(input_batches, seq)
                candidate_tokens = candidate_tokens.detach().cpu()

                # if decode_step returns no candidates, keep the current candidate.
                if candidate_tokens.size(0) == 0:
                    new_candidates.append((seq, count))
                    continue

                for token in candidate_tokens:
                    # token is a vector; compute its norm to decide if it's "nonzero"
                    new_seq = seq + [token]
                    new_count = count + (1 if torch.norm(token).item() > 0 else 0)
                    new_candidates.append((new_seq, new_count))

            # if no new candidates generated, fallback to previous candidates.
            if not new_candidates:
                new_candidates = candidates

            # sort by how close the nonzero count is to the target length then prune
            new_candidates.sort(key=lambda x: abs(x[1] - self.desired_seq_length))
            candidates = new_candidates[:beam_width]

            exact_candidates = [c for c in candidates if c[1] == self.desired_seq_length]
            if exact_candidates:
                exact_candidates.sort(key=lambda x: abs(x[1] - self.desired_seq_length))
                return exact_candidates[0][0]
        return None

    def decode_step(self, inputs, seq_batch, beam_width=5, noise_scale=0.01):
        """
        Generates candidate token vectors for the next time step for each sequence in the batch.

        Parameters:
          seq_batch : list of list
              A list of sequences (length=batch_size), each being a list of tokens generated so far.

        Returns torch.Tensor of shape (batch_size, beam_width, output_dim) containing candidate tokens
        """
        batch_size = inputs.size(0)
        # Assume all sequences in the batch have the same length; if empty, current_length=0.
        current_length = len(seq_batch[0]) if seq_batch and len(seq_batch[0]) > 0 else 0
        desired_length = current_length + 1

        output = self.forward(inputs, seq_length=desired_length) # shape: (batch_size, desired_length, output_dim)
        final_tokens = output[:, -1, :] # shape: (batch_size, output_dim)

        candidates = final_tokens.unsqueeze(1).expand(batch_size, beam_width, -1) # shape: (batch_size, beam_width, output_dim)
        noise = torch.randn_like(candidates) * noise_scale
        candidates = candidates + noise
        return candidates

input_dim = 5
global_latent_dim = 8
model_dim = 16
num_layers = 2
num_heads = 2
num_features_per_state = 6
seq_length = 100
beam_width = 5
num_series_to_generate = 1000

model = HierarchicalTransformerGenerator(
    input_dim, global_latent_dim, model_dim, num_layers, num_heads, num_features_per_state, seq_length
)

batch_inputs = torch.tensor([np.random.randn(input_dim) * num_series_to_generate], dtype=torch.float32)
generated_sequences = model.constrained_decode(batch_inputs)
print(generated_sequences)

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

all_metrics = np.array([[m['lzc'], m['he'], m['hfd']] for m, _ in generated_sequences])
lzc_vals = all_metrics[:, 0]
he_vals = all_metrics[:, 1]
hfd_vals = all_metrics[:, 2]

lzc_edges = np.linspace(np.min(lzc_vals), np.max(lzc_vals), num_bins_per_metric + 1)
he_edges = np.linspace(np.min(he_vals), np.max(he_vals), num_bins_per_metric + 1)
hfd_edges = np.linspace(np.min(hfd_vals), np.max(hfd_vals), num_bins_per_metric + 1)

series_metric_grid = {}
for metrics, series in generated_sequences:
    lzc, he, hfd = metrics['lzc'], metrics['he'], metrics['hfd']
    lzc_bin = np.digitize(lzc, lzc_edges) - 1
    he_bin = np.digitize(he, he_edges) - 1
    hfd_bin = np.digitize(hfd, hfd_edges) - 1
    metrics_key = (lzc_bin, he_bin, hfd_bin)
    # only store first encountered series per cell
    if metrics_key not in series_metric_grid:
        series_metric_grid[metrics_key] = (metrics, series)

print('Number of grid cells covered: ', len(series_metric_grid))
print('Saving usable series to disk ...')
for metrics, series in series_metric_grid:
    print(metrics)
    filename = f"series_cell_{key[0]}_{key[1]}_{key[2]}.npy"
    np.save(filename, series)