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

num_features = 6
num_series_to_generate = 1000
num_blocks_per_series = 10
num_samples_per_block = 50
num_time_steps_to_taper = num_samples_per_block // 10
generated_data_series = []
for i in range(num_series_to_generate):
    print('Generating data series ' + str(i) + f' ({i / num_series_to_generate}% complete)')
    series = []
    for _ in range(num_blocks_per_series):
        # have to blend multiple series together to ensure non-stationarity
        new_block = get_random_fourier_series(num_samples_per_block, num_features)
        if len(series) > 0:
            series = blend_with_new_block(series, new_block, num_time_steps_to_taper)
        else:
            series = new_block
    metrics = {
        'lzc': lempel_ziv_complexity_continuous(series),
        'he': np.mean(hurst_exponent(series)),
        'hfd': np.mean(higuchi_fractal_dimension(series))
    }
    generated_data_series.append((metrics, series))

print('Determining which series to keep')
num_bins_per_metric = 10

all_metrics = np.array([[m['lzc'], m['he'], m['hfd']] for m, _ in generated_data_series])
lzc_vals = all_metrics[:, 0]
he_vals = all_metrics[:, 1]
hfd_vals = all_metrics[:, 2]

lzc_edges = np.linspace(np.min(lzc_vals), np.max(lzc_vals), num_bins_per_metric + 1)
he_edges = np.linspace(np.min(he_vals), np.max(he_vals), num_bins_per_metric + 1)
hfd_edges = np.linspace(np.min(hfd_vals), np.max(hfd_vals), num_bins_per_metric + 1)

series_metric_grid = {}
for metrics, series in generated_data_series:
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