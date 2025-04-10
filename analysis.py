import numpy as np
import torch
from qiskit.quantum_info import partial_trace, entropy
import antropy
from astropy.stats import bayesian_blocks

from typing import Dict
from dataclasses import dataclass, field

from data_importers import import_generated
num_states_per_block = 20
LOSS_TYPES = ['Prediction', 'Bottleneck Trash']
MODEL_TYPES = ['qae', 'qae_recurrent', 'qte', 'qte_recurrent', 'cae', 'cae_recurrent', 'cte', 'cte_recurrent']

def check_for_overfitting(training_costs, validation_costs_per_series, threshold=.15):
    def check_overfit(tc, vc, loss_type):
        overfit_ratio = (vc-tc)/tc
        if overfit_ratio > threshold:
            print(f'WARNING: Overfit likely based on {loss_type} costs (validation higher by {(100*overfit_ratio):.1f}%)')
    for tc, vc, loss_type in zip(training_costs, np.sum(validation_costs_per_series[:,1:], axis=0), LOSS_TYPES):
        check_overfit(tc, vc, loss_type)

    total_training_cost = np.sum(training_costs)
    total_validation_cost = np.sum(validation_costs_per_series[:,1:])
    check_overfit(total_training_cost, total_validation_cost, 'Total')

def multimodal_differential_entropy_per_feature(data):
    """
    data (np.ndarray): shape should be (sequence_length, num_features)
    Uses adaptive width per bin to allow for multimodality in underlying series.
    """
    entropy_per_feature = []
    num_features = data.shape[1]

    for f in range(num_features):
        feature_data = data[:, f]
        # use bayesian_blocks from astropy to adaptively determine bin edges
        # use density normalization so that the integral is one
        hist, edges = np.histogram(feature_data, bins=bayesian_blocks(feature_data), density=True)

        bin_widths = np.diff(edges) # 1D differences from the histogram edges
        bin_prob_mass = hist * bin_widths

        nonzero = bin_prob_mass > 0
        de = -np.sum(bin_prob_mass[nonzero] * np.log(bin_prob_mass[nonzero]))
        entropy_per_feature.append(de)
    return entropy_per_feature

def gaussian_total_differential_entropy(data):
    # data: numpy array of shape (num_states, num_features)
    variances = np.var(data, axis=0)
    # per latent dimension (with epsilon added to avoid log(0))
    entropies = np.log(2 * np.pi * np.e * (variances + 1E-13)) / 2
    return np.sum(entropies)

def series_gaussian_differential_entropy(series):
    # fractional brownian motion is a gaussian process but the concatenation
    # of multiple blocks each with different params means multimodality
    current_index = 0
    diff_entropy = 0.0
    while current_index < len(series):
        gaussian_block = series[current_index : current_index+num_states_per_block]
        diff_entropy += gaussian_total_differential_entropy(gaussian_block)
        current_index += num_states_per_block
    return diff_entropy

def joint_differential_entropy(data):
    """
    data (np.ndarray): shape (sequence_length, num_features), where each column is a sample
    """
    num_features = data.shape[1]

    bins = []
    for d in range(num_features):
        feature_data = samples[:, d]
        q75, q25 = np.percentile(feature_data, [75, 25])
        IQR = q75 - q25
        bin_width = 2 * IQR / np.cbrt(len(feature_data))
        nb = int(np.ceil((np.max(feature_data) - np.min(feature_data)) / bin_width))
        bins.append(nb)

    hist, edges = np.histogramdd(samples, bins=bins, density=True)

    widths_list = [np.diff(edge) for edge in edges]
    # create a meshgrid to combine bin widths across dimensions
    mesh = np.meshgrid(*widths_list, indexing='ij')
    bin_volumes = np.ones_like(mesh[0])
    for w in mesh:
        bin_volumes *= w

    bin_prob_mass = hist * bin_volumes

    nonzero = bin_prob_mass > 0
    joint_entropy = -np.sum(bin_prob_mass[nonzero] * np.log(bin_prob_mass[nonzero] / bin_volumes[nonzero]))
    return joint_entropy

def von_neumann_entropy(dm, log_base=2) -> float:
    dm_eigenvalues = np.linalg.eigvalsh(dm.data)
    dm_eigenvalues = dm_eigenvalues[dm_eigenvalues > 1e-12] # for stability
    return -np.sum(dm_eigenvalues * np.log(dm_eigenvalues)) / np.log(log_base)

def entanglement_entropy(state):
    """
    Computes the Meyer-Wallach global entanglement measure for an n-qubit pure state as
    (2/n) * sum_{r=1}^{n} [1 - Tr(dm_r^2)]
    where dm_r is the reduced density matrix of qubit r
    """
    n = state.num_qubits
    total = 0.0

    for r in range(n):
        trace_out = [i for i in range(n) if i != r]
        reduced_state = partial_trace(state, trace_out)
        # purity = Tr(dm_r^2); (1 - purity) is linear entropy of this qubit
        total += (1 - reduced_state.purity())

    # Meyer-Wallach normalizes by the number of qubits
    return (2 / n) * total

# TODO: better method for deciding number of symbols
def quantize_signal(data, num_symbols=30):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.ndim == 1:
        data_min, data_max = np.min(data), np.max(data)
        if data_max == data_min:
            quantized = np.zeros_like(data, dtype=int)
        else:
            quantized = np.floor((data - data_min) / (data_max - data_min) * num_symbols).astype(int)
            quantized[quantized == num_symbols] = num_symbols - 1 # enforce symbol bounds
        return quantized.tolist()
    elif data.ndim == 2:
        n_samples, n_features = data.shape
        quantized_features = []
        for i in range(n_features): # quantize each feature separately
            channel = data[:, i]
            channel_min, channel_max = np.min(channel), np.max(channel)
            if channel_max == channel_min:
                quanta = np.zeros_like(channel, dtype=int)
            else:
                quanta = np.floor((channel - channel_min) / (channel_max - channel_min) * num_symbols).astype(int)
                quanta[quanta == num_symbols] = num_symbols - 1
            quantized_features.append(quanta)
        quantized_features = np.stack(quantized_features, axis=1) # shape (n_samples, n_features)
        # combine features using mixed-radix encoding (treat each feature’s quantized value as a digit in a number with base equal to num_symbols)
        composite_symbols = np.sum(quantized_features * (num_symbols ** np.arange(n_features)), axis=1)
        return composite_symbols.tolist()
    else:
        raise ValueError("Data must be 1D or 2D.")

def lempel_ziv_complexity_continuous(data, num_symbols=30):
    symbol_seq = quantize_signal(data, num_symbols)
    i = 0
    complexity = 0
    while i < len(symbol_seq):
        l = 1
        # get all substrings of length l starting from index 0 up to i
        strings_so_far = {tuple(symbol_seq[k:k+l]) for k in range(i)}
        while i + l <= len(symbol_seq) and tuple(symbol_seq[i:i+l]) in strings_so_far:
            l += 1
            strings_so_far = {tuple(symbol_seq[k:k+l]) for k in range(i)}
        complexity += 1
        i += l
    return complexity

def _hurst_exponent_1d(data, window_sizes):
    """
    slope of log-log regression
    """
    RS = []
    for window in window_sizes:
        n_segments = len(data) // window
        RS_vals = []
        for i in range(n_segments):
            segment = data[i * window:(i + 1) * window]
            mean_seg = np.mean(segment)
            Y = segment - mean_seg
            cumulative_Y = np.cumsum(Y)
            R = np.max(cumulative_Y) - np.min(cumulative_Y)
            S = np.std(segment)
            if S != 0:
                RS_vals.append(R / S)
        if RS_vals:
            RS.append(np.mean(RS_vals))
    if len(RS) == 0:
        raise ValueError("No valid RS values computed; check window sizes and data.")
    logs = np.log(window_sizes[:len(RS)])
    log_RS = np.log(RS)
    slope, _ = np.polyfit(logs, log_RS, 1)
    return slope

def hurst_exponent(data):
    # compute separately for each feature (column)
    n_samples, n_features = data.shape
    hurst_vals = []
    for f_i in range(n_features):
        col = data[:, f_i]
        # use logspace for mixed local / ranged correlation structure
        window_sizes = np.unique(np.floor(np.logspace(np.log10(10), np.log10(n_samples // 2), num=20)).astype(int))
        window_sizes = window_sizes[window_sizes > 0]
        hurst_vals.append(_hurst_exponent_1d(col, window_sizes))
    return hurst_vals

def higuchi_fractal_dimension(data, kmax=10):
    n_samples, n_features = data.shape

    hfds = []
    for feature in range(n_features):
        feature_series = data[:, feature]
        try:
            hfd = antropy.higuchi_fd(feature_series, kmax=kmax)
        except Exception as e:
            hfd = np.nan
            print('HFD NaN for kmax of ', kmax, ':', feature_series.shape)
        hfds.append(hfd)
    return hfds

def per_patient(func, data, **kwargs):
    final_values = []
    for p in range(data.shape[0]):
        final_values.append(func(data[p], **kwargs))
    return final_values


if __name__ == '__main__':
    import argparse
    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Train a QTE and QAE and generate correlation plots."
    )
    parser.add_argument("datasets_directory", type=str, help="Path to the directory containing the generated datasets.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix to use for every saved file name in this run")
    parser.add_argument("--overfit_threshold", type=float, default=.15, help="Detection threshold for overfit ratio (max % for increase in validation cost vs training cost)")
    args = parser.parse_args()

    run_prefix = args.prefix if args.prefix else ''
    datasets = import_generated(args.datasets_directory)

    MODEL_STATS_CONFIG = {
        # lambda to parse rows into {series_index: individual}
        # lambda to aggregate rows into single value
        'validation_costs': {
            LOSS_TYPES[i]: (
                lambda rows: {row[0]: row[i+1] for row in rows},
                lambda rows: np.mean([row[i+1] for row in rows])
            ) for i in range(len(LOSS_TYPES))
        },
        'bottleneck_differential_entropy': (
            lambda rows: {row[0]: np.mean(row[1:]) for row in rows},
            lambda rows: np.mean([np.mean(row[1:]) for row in rows])
        ),
        'bottleneck_entanglement_entropy': (
            lambda rows: {row[0]: np.mean(row[1:]) for row in rows},
            lambda rows: np.mean([np.mean(row[1:]) for row in rows])
        ),
        'bottleneck_full_vn_entropy': (
            lambda rows: {row[0]: np.mean(row[1:]) for row in rows},
            lambda rows: np.mean([np.mean(row[1:]) for row in rows])
        )
    }
    MODEL_STATS_CONFIG['validation_costs']['Total'] = (
        lambda rows: {row[0]: sum(row[1:]) for row in rows},
        lambda rows: np.mean([sum(row[1:]) for row in rows])
    )
    SERIES_STATS_CONFIG = {
        # lambda to map series into single value
        'hurst_exponent':            lambda series: np.mean(hurst_exponent(series)),
        'lempel_ziv_complexity':     lambda series: lempel_ziv_complexity_continuous(series),
        'higuchi_fractal_dimension': lambda series: np.mean(higuchi_fractal_dimension(series)),
        'differential_entropy':      lambda series: series_gaussian_differential_entropy(series),
    }
    independent_keys = list(SERIES_STATS_CONFIG.keys())
    dependent_keys = []
    for dependent_var_key in MODEL_STATS_CONFIG.keys():
        parsers = MODEL_STATS_CONFIG[dependent_var_key]
        if isinstance(parsers, Dict):
            dependent_keys.extend(parsers.keys())
        else:
            dependent_keys.append(dependent_var_key)

    @dataclass
    class ModelStats:
        data: dict = field(default_factory=dict)

        def load(self, dataset_index, model_type, dsets_dir, run_prefix):
            for key in MODEL_STATS_CONFIG.keys():
                if model_type.startswith('c') and ('vn' in key or 'entangle' in key):
                    continue
                filepath = os.path.join(dsets_dir, f'{run_prefix}dataset{dataset_index}_{model_type}_{key}.npy')
                self.data[key] = np.load(filepath)
                if key == 'validation_costs':
                    for k in MODEL_STATS_CONFIG[key].keys():
                        self.data[k] = self.data[key]
            filepath = os.path.join(dsets_dir, f'{run_prefix}dataset{dataset_index}_{model_type}_cost_history.npy')
            self.data['cost_history'] = np.load(filepath)

    @dataclass
    class SeriesStats:
        data: dict = field(default_factory=dict)

        def compute(self, series):
            for key, func in SERIES_STATS_CONFIG.items():
                self.data[key] = func(series)

    # load model statistics for each dataset and model type (qae and qte)
    stats_per_model = {}
    for d_i in datasets:
        for model_type in MODEL_TYPES:
            print(f'Loading {model_type} model statistics for dataset {d_i}')
            stats = ModelStats()
            stats.load(d_i, model_type, args.datasets_directory, run_prefix)
            stats_per_model[(d_i, model_type)] = stats
            check_for_overfitting(stats.data['cost_history'][-1], stats.data['validation_costs'], args.overfit_threshold)

    # compute complexity metrics for all validation series
    dataset_series_stats = {}
    for d_i, (training_series, validation_series) in datasets.items():
        for s_i, series in validation_series:
            num_features = len(series[0])
            print(f'Computing complexity metrics for dataset {d_i} series {s_i} ({num_features} features)')
            series_stats = SeriesStats()
            series_stats.compute(series)
            # store as tuple (s_i, series_stats) for later annotation
            dataset_series_stats.setdefault(d_i, []).append((s_i, series_stats))

    individual_plot_data = {i_key: {d_key: {model: [] for model in MODEL_TYPES} for d_key in dependent_keys} for i_key in independent_keys}
    aggregated_plot_data = {i_key: {d_key: {model: [] for model in MODEL_TYPES} for d_key in dependent_keys} for i_key in independent_keys}

    for (d_i, model_type), m_stats in stats_per_model.items():
        dependent_individual = {}
        dependent_aggregated = {}
        for dependent_var_key in MODEL_STATS_CONFIG.keys():
            parsers = MODEL_STATS_CONFIG[dependent_var_key]
            if isinstance(parsers, Dict): # allow multiple values from same file
                for key, (parse_individual, parse_aggregated) in parsers.items():
                    dependent_individual[key] = parse_individual(m_stats.data[key])
                    dependent_aggregated[key] = parse_aggregated(m_stats.data[key])
            else:
                (parse_individual, parse_aggregated) = parsers
                dependent_individual[dependent_var_key] = parse_individual(m_stats.data[key])
                dependent_aggregated[dependent_var_key] = parse_aggregated(m_stats.data[key])

        # Get the series stats for the current dataset.
        series_stats_list = dataset_series_stats[d_i]

        num_entropy_warnings = 0
        for s_i, s_stats in series_stats_list:
            for i_key in independent_keys:
                for d_key in dependent_keys:
                    individual_plot_data[i_key][d_key][model_type].append((s_stats.data[i_key], dependent_individual[d_key][s_i], d_i, s_i))
                    aggregated_plot_data[i_key][d_key][model_type].append((s_stats.data[i_key], dependent_aggregated[d_key], d_i, s_i))

            ent_entropy = dependent_individual['bottleneck_entanglement_entropy'].get(s_i)
            full_vn = dependent_individual['bottleneck_full_vn_entropy'].get(s_i)
            if ent_entropy is None:
                raise Exception(f'ERROR: missing entanglement entropy for dataset {d_i} series {s_i} {model_type} model')
            if full_vn is None:
                raise Exception(f'ERROR: missing full VN entropy for dataset {d_i} series {s_i} {model_type} model')
            if full_vn < ent_entropy - 1E-15:
                print(f'WARNING: full VN entropy < entanglement entropy by {abs(full_vn - ent_entropy)} for dataset {d_i} series {s_i}')
                num_entropy_warnings += 1
        if num_entropy_warnings > 0:
            percent = num_entropy_warnings / len()
            print(f'{num_entropy_warnings} total warnings ({percent}%) for unexpected quantum entropy relationship w/ dataset {d_i} {model_type}')

    color_map = plt.get_cmap('viridis')
    spacing = np.linspace(0, 1, len(MODEL_TYPES))
    colors = color_map(spacing)
    colors = {model: color for (model, color) in zip(MODEL_TYPES, colors)}

    def plot_model_data_and_save(data, label_prefix, annotation_func, color):
        x_vals = [d[0] for d in data]
        y_vals = [d[1] for d in data]
        plt.scatter(x_vals, y_vals, color=color, label=label_prefix)
        coeffs = np.polyfit(x_vals, y_vals, 1)
        slope = float(coeffs[0])
        poly_eqn = np.poly1d(coeffs)
        x_fit = np.linspace(min(x_vals), max(x_vals), 100)
        plt.plot(x_fit, poly_eqn(x_fit), color=color, linestyle='--', label=f"{label_prefix} (slope={slope:.5f})")
        for xi, yi, d in zip(x_vals, y_vals, data):
            plt.annotate(annotation_func(d), (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=5)

    def plot_data(data_dict, x_label, y_label, title, filename, annotation_fn):
        plt.figure()
        if isinstance(next(iter(data_dict.values())), dict):
            for loss_type, model_data in data_dict.items():
                for model_type, data in model_data.items():
                    label_prefix = f"{model_type.upper()}-{loss_type}"
                    plot_model_data_and_save(data, label_prefix, annotation_fn, colors[model_type])
        else:
            for model_type, data in data_dict.items():
                label_prefix = model_type.upper()
                plot_model_data_and_save(data, label_prefix, annotation_fn, colors[model_type])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        save_path = os.path.join(args.datasets_directory, run_prefix + filename)
        plt.savefig(save_path)

    def plot_scatter_individual(data_dict, x_label, y_label, title, filename):
        plot_data(data_dict, x_label, y_label, title, filename, lambda d: '')
        print(f"Saved individual series plot to {filename}")

    def plot_scatter_aggregated(data_dict, x_label, y_label, title, filename):
        plot_data(data_dict, x_label, y_label, title, filename, lambda d: f"D{d[2]}")
        print(f"Saved aggregated plot to {filename}")

    # TODO: need to plot classical and quantum losses sepoarately due to scale differences
    for i_key in independent_keys:
        for d_key in dependent_keys:
            x_label = i_key.replace('_', ' ').title()
            y_label = d_key.replace('_', ' ').title()
            print(f'Plotting individual data for {y_label} vs {x_label}')
            plot_scatter_individual(
                data_dict=individual_plot_data[i_key][d_key],
                x_label=f'{x_label} (Individual)',
                y_label=f'{y_label} (Validation)',
                title=f'{x_label} vs {y_label} Per Series',
                filename=f'{run_prefix}{d_key}_vs_{i_key}_individual.png'
            )

            print(f'Plotting aggregated data for {y_label} vs {x_label}')
            plot_scatter_aggregated(
                data_dict=aggregated_plot_data[i_key][d_key],
                x_label=f'Mean {x_label}',
                y_label=f'{y_label} (Validation)',
                title=f'Mean {x_label} vs {y_label} Per Dataset',
                filename=f'{run_prefix}{d_key}_vs_{i_key}_aggregated.png'
            )

    # TODO: load and plot cost part history per model (all on same plot) w/ each cost part as separate plot

    plt.show()
