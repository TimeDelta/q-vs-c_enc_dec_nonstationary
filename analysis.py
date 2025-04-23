import numpy as np
import numpy.fft as fft
import torch
from qiskit.quantum_info import partial_trace, entropy
import antropy
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN
from pyentrp import entropy

import math
from typing import Dict
import random
from dataclasses import dataclass, field
import colorsys
import os

from data_importers import import_generated

num_states_per_block = 5
LOSS_TYPES = ['Prediction', 'Bottleneck Trash']
MODEL_TYPES = ['qae', 'qrae', 'qte', 'qrte', 'cae', 'crae', 'cte', 'crte']
# q = quantum; c = classical
# r = recurrent
# ae = auto-encoder; te = transition encoder (auto-regressive)

def check_for_overfitting(training_costs, validation_costs, threshold=.15):
    """
    each cost part must already be normalized by number of series in that partition
    """
    def check_overfit(tc, vc, loss_type):
        overfit_ratio = (vc-tc)/tc
        if overfit_ratio > threshold:
            print(f'WARNING: Overfit likely based on {loss_type} costs (validation higher by {(100*overfit_ratio):.1f}%)')
            return True
        return False
    for tc, vc, loss_type in zip(training_costs, validation_costs, LOSS_TYPES):
        check_overfit(tc, vc, loss_type)

    total_training_cost = np.sum(training_costs)
    total_validation_cost = np.sum(validation_costs)
    return check_overfit(total_training_cost, total_validation_cost, 'Total')

def differential_entropy(data, quantizer):
    discrete_signal = quantizer(data)
    hist, _ = np.histogram(discrete_signal, density=True)
    nonzero = hist > 0
    return -np.sum(hist[nonzero] * np.log(hist[nonzero])) / np.log(len(np.unique(discrete_signal)))

def differential_entropy_per_feature(data, quantizer):
    """
    data (np.ndarray): shape should be (sequence_length, num_features)
    Uses adaptive width per bin to allow for multimodality in underlying series.
    """
    num_features = data.shape[1]
    return [differential_entropy(data[:, f], quantizer) for f in range(num_features)]

def von_neumann_entropy(dm, log_base=2) -> float:
    dm_eigenvalues = np.linalg.eigvalsh(dm.data)
    dm_eigenvalues = dm_eigenvalues[dm_eigenvalues > 1e-12] # for stability
    return -np.sum(dm_eigenvalues * np.log(dm_eigenvalues)) / np.log(log_base)

def meyer_wallach_global_entanglement(state):
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

def quantize_signal_equal_feature_bins(data):
    num_symbols = data.shape[0] // 10
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

def quantize_signal_bayesian_block_feature_bins(data):
    if data.ndim == 1:
        edges = bayesian_blocks(data[:, i])
        quantized = np.digitize(data[:, i], edges) - 1
        return quantized.tolist()
    elif data.ndim == 2:
        n_samples, n_features = data.shape
        quantized_features = []
        bases = []

        # quantize each feature and record its number of bins
        for i in range(n_features):
            edges = bayesian_blocks(data[:, i])
            quantized_features.append(np.digitize(data[:, i], edges) - 1)
            bases.append(len(edges))

        quantized_features = np.stack(quantized_features, axis=1)
        bases = np.array(bases, dtype=int)

        # compute mixed‑radix weights: product of previous bases
        # weights[0] = 1, weights[i] = prod(bases[:i])
        weights = np.cumprod(np.concatenate(([1], bases[:-1])))
        composite_symbols = np.sum(quantized_features * weights, axis=1)
        return composite_symbols.tolist()
    else:
        raise ValueError("Data must be 1D or 2D.")

def quantize_signal_hdbscan(data):
    n_samples, _ = data.shape
    nbrs = NearestNeighbors(n_neighbors=max(n_samples//20, 2)).fit(data)
    interpoint_distances, _ = nbrs.kneighbors(data)
    interpoint_distances = interpoint_distances[:,1:] # ignore the 0s
    mean = np.mean(interpoint_distances)
    std_dev = np.std(interpoint_distances)

    return HDBSCAN(
        min_cluster_size=2,
        min_samples=None,
        cluster_selection_epsilon=float(mean + std_dev),
        cluster_selection_method='leaf'
    ).fit_predict(data)

def lempel_ziv_complexity_continuous(data, quantizer):
    symbol_seq = quantizer(data)
    phrase_start = 0
    complexity = 0

    while phrase_start < len(symbol_seq):
        phrase_length = 1
        while True:
            # so that a substring of target phrase length sits entirely before phrase_start
            max_prefix_start = phrase_start - phrase_length + 1

            if max_prefix_start > 0:
                # all substrings of phrase_length in the prefix [0 : phrase_start]
                previous_substrings = {
                    tuple(symbol_seq[k : k + phrase_length])
                    for k in range(max_prefix_start)
                }
            else:
                previous_substrings = set()

            end_of_candidate = phrase_start + phrase_length

            # does it still perfectly match something in the prefix?
            if (
                end_of_candidate <= len(symbol_seq)
                and tuple(symbol_seq[phrase_start : end_of_candidate])
                in previous_substrings
            ):
                phrase_length += 1
                continue
            else:
                break
        complexity += 1
        phrase_start += phrase_length
    alphabet_size = len(np.unique(symbol_seq))
    max_complexity = len(symbol_seq) / np.emath.logn(alphabet_size, len(symbol_seq))
    return complexity / max_complexity

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

def optimized_multiscale_permutation_entropy(time_series) -> float:
    """
    Compute the mean Multiscale Permutation Entropy (MPE) over:
      - orders m = 2 and 3 (averaged)
      - delays swept from min_delay to max_delay (averaged)
      - scale fixed to 3
    """
    # Orders to average over (maintains N ≫ m! guideline)
    center_delay = np.correlate(time_series, time_series, model='full')
    min_delay = max(1, int(min(center_delay-sequence_length/20, 0.8*center_delay)))
    max_delay = max(int(center_delay + span), int(1.2 * center_delay))
    delays = list(range(min_delay, max_delay + 1))

    scale = 3
    mpe_vals = []
    for order in [2, 3]:
        for delay in delays:
            mpe = entropy.multiscale_permutation_entropy(time_series, order, delay, scale) / np.log2(math.factorial(order))
            mpe_vals.append(mpe.mean())
    return float(np.mean(mpe_vals))

def run_analysis(datasets, data_dir, overfit_threshold=.15, quantizer='bayesian_block', quantum_bottleneck_feature='z'):
    if quantizer == 'bayesian_block':
        quantizer = quantize_signal_bayesian_block_feature_bins
    elif quantizer == 'hdbscan':
        quantizer = quantize_signal_hdbscan
    elif quantizer == 'equal_width':
        quantizer = quantize_signal_equal_feature_bins
    else:
        raise Exception('Unknown quantizer')
    num_training_series = len(next(iter(datasets.values()))[0])
    num_validation_series = len(next(iter(datasets.values()))[1])

    # lambda to parse each individual series into a single value
    #   * !! series index gets stored as a complex value and sometimes has rounding errors
    # lambda to aggregate all series of a dataset into a single value
    MODEL_MEAN_MEAN_STAT_LAMBDAS = (
        lambda rows: {int(round(float(row[0]))): np.mean(row[1:]) for row in rows},
        lambda rows: np.mean([np.mean(row[1:]) for row in rows])
    )
    MODEL_MEAN_SUM_STAT_LAMBDAS = (
        lambda rows: {int(round(float(row[0]))): np.sum(row[1:]) for row in rows},
        lambda rows: np.mean([np.sum(row[1:]) for row in rows])
    )
    MODEL_MEAN_SINGLE_VALUE_STAT_LAMBDAS = (
        lambda rows: {int(round(float(row[0]))): row[1] for row in rows},
        lambda rows: np.mean([row[1] for row in rows])
    )
    MODEL_STATS_CONFIG = {
        # cost_history gets added separately
        'validation_costs': {
            LOSS_TYPES[i]: (
                lambda rows: {int(round(float(row[0]))): row[i+1] for row in rows},
                lambda rows: np.mean([row[i+1] for row in rows])
            ) for i in range(len(LOSS_TYPES))
        },
        'bottleneck_de': MODEL_MEAN_SUM_STAT_LAMBDAS,
        'bottleneck_mw_global_entanglement': MODEL_MEAN_SUM_STAT_LAMBDAS,
        'bottleneck_full_vn_entropy': MODEL_MEAN_SUM_STAT_LAMBDAS,
        'bottleneck_lzc': MODEL_MEAN_SINGLE_VALUE_STAT_LAMBDAS,
        'bottleneck_he': MODEL_MEAN_MEAN_STAT_LAMBDAS,
        'bottleneck_hfd': MODEL_MEAN_MEAN_STAT_LAMBDAS,
    }
    MODEL_STATS_CONFIG['validation_costs']['Total'] = (
        # series index gets stored as a complex value and sometimes has rounding errors
        lambda rows: {int(round(float(row[0]))): sum(row[1:]) for row in rows},
        lambda rows: np.mean([sum(row[1:]) for row in rows])
    )
    SERIES_STATS_CONFIG = {
        # lambda to map series into single value
        'hurst_exponent':            lambda series: np.mean(hurst_exponent(series)),
        'lempel_ziv_complexity':     lambda series: lempel_ziv_complexity_continuous(series, quantizer),
        'higuchi_fractal_dimension': lambda series: np.mean(higuchi_fractal_dimension(series)),
        'differential_entropy':      lambda series: differential_entropy(series, quantizer),
    }
    MAPPINGS_TO_PLOT = { # {series_attribute: [model_attribute]}
        # ALWAYS PUT METRIC SELF-COMPARISON IN FIRST INDEX
        'hurst_exponent': ['bottleneck_he', 'bottleneck_mw_global_entanglement', 'bottleneck_full_vn_entropy'],
        'lempel_ziv_complexity': ['bottleneck_lzc', 'bottleneck_mw_global_entanglement', 'bottleneck_full_vn_entropy'],
        'higuchi_fractal_dimension': ['bottleneck_hfd', 'bottleneck_mw_global_entanglement', 'bottleneck_full_vn_entropy'],
        'differential_entropy': ['bottleneck_de', 'bottleneck_mw_global_entanglement', 'bottleneck_full_vn_entropy'],
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
                if key in ['bottleneck_lzc', 'bottleneck_he', 'bottleneck_hfd', 'bottleneck_de']:
                    continue # calculated in this analysis
                filepath = os.path.join(dsets_dir, f'{run_prefix}dataset{dataset_index}_{model_type}_{key}.npy')
                self.data[key] = np.load(filepath)
                if key == 'validation_costs':
                    for k in MODEL_STATS_CONFIG[key].keys():
                        self.data[k] = self.data[key]

            filepath = os.path.join(dsets_dir, f'{run_prefix}dataset{dataset_index}_{model_type}_cost_history.npy')
            self.data['cost_history'] = np.load(filepath)
            filepath = os.path.join(dsets_dir, f'{run_prefix}dataset{dataset_index}_{model_type}_gradient_norms.npy')
            self.data['gradient_norm_history'] = np.load(filepath)

            if model_type.startswith('q'):
                filepath = os.path.join(dsets_dir, f'{run_prefix}dataset{dataset_index}_{model_type}_{quantum_bottleneck_feature}_bottlenecks.npy')
            elif model_type.startswith('c'):
                filepath = os.path.join(dsets_dir, f'{run_prefix}dataset{dataset_index}_{model_type}_bottlenecks.npy')
            else:
                raise Exception(f"Unknown model type ({model_type}): don't know how to get bottlenecks")
            bottlenecks = np.load(filepath)
            self.data['bottlenecks'] = bottlenecks
            de, lzc, he, hfd = [], [], [], []
            for series_bottlenecks in bottlenecks: # start with state full of associated series index
                s_i = int(np.real(series_bottlenecks[0][0]))
                de.append([s_i, differential_entropy(series_bottlenecks[1:], quantizer)])
                lzc.append([s_i, lempel_ziv_complexity_continuous(series_bottlenecks[1:], quantizer)])
                he.append([s_i, np.mean(hurst_exponent(series_bottlenecks[1:]))])
                hfd.append([s_i, np.mean(higuchi_fractal_dimension(series_bottlenecks[1:]))])
            self.data['bottleneck_de'] = np.array(de)
            self.data['bottleneck_lzc'] = np.array(lzc)
            self.data['bottleneck_he'] = np.array(he)
            self.data['bottleneck_hfd'] = np.array(hfd)

    @dataclass
    class SeriesStats:
        data: dict = field(default_factory=dict)

        def compute(self, series):
            for key, func in SERIES_STATS_CONFIG.items():
                value = func(series)
                if np.isnan(value) or math.isnan(value):
                    raise Exception(f'{key} is NaN')
                self.data[key] = value

    bar = '=-=-=-=-=-=-=-=-=-=-=-=-=-=-='

    max_hfd = 0.0
    # load model statistics for each (dataset, model_type)
    stats_per_model = {}
    for d_i in datasets:
        dataset_stats = {}
        skip_dset = False # for testing purposes
        for model_type in MODEL_TYPES:
            print(f'Loading {model_type} model statistics for dataset {d_i}')
            stats = ModelStats()
            try:
                stats.load(d_i, model_type, data_dir, run_prefix)
                dataset_stats[model_type] = stats
                if stats.data['bottleneck_hfd'] > max_hfd:
                    max_hfd = stats.data['bottleneck_hfd']
                mean_training_costs = stats.data['cost_history'][-1] / num_training_series
                mean_validation_costs = np.sum(stats.data['validation_costs'][:,1:], axis=0) / num_validation_series
                check_for_overfitting(mean_training_costs, mean_validation_costs, overfit_threshold)
            except Exception as e:
                if args.test:
                    print('  skipping due to exception: ' + str(e))
                    skip_dset = True # skip entire dataset to ensure homogenous array dimensions
                    break
                else:
                    raise e
        if not skip_dset:
            for model_type in MODEL_TYPES:
                stats_per_model[(d_i, model_type)] = dataset_stats[model_type]

    print('\n\n\n' + bar)

    # compute complexity metrics for all validation series
    dataset_series_stats = {}
    for d_i, (training_series, validation_series) in datasets.items():
        for s_i, series in validation_series:
            num_features = len(series[0])
            series_index = int(round(float(s_i))) # gets stored as a complex value and sometimes has rounding errors
            print(f'Computing complexity metrics for dataset {d_i} series {series_index} ({num_features} features)')
            series_stats = SeriesStats()
            try:
                series_stats.compute(series)
                if series_stats.data['higuchi_fractal_dimension'] > max_hfd:
                    max_hfd = series_stats.data['higuchi_fractal_dimension']
            except Exception as e:
                print(series)
                raise e
            # store as tuple (s_i, series_stats) for later annotation
            dataset_series_stats.setdefault(d_i, []).append((series_index, series_stats))

    # final HFD value normalization step
    for (d_i, model_type), dataset_stats in stats_per_model.items():
        dataset_stats.data['bottleneck_hfd'] /= max_hfd
    for d_i, (s_i, series_stats) in dataset_series_stats.items():
        series_stats['higuchi_fractal_dimension'] /= max_hfd

    individual_plot_data = {i_key: {d_key: {model: [] for model in MODEL_TYPES} for d_key in dependent_keys} for i_key in independent_keys}
    aggregated_plot_data = {i_key: {d_key: {model: [] for model in MODEL_TYPES} for d_key in dependent_keys} for i_key in independent_keys}

    for (d_i, model_type), m_stats in stats_per_model.items():
        dependent_individual = {}
        dependent_aggregated = {}
        for dependent_var_key in MODEL_STATS_CONFIG.keys():
            if model_type.startswith('c') and ('entangle' in dependent_var_key or 'vn' in dependent_var_key):
                continue

            parsers = MODEL_STATS_CONFIG[dependent_var_key]
            if isinstance(parsers, Dict): # allow multiple values from same file
                for key, (parse_individual, parse_aggregated) in parsers.items():
                    dependent_individual[key] = parse_individual(m_stats.data[key])
                    dependent_aggregated[key] = parse_aggregated(m_stats.data[key])
            else:
                (parse_individual, parse_aggregated) = parsers
                dependent_individual[dependent_var_key] = parse_individual(m_stats.data[dependent_var_key])
                dependent_aggregated[dependent_var_key] = parse_aggregated(m_stats.data[dependent_var_key])

        # Get the series stats for the current dataset.
        series_stats_list = dataset_series_stats[d_i]

        num_entropy_warnings = 0
        for s_i, s_stats in series_stats_list:
            for i_key in independent_keys:
                for d_key in dependent_keys:
                    if model_type.startswith('c') and ('entangle' in d_key or 'vn' in d_key):
                        continue
                    individual_plot_data[i_key][d_key][model_type].append((s_stats.data[i_key], dependent_individual[d_key][s_i], d_i, s_i))
                    aggregated_plot_data[i_key][d_key][model_type].append((s_stats.data[i_key], dependent_aggregated[d_key], d_i, s_i))

    """
    Generate 'number_of_colors' distinct colors using the HSV (HSB) color space.
    Each color is evenly spaced in hue, with a random brightness between .4 and .8
    * Based on my code at https://github.com/TimeDelta/introspective/blob/32af5154e2c6bd0bc4c7196d44f76076281abebc/Introspective/UI/Graphing/GraphDataGenerators/XYGraphDataGenerator.swift#L299
    """
    colormap_colors = []
    for color_index in range(len(MODEL_TYPES)):
        hue_value = float(color_index) / len(MODEL_TYPES) # even spacing across the hue spectrum

        brightness_range_low = 0.4
        brightness_range_high = 0.8

        brightness_value = random.uniform(brightness_range_low, brightness_range_high)
        saturation_value = 1.0

        red_value, green_value, blue_value = colorsys.hsv_to_rgb(hue_value, saturation_value, brightness_value)
        colormap_colors.append((red_value, green_value, blue_value))
    colors = {model: color for (model, color) in zip(MODEL_TYPES, colormap_colors)}

    def plot_data_and_save(data_dict, x_label, y_label, title, filename):
        def plot_model_data(data, label_prefix, color):
            x_vals = [d[0] for d in data]
            y_vals = [d[1] for d in data]
            plt.scatter(x_vals, y_vals, color=color, s=5)

            try:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                slope = float(coeffs[0])
                poly_eqn = np.poly1d(coeffs)
                x_fit = np.linspace(min(x_vals), max(x_vals), 100)
                plt.plot(x_fit, poly_eqn(x_fit), color=color, linestyle='--', label=f'{label_prefix} (slope={slope:.5f})')
            except np.linalg.LinAlgError as e:
                for point in zip(x_vals, y_vals):
                    print(point)
                print('Line of best fit failure for above series')
                raise e

        plt.figure()
        if isinstance(next(iter(data_dict.values())), dict):
            for loss_type, model_data in data_dict.items():
                for model_type, data in model_data.items():
                    if len(data) == 0:
                        continue
                    label_prefix = f"{model_type.upper()}-{loss_type}"
                    plot_model_data(data, label_prefix, colors[model_type])
        else:
            for model_type, data in data_dict.items():
                if len(data) == 0:
                    continue
                label_prefix = model_type.upper()
                plot_model_data(data, label_prefix, colors[model_type])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        save_path = os.path.join(data_dir, run_prefix + filename)
        plt.savefig(save_path)

    print('\n\n\n' + bar)

    # TODO: need to plot classical and quantum losses sepoarately due to scale differences
    for (i_key, dependent_keys) in MAPPINGS_TO_PLOT.items():
        for d_key in dependent_keys:
            x_label = i_key.replace('_', ' ').title()
            y_label = d_key.replace('_', ' ').title()
            title = f'{y_label} vs {x_label}'
            if d_key == dependent_keys[0]:
                y_label = f'Bottleneck {x_label}'
                title = x_label
            print(f'Plotting individual data for {y_label} vs {x_label}')
            plot_data_and_save(
                data_dict=individual_plot_data[i_key][d_key],
                x_label=f'Series {x_label}',
                y_label=y_label,
                title=f'{title} Per Series',
                filename=f'{run_prefix}{d_key}_vs_{i_key}_individual.png'
            )

            x_label = f'Mean {x_label}'
            if d_key == dependent_keys[0]:
                title = x_label
            print(f'Plotting aggregated data for {y_label} vs {x_label}')
            plot_data_and_save(
                data_dict=aggregated_plot_data[i_key][d_key],
                x_label=f'Dataset {x_label}',
                y_label=y_label,
                title=f'{title} Per Dataset',
                filename=f'{run_prefix}{d_key}_vs_{i_key}_aggregated.png'
            )



    ####################################
    # Training Metric History Analysis #
    ####################################

    # randomly sample 10% datasets to plot individual histories per model type to avoid too much clutter
    # choose indices outside of function to ensure consistent plotting
    individual_datasets_to_plot = list(set(d_i for (d_i, model_type) in stats_per_model.keys()))
    random.shuffle(individual_datasets_to_plot)
    individual_datasets_to_plot = individual_datasets_to_plot[:max(len(individual_datasets_to_plot)//10, 1)]

    def get_mean_training_metric_history(metric_key):
        # aggregate history for each model type across datasets
        history_by_model_type = {m: [] for m in MODEL_TYPES}
        for (dataset_index, model_type), model_stats in stats_per_model.items():
            history_by_model_type[model_type].append(model_stats.data[metric_key])
        mean_history_by_model_type = {}
        for (model_type, history_list) in history_by_model_type.items():
            history_arrays = np.array(history_list) # shape: (num_runs, num_epochs)
            if history_arrays.ndim >= 2 and history_arrays.shape[0] > 1:
                mean_history = history_arrays.mean(axis=0)
            elif history_arrays.shape[0] == 1:
                mean_history = history_arrays[0]
            elif args.test:
                print(f'WARNING: Missing {model_type} {metric_key}')
                continue
            else:
                raise Exception(f'Unable to find any {model_type} model {metric_key}')
            mean_history_by_model_type[model_type] = mean_history
        return mean_history_by_model_type

    def plot_training_metric_histories(metric_history_lambda, metric_description, mean_history_by_model_type):
        plt.figure()
        model_types_plotted = set()
        for (d_i, model_type), model_stats in stats_per_model.items():
            if d_i not in individual_datasets_to_plot:
                continue
            history = metric_history_lambda(model_stats.data)
            if model_type in model_types_plotted:
                plt.plot(range(len(history)), history, color=colors[model_type])
            else:
                plt.plot(range(len(history)), history, label=model_type.upper(), color=colors[model_type])
            model_types_plotted.add(model_type)
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric_description}')
        plt.title(f'Sample {metric_description} Histories')
        plt.legend()
        save_filepath = os.path.join(data_dir, f'{run_prefix}_sample_{metric_description.replace(" ", "_").lower()}.png')
        plt.savefig(save_filepath)
        print(f'Saved sample {metric_description} histories plot to {save_filepath}')

        plt.figure()
        model_types_plotted = set()
        for model_type, history in mean_history_by_model_type.items():
            if model_type in model_types_plotted:
                plt.plot(range(len(history)), history, color=colors[model_type])
            else:
                plt.plot(range(len(history)), history, label=model_type.upper(), color=colors[model_type])
            model_types_plotted.add(model_type)
        plt.xlabel('Epoch')
        plt.ylabel(f'Mean {metric_description}')
        plt.title(f'Mean {metric_description} History per Model Type')
        plt.legend()

        save_filepath = os.path.join(data_dir, f'{run_prefix}_mean_{metric_description.replace(" ", "_").lower()}.png')
        plt.savefig(save_filepath)
        print(f'Saved mean {metric_description} history plot to {save_filepath}')

    mean_cost_history_per_model_type = get_mean_training_metric_history('cost_history')
    sample = next(iter(mean_cost_history_per_model_type.values()))
    num_epochs, num_loss_types = sample.shape

    # plot cost history for each cost part separately
    for cost_part_index in range(num_loss_types):
        loss_label = LOSS_TYPES[cost_part_index]
        metric_description = f'{loss_label} Loss'
        mean_history_per_model_type = {k: v[:, cost_part_index] for k,v in mean_cost_history_per_model_type.items()}
        plot_training_metric_histories(lambda data: data['cost_history'][:, cost_part_index], metric_description, mean_history_per_model_type)

    mean_gradient_norm_history_per_model_type = get_mean_training_metric_history('gradient_norm_history')
    plot_training_metric_histories(lambda data: data['gradient_norm_history'], 'Gradient Norms', mean_gradient_norm_history_per_model_type)
    plt.show()

    # precompute and cache 1st/2nd derivatives, FFTs per model type per metric history
    costs_cache = {
        'history': {},
        'first_derivatives': {},
        'second_derivatives': {},
        'fft_first': {},
        'fft_second': {},
    }
    gradient_norms_cache = {
        'history': {},
        'first_derivatives': {},
        'second_derivatives': {},
        'fft_first': {},
        'fft_second': {},
    }
    for model_type in MODEL_TYPES:
        cost_history = mean_cost_history_per_model_type[model_type] # shape: (num_epochs, num_loss_types)
        gradient_norm_history = mean_gradient_norm_history_per_model_type[model_type] # shape: (num_epochs)

        cost_first_derivatives = np.gradient(cost_history, axis=0)
        model_fft_cost_first = {}
        for cost_part_index in range(num_loss_types):
            model_fft_cost_first[cost_part_index] = fft.fft(cost_first_derivatives[:, cost_part_index])
        gradient_first_derivatives = np.gradient(gradient_norm_history)
        model_fft_gradient_first = fft.fft(gradient_first_derivatives)

        cost_second_derivatives = np.gradient(cost_first_derivatives, axis=0)
        model_fft_cost_second = {}
        for cost_part_index in range(num_loss_types):
            model_fft_cost_second[cost_part_index] = fft.fft(cost_second_derivatives[:, cost_part_index])
        gradient_second_derivatives = np.gradient(gradient_first_derivatives) # gradients of gradients of gradient norms lol
        model_fft_gradient_second = fft.fft(gradient_second_derivatives)

        frequencies = fft.fftfreq(cost_history.shape[0]) # freqs available based on nyquist sampling (from epoch indices)
        costs_cache['history'][model_type] = cost_history
        costs_cache['first_derivatives'][model_type] = cost_first_derivatives
        costs_cache['second_derivatives'][model_type] = cost_second_derivatives
        costs_cache['fft_first'][model_type] = model_fft_cost_first
        costs_cache['fft_second'][model_type] = model_fft_cost_second
        gradient_norms_cache['history'][model_type] = gradient_norm_history
        gradient_norms_cache['first_derivatives'][model_type] = gradient_first_derivatives
        gradient_norms_cache['second_derivatives'][model_type] = gradient_second_derivatives
        gradient_norms_cache['fft_first'][model_type] = model_fft_gradient_first
        gradient_norms_cache['fft_second'][model_type] = model_fft_gradient_second

    model_types_header = '\t'.join([m.upper().replace('_', ' ') for m in MODEL_TYPES])
    def analyze_history(cache, label):
        histories = cache['history']
        first_derivatives = cache['first_derivatives']
        second_derivatives = cache['second_derivatives']
        print(f'\n\n\n{bar}\n{label}:\n{bar}')
        print('  Pairwise Pearson Correlations:')
        print('\t\t' + model_types_header)
        for i, model_i in enumerate(MODEL_TYPES):
            row_values = [model_i.upper()]
            series_i = histories[model_i]
            for j, model_j in enumerate(MODEL_TYPES):
                series_j = histories[model_j]
                corr_coeff = np.corrcoef(series_i, series_j)[0, 1]
                row_values.append(f'{corr_coeff:.3f}')
            print('\t' + '\t'.join(row_values))

        # mean absolute 1st & 2nd derivatives
        first_derivatives_means = {}
        second_derivatives_means = {}
        for model in MODEL_TYPES:
            first_derivatives_means[model] = np.mean(np.abs(first_derivatives[model]))
            second_derivatives_means[model] = np.mean(np.abs(second_derivatives[model]))
        print('  Mean Absolute 1st Derivative per Model Type:')
        for model_type in MODEL_TYPES:
            print(f'    {model_type.upper()}: {first_derivatives_means[model_type]:.10f}')
        print('  Mean Absolute 2nd Derivative per Model Type:')
        for model_type in MODEL_TYPES:
            print(f'    {model_type.upper()}: {second_derivatives_means[model_type]:.10f}')

        def compute_and_print_cross_corr_similarity(metric_history_per_type):
            mean_centered_values = {}
            for model_type in MODEL_TYPES:
                values = metric_history_per_type[model_type]
                mean_centered_values[model_type] = values - np.mean(values)
            print('\t\t' + model_types_header)
            for i, model_i in enumerate(MODEL_TYPES):
                row_values = [model_i.upper()]
                for j, model_j in enumerate(MODEL_TYPES):
                    x = mean_centered_values[model_i]
                    y = mean_centered_values[model_j]
                    cross_corr = np.correlate(x, y, mode='full')
                    # normalize by product of norms to get similarity metric
                    norm_product = np.linalg.norm(x) * np.linalg.norm(y)
                    similarity = np.max(np.abs(cross_corr)) / norm_product if norm_product > 0 else np.nan
                    row_values.append(f'{similarity:.3f}')
                print('\t' + '\t'.join(row_values))
        print('  Pairwise Max Normalized Cross-Correlation of Mean-Centered Raw Histories:')
        compute_and_print_cross_corr_similarity(histories)
        print('  ... 1st Derivatives:')
        compute_and_print_cross_corr_similarity(first_derivatives)
        print('  ... 2nd Derivatives:')
        compute_and_print_cross_corr_similarity(second_derivatives)

        # compute PSD over high frequency 1st / 2nd derivatives to determine
        # "high" frequency cutoff threshold for each derivative order
        energy_cutoff_ratio = 0.95
        for derivative_type in ['first', 'second']:
            # compute aggregated cumulative energy distribution
            aggregated_thresholds = {}
            aggregated_power = None
            for model_type in MODEL_TYPES: # threshold is the same across model types
                fft_result = cache[f'fft_{derivative_type}'][model_type]
                power_spectrum = np.abs(fft_result) ** 2
                if aggregated_power is None:
                    aggregated_power = power_spectrum.copy()
                else:
                    aggregated_power += power_spectrum
            sorted_indices = np.argsort(np.abs(frequencies))
            sorted_freqs = np.abs(frequencies)[sorted_indices]
            sorted_power = aggregated_power[sorted_indices]
            cumulative_energy = np.cumsum(sorted_power)
            total_energy = cumulative_energy[-1]

            # find frequency such that energy_cutoff_ratio of energy is below it
            dynamic_threshold_index = np.searchsorted(cumulative_energy, energy_cutoff_ratio * total_energy)
            threshold = sorted_freqs[dynamic_threshold_index]
            print(f'  High Frequency {derivative_type.title()} Derivative Threshold (based on {int(100*energy_cutoff_ratio)}% energy cutoff ratio): {threshold:.4f}')

            # compute the high-frequency energy ratio
            hf_energy_uniform = {}
            for model_type in MODEL_TYPES:
                hf_energy_uniform[model_type] = {}
                fft_result = cache[f'fft_{derivative_type}'][model_type]
                power_spectrum = np.abs(fft_result) ** 2
                high_freq_mask = np.abs(frequencies) > threshold
                high_freq_energy = np.sum(power_spectrum[high_freq_mask])
                total_energy = np.sum(power_spectrum)
                hf_ratio = high_freq_energy / total_energy if total_energy != 0 else np.nan
                hf_energy_uniform[model_type] = hf_ratio
                print(f'    Model {model_type.upper()}: High Frequency Energy Ratio = {hf_ratio:.4f}')
    for cost_part_index in range(num_loss_types):
        label = f'{LOSS_TYPES[cost_part_index]} Loss Analysis'
        cache = {k: {m: costs_cache[k][m][cost_part_index] for m in MODEL_TYPES} for k in costs_cache.keys()}
        analyze_history(cache, label)
    analyze_history(gradient_norms_cache, 'Gradient Norm Analysis')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a QTE and QAE and generate correlation plots."
    )
    parser.add_argument("datasets_directory", type=str, nargs='?', default='generated_datasets', help="Path to the directory containing the generated datasets.")
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--prefix", type=str, default=None, help="Prefix to use when loading saved files")
    parser.add_argument("--overfit_threshold", type=float, default=.15, help="Detection threshold for overfit ratio (max % for increase in validation cost vs training cost)")
    parser.add_argument("--quantizer", type=str, default='bayesian_block', choices=['bayesian_block', 'hdbscan', 'equal_width'])
    parser.add_argument("--quantum_bottleneck_feature", type=str, default='z', choices=['z', 'marginal'])
    args = parser.parse_args()

    run_prefix = args.prefix if args.prefix else ''
    datasets = import_generated(args.datasets_directory)
    run_analysis(datasets, args.datasets_directory, args.overfit_threshold, args.quantizer, args.quantum_bottleneck_feature)
