import numpy as np
import torch
from qiskit.quantum_info import partial_trace, entropy
import antropy
from data_importers import import_generated

def differential_entropy(data):
    """
    data (np.ndarray): shape should be (num_features, sequence_length)
    Uses Freedman-Diaconis Rule to determine num_bins due to non-normal data
    """
    entropy_per_feature = []
    num_features = data.shape[0]
    for f in range(num_features):
        feature_data = data[f]
        q75, q25 = np.percentile(feature_data, [75 ,25])
        IQR = q75 - q25
        bin_width = 2 * IQR / np.cbrt(len(feature_data))
        num_bins = int(np.ceil((np.max(feature_data) - np.min(feature_data)) / bin_width))
        hist, edges = np.histogram(feature_data, bins=num_bins, density=True)
        dimensions = data.shape[1]

        widths_list = [np.diff(edge) for edge in edges]
        # create a meshgrid to combine bin widths across dimensions
        mesh = np.meshgrid(*widths_list, indexing='ij')
        bin_volumes = np.ones_like(mesh[0])
        for w in mesh:
            bin_volumes *= w

        bin_prob_mass = hist * bin_volumes

        nonzero = bin_prob_mass > 0
        de = -np.sum(bin_prob_mass[nonzero] * np.log(bin_prob_mass[nonzero] * bin_volumes[nonzero]))
        entropy_per_feature.append(de)
    return entropy_per_feature

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

class ModelStats:
    cost_history = None
    validation_costs = None
    bottleneck_entanglement_entropies = None
    bottleneck_full_vn_entropies = None

class SeriesStats:
    hurst_exponent = None
    lempel_ziv_complexity = None
    higuchi_fractal_dimension = None

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
    args = parser.parse_args()

    datasets = import_generated(args.datasets_directory)

    # load model statistics for each dataset and model type (qae and qte)
    stats_per_model = {}
    for d_i in datasets:
        for model_type in ['qae', 'qte']:
            print(f'Loading {model_type} model statistics for dataset {d_i}')
            stats = ModelStats()
            stats.cost_history = np.load(os.path.join(args.datasets_directory, f'dataset{d_i}_{model_type}_cost_history.npy'))
            stats.validation_costs = np.load(os.path.join(args.datasets_directory, f'dataset{d_i}_{model_type}_validation_costs.npy'))
            stats.entanglement_entropies = np.load(os.path.join(args.datasets_directory, f'dataset{d_i}_{model_type}_bottleneck_entanglement_entropies.npy'))
            stats.full_vn_entropies = np.load(os.path.join(args.datasets_directory, f'dataset{d_i}_{model_type}_bottleneck_full_vn_entropies.npy'))
            stats_per_model[(d_i, model_type)] = stats

    # compute complexity metrics for all validation series
    dataset_series_stats = {}
    for d_i, (training_series, validation_series) in datasets.items():
        for s_i, series in validation_series:
            num_features = len(series[0])
            print(f'Computing complexity metrics for dataset {d_i} series {s_i} ({num_features} features)')
            series_stats = SeriesStats()
            series_stats.lempel_ziv_complexity = lempel_ziv_complexity_continuous(series)
            series_stats.hurst_exponent = np.mean(hurst_exponent(series))
            series_stats.higuchi_fractal_dimension = np.mean(higuchi_fractal_dimension(series))
            # store as tuple (s_i, series_stats) for later annotation
            dataset_series_stats.setdefault(d_i, []).append((s_i, series_stats))

    # per series
    loss_vs_hurst    = {'qae': [], 'qte': []}
    loss_vs_lzc      = {'qae': [], 'qte': []}
    loss_vs_hfd      = {'qae': [], 'qte': []}
    entropy_vs_hurst = {'qae': [], 'qte': []}
    entropy_vs_lzc   = {'qae': [], 'qte': []}
    entropy_vs_hfd   = {'qae': [], 'qte': []}
    full_vn_vs_hurst = {'qae': [], 'qte': []}
    full_vn_vs_lzc   = {'qae': [], 'qte': []}
    full_vn_vs_hfd   = {'qae': [], 'qte': []}

    for (d_i, model_type), stats in stats_per_model.items():
        # first element of each numpy array is series index
        val_cost_dict = {row[0]: row[1] for row in stats.validation_costs}
        ent_dict = {row[0]: np.mean(row[1:]) for row in stats.entanglement_entropies}
        full_vn_dict = {row[0]: np.mean(row[1:]) for row in stats.full_vn_entropies}
        series_stats_list = dataset_series_stats[d_i]
        for s_i, s_stats in series_stats_list:
            cost = val_cost_dict.get(s_i, np.nan)
            ent = ent_dict.get(s_i, np.nan)
            full_vn = full_vn_dict.get(s_i, np.nan)

            loss_vs_hurst[model_type].append((s_stats.hurst_exponent, cost, d_i, s_i))
            loss_vs_lzc[model_type].append((s_stats.lempel_ziv_complexity, cost, d_i, s_i))
            loss_vs_hfd[model_type].append((s_stats.higuchi_fractal_dimension, cost, d_i, s_i))

            entropy_vs_hurst[model_type].append((s_stats.hurst_exponent, ent, d_i, s_i))
            entropy_vs_lzc[model_type].append((s_stats.lempel_ziv_complexity, ent, d_i, s_i))
            entropy_vs_hfd[model_type].append((s_stats.higuchi_fractal_dimension, ent, d_i, s_i))

            full_vn_vs_hurst[model_type].append((s_stats.hurst_exponent, full_vn, d_i, s_i))
            full_vn_vs_lzc[model_type].append((s_stats.lempel_ziv_complexity, full_vn, d_i, s_i))
            full_vn_vs_hfd[model_type].append((s_stats.higuchi_fractal_dimension, full_vn, d_i, s_i))


    # per dataset (mean complexity metrics over all validation series)
    aggregated_loss_vs_hurst    = {'qae': [], 'qte': []}
    aggregated_loss_vs_lzc      = {'qae': [], 'qte': []}
    aggregated_loss_vs_hfd      = {'qae': [], 'qte': []}
    aggregated_entropy_vs_hurst = {'qae': [], 'qte': []}
    aggregated_entropy_vs_lzc   = {'qae': [], 'qte': []}
    aggregated_entropy_vs_hfd   = {'qae': [], 'qte': []}
    aggregated_full_vn_vs_hurst = {'qae': [], 'qte': []}
    aggregated_full_vn_vs_lzc   = {'qae': [], 'qte': []}
    aggregated_full_vn_vs_hfd   = {'qae': [], 'qte': []}

    for (d_i, model_type), stats in stats_per_model.items():
        # first element of each numpy array is series index
        final_loss = np.mean([row[1] for row in stats.validation_costs])
        avg_entropy = np.mean([row[1:] for row in stats.entanglement_entropies])
        avg_full_vn = np.mean([row[1:] for row in stats.full_vn_entropies])

        series_stats_list = dataset_series_stats[d_i]
        agg_hurst = np.mean([s.hurst_exponent for (_, s) in series_stats_list])
        agg_lzc   = np.mean([s.lempel_ziv_complexity for (_, s) in series_stats_list])
        agg_hfd   = np.mean([s.higuchi_fractal_dimension for (_, s) in series_stats_list])

        aggregated_loss_vs_hurst[model_type].append((agg_hurst, final_loss, d_i))
        aggregated_loss_vs_lzc[model_type].append((agg_lzc, final_loss, d_i))
        aggregated_loss_vs_hfd[model_type].append((agg_hfd, final_loss, d_i))

        aggregated_entropy_vs_hurst[model_type].append((agg_hurst, avg_entropy, d_i))
        aggregated_entropy_vs_lzc[model_type].append((agg_lzc, avg_entropy, d_i))
        aggregated_entropy_vs_hfd[model_type].append((agg_hfd, avg_entropy, d_i))

        aggregated_full_vn_vs_hurst[model_type].append((agg_hurst, avg_full_vn, d_i))
        aggregated_full_vn_vs_lzc[model_type].append((agg_lzc, avg_full_vn, d_i))
        aggregated_full_vn_vs_hfd[model_type].append((agg_hfd, avg_full_vn, d_i))

    def plot_scatter_individual(data_dict, x_label, y_label, title, filename):
        plt.figure()
        colors = {"qae": "blue", "qte": "orange"}
        for model_type, data in data_dict.items():
            if not data:
                continue
            x_vals = [d[0] for d in data]
            y_vals = [d[1] for d in data]
            plt.scatter(x_vals, y_vals, color=colors[model_type])
            coeffs = np.polyfit(x_vals, y_vals, 1)
            slope = coeffs[0]
            poly_eqn = np.poly1d(coeffs)
            x_fit = np.linspace(min(x_vals), max(x_vals), 100)
            plt.plot(x_fit, poly_eqn(x_fit), color=colors[model_type], linestyle='--', label=f"{model_type.upper()} (slope={slope:.3f})")
            for xi, yi, d in zip(x_vals, y_vals, data):
                plt.annotate('', (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=8)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        save_path = os.path.join(args.datasets_directory, filename)
        plt.savefig(save_path)
        print(f"Saved individual series plot to {save_path}")

    def plot_scatter_aggregated(data_dict, x_label, y_label, title, filename):
        plt.figure()
        colors = {"qae": "blue", "qte": "orange"}
        for model_type, data in data_dict.items():
            if not data:
                continue
            x_vals = [d[0] for d in data]
            y_vals = [d[1] for d in data]
            plt.scatter(x_vals, y_vals, marker='s', color=colors[model_type])
            coeffs = np.polyfit(x_vals, y_vals, 1)
            slope = coeffs[0]
            poly_eqn = np.poly1d(coeffs)
            x_fit = np.linspace(min(x_vals), max(x_vals), 100)
            plt.plot(x_fit, poly_eqn(x_fit), color=colors[model_type], linestyle='--', label=f"{model_type.upper()} (slope={slope:.3f})")
            for xi, yi, d in zip(x_vals, y_vals, data):
                plt.annotate(f"{d[2]}", (xi, yi), textcoords="offset points", xytext=(5,5), fontsize=10, color='red')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        save_path = os.path.join(args.datasets_directory, filename)
        plt.savefig(save_path)
        print(f"Saved aggregated plot to {save_path}")


    # ===== per series =====
    # Loss vs. Complexity
    plot_scatter_individual(loss_vs_hurst,
                            "Hurst Exponent",
                            "Mean Validation Loss",
                            "Loss vs. Hurst Exponent (Individual)",
                            "loss_vs_hurst_individual.png")
    plot_scatter_individual(loss_vs_lzc,
                            "Lempel-Ziv Complexity",
                            "Mean Validation Loss",
                            "Loss vs. LZC (Individual)",
                            "loss_vs_lzc_individual.png")
    plot_scatter_individual(loss_vs_hfd,
                            "Higuchi Fractal Dimension",
                            "Mean Validation Loss",
                            "Loss vs. HFD (Individual)",
                            "loss_vs_hfd_individual.png")

    # Entropy vs. Complexity
    plot_scatter_individual(entropy_vs_hurst,
                            "Hurst Exponent",
                            "Entanglement Entropy",
                            "Entropy vs. Hurst Exponent (Individual)",
                            "entropy_vs_hurst_individual.png")
    plot_scatter_individual(entropy_vs_lzc,
                            "Lempel-Ziv Complexity",
                            "Entanglement Entropy",
                            "Entropy vs. LZC (Individual)",
                            "entropy_vs_lzc_individual.png")
    plot_scatter_individual(entropy_vs_hfd,
                            "Higuchi Fractal Dimension",
                            "Entanglement Entropy",
                            "Entropy vs. HFD (Individual)",
                            "entropy_vs_hfd_individual.png")

    # Full VN Entropy vs. Complexity (Individual)
    plot_scatter_individual(full_vn_vs_hurst,
                            "Hurst Exponent",
                            "Full VN Entropy",
                            "Full VN Entropy vs. Hurst (Individual)",
                            "full_vn_vs_hurst_individual.png")
    plot_scatter_individual(full_vn_vs_lzc,
                            "Lempel-Ziv Complexity",
                            "Full VN Entropy",
                            "Full VN Entropy vs. LZC (Individual)",
                            "full_vn_vs_lzc_individual.png")
    plot_scatter_individual(full_vn_vs_hfd,
                            "Higuchi Fractal Dimension",
                            "Full VN Entropy",
                            "Full VN Entropy vs. HFD (Individual)",
                            "full_vn_vs_hfd_individual.png")

    # ===== per dataset =====
    # Loss vs. Complexity
    plot_scatter_aggregated(aggregated_loss_vs_hurst,
                            "Mean Hurst Exponent",
                            "Mean Validation Loss",
                            "Loss vs. Hurst Exponent (Aggregated)",
                            "loss_vs_hurst_aggregated.png")
    plot_scatter_aggregated(aggregated_loss_vs_lzc,
                            "Mean Lempel-Ziv Complexity",
                            "Mean Validation Loss",
                            "Loss vs. LZC (Aggregated)",
                            "loss_vs_lzc_aggregated.png")
    plot_scatter_aggregated(aggregated_loss_vs_hfd,
                            "Mean Higuchi Fractal Dimension",
                            "Mean Validation Loss",
                            "Loss vs. HFD (Aggregated)",
                            "loss_vs_hfd_aggregated.png")

    # Entropy vs. Complexity
    plot_scatter_aggregated(aggregated_entropy_vs_hurst,
                            "Mean Hurst Exponent",
                            "Mean Entanglement Entropy",
                            "Entropy vs. Hurst Exponent (Aggregated)",
                            "entropy_vs_hurst_aggregated.png")
    plot_scatter_aggregated(aggregated_entropy_vs_lzc,
                            "Mean Lempel-Ziv Complexity",
                            "Mean Entanglement Entropy",
                            "Entropy vs. LZC (Aggregated)",
                            "entropy_vs_lzc_aggregated.png")
    plot_scatter_aggregated(aggregated_entropy_vs_hfd,
                            "Mean Higuchi Fractal Dimension",
                            "Mean Entanglement Entropy",
                            "Entropy vs. HFD (Aggregated)",
                            "entropy_vs_hfd_aggregated.png")

    # Full VN Entropy vs. Complexity
    plot_scatter_aggregated(aggregated_full_vn_vs_hurst,
                            "Mean Hurst Exponent",
                            "Mean Full VN Entropy",
                            "Full VN Entropy vs. Hurst (Aggregated)",
                            "full_vn_vs_hurst_aggregated.png")
    plot_scatter_aggregated(aggregated_full_vn_vs_lzc,
                            "Mean Lempel-Ziv Complexity",
                            "Mean Full VN Entropy",
                            "Full VN Entropy vs. LZC (Aggregated)",
                            "full_vn_vs_lzc_aggregated.png")
    plot_scatter_aggregated(aggregated_full_vn_vs_hfd,
                            "Mean Higuchi Fractal Dimension",
                            "Mean Full VN Entropy",
                            "Full VN Entropy vs. HFD (Aggregated)",
                            "full_vn_vs_hfd_aggregated.png")

    plt.show()
