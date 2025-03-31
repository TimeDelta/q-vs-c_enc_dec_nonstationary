import numpy as np
import torch
from qiskit.quantum_info import partial_trace, entropy
import antropy

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

def entanglement_entropy(state, subsystem=None):
    if subsystem:
        total_qubits = state.num_qubits
        trace_out = [i for i in range(total_qubits) if i not in subsystem]
        reduced_state = partial_trace(state, trace_out)
    else:
        reduced_state = state
    return entropy(reduced_state, base=2)

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
        # combine feeatures using mixed-radix encoding (treat each feature’s quantized value as a digit in a number with base equal to num_symbols)
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
