import numpy as np

def get_random_fourier_series(length):
    time_steps = np.linspace(0, 1, length, endpoint=False)
    num_terms = length // 2
    amplitudes = np.random.randn(num_terms)
    phases = np.random.uniform(0, 2*np.pi, num_terms)

    freqs = np.arange(1, num_terms + 1)[:, np.newaxis]
    return np.sum(amplitudes[:, None] * np.cos(2 * np.pi * freqs * time_steps + phases[:, None]), axis=0)
