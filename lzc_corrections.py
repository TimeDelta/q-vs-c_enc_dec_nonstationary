import numpy as np

from analysis import lempel_ziv_complexity_continuous, quantize_signal
from data_importers import import_generated
partitions = import_generated('generated_datasets')

def lempel_ziv_complexity_continuous2(data, num_symbols=30):
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

correct_values = []
differences = []
for _, (t, v) in partitions.items():
    for s in t:
        correct = lempel_ziv_complexity_continuous(s[1])
        differences.append(correct - lempel_ziv_complexity_continuous2(s[1]))
        correct_values.append(correct)
    for s in v:
        correct = lempel_ziv_complexity_continuous(s[1])
        differences.append(correct - lempel_ziv_complexity_continuous2(s[1]))
        correct_values.append(correct)
print('mean', np.mean(differences))
print('median', np.median(differences))
print('max', np.max(differences))

print('min correct value', np.min(correct_values))
print('percentage:', np.max(differences) / np.min(correct_values))
