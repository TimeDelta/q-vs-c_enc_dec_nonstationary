import numpy as np

from analysis import lempel_ziv_complexity_continuous, quantize_signal
from data_importers import import_generated
partitions = import_generated('generated_datasets')

def lempel_ziv_complexity_continuous2(data, num_symbols=30):
    symbol_seq = quantize_signal(data, num_symbols)
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
    return complexity

correct_values = []
differences = []
for _, (t, v) in partitions.items():
    for s in t:
        correct = lempel_ziv_complexity_continuous2(s[1])
        differences.append(correct - lempel_ziv_complexity_continuous(s[1]))
        correct_values.append(correct)
    for s in v:
        correct = lempel_ziv_complexity_continuous2(s[1])
        differences.append(correct - lempel_ziv_complexity_continuous(s[1]))
        correct_values.append(correct)
print('mean', np.mean(differences))
print('median', np.median(differences))
print('max', np.max(differences))

print('min correct value', np.min(correct_values))
print('percentage:', np.max(differences) / np.min(correct_values))
