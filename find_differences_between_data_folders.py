from os import walk
from shutil import rmtree
import numpy as np


if __name__ == '__main__':
    data_folder1 = r'C:\Users\root\Data\Angiographie'
    data_folder2 = r'C:\Users\root\Data\Angiographie_old'

    sequences1 = set()
    sequences2 = set()

    for i in range(2):
        data_folder = data_folder1 if i == 0 else data_folder2
        sequences = sequences1 if i == 0 else sequences2
        for path, subfolders, files in walk(data_folder):
            if "relevant_frames.txt" in files and "r-peaks.npy" in files:
                split_sequence = path.split(data_folder)[1][1:].split("\\export\\")
                sequences.add((split_sequence[0], split_sequence[1]))

    new_sequences = []
    for sequence in sequences1:
        if sequence not in sequences2:
            new_sequences.append(sequence)
    print("\nNew sequences:", new_sequences)

    missing_sequences = []
    for sequence in sequences2:
        if sequence not in sequences1:
            missing_sequences.append(sequence)
    print("\nMissing sequences:", missing_sequences)
