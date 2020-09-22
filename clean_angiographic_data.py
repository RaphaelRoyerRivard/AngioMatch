from os import walk
from shutil import rmtree
import numpy as np


if __name__ == '__main__':
    input_folder = r'C:\Users\root\Data\Angiographie_new'

    for path, subfolders, files in walk(input_folder):
        print(path)

        if "r-peaks.npy" in files:
            delete_folder = False
            if "relevant_frames.txt" not in files:
                delete_folder = True
            else:
                r_peaks = np.load(path + "\\r-peaks.npy")
                if len(r_peaks) < 3:
                    delete_folder = True
            if delete_folder:
                rmtree(path)
                print("folder deleted")
