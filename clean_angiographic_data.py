from os import walk
from shutil import rmtree


if __name__ == '__main__':
    input_folder = r'C:\Users\root\Data\Angiographie\sacre_coeur'

    for path, subfolders, files in walk(input_folder):
        print(path)

        if "r-peaks.txt" in files and "relevant_frames.txt" not in files:
            rmtree(path)
            print("folder deleted")
