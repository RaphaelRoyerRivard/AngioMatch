from os import walk
from os.path import exists
from shutil import copyfile


if __name__ == '__main__':
    base_destination_path = r'C:\Users\root\Data\Angiographie'
    new_data_path = r'C:\Users\root\Data\new_data'
    mitchel_path = r'C:\Users\root\Data\donnees_Mitchel_Benovoy'
    severine_path = r'C:\Users\root\Data\donnees_Severine_Habert'

    # NEW DATA
    for path, subfolders, files in walk(new_data_path):
        print(path)

        for file in files:
            if file.endswith(".dcm.ecg.r-peaks.txt"):
                patient = path.split("\\")[-1]
                angle = file.split(".")[0]
                destination_path = base_destination_path + "\\" + patient + "\\export\\" + angle
                if exists(destination_path):
                    copyfile(path + "\\" + file, destination_path + "\\r-peaks.txt")
                    print(file + " copied")

    # MITCHEL DATA
    for path, subfolders, files in walk(mitchel_path):
        print(path)

        for file in files:
            if file.endswith(".dcm.ecg.r-peaks.txt"):
                patient = path.split("\\")[-1]
                angle = str(int(file[6:8]))
                destination_path = base_destination_path + "\\" + patient + "\\export\\" + angle
                if exists(destination_path):
                    copyfile(path + "\\" + file, destination_path + "\\r-peaks.txt")
                    print(file + " copied")

    # SEVERINE DATA
    folder_pairs = {}
    for path, subfolders, files in walk(severine_path):
        print(path)

        for file in files:
            if file.endswith("-Frame1.jpg"):
                print(file)
                folder = path.split(severine_path)[1]
                filename = file.split("-")[0]
                folder_pairs[folder] = filename
                break

    for folder, file in folder_pairs.items():
        patient = folder.split("\\")[1]
        full_filename = severine_path + "\\" + patient + "\\anon\\" + file + ".dcm.ecg.r-peaks.txt"
        copyfile(full_filename, base_destination_path + folder + "\\r-peaks.txt")
        print(file + ".dcm.ecg.r-peaks.txt copied")
