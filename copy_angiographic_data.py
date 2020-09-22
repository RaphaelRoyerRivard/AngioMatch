from os import walk, rename
from os.path import exists
from shutil import copyfile


if __name__ == '__main__':
    base_destination_path = r'C:\Users\root\Data\Angiographie_new'
    new_data_path = r'C:\Users\root\Data\new_data'
    mitchel_path = r'C:\Users\root\Data\donnees_Mitchel_Benovoy'
    severine_path = r'C:\Users\root\Data\donnees_Severine_Habert'
    sacre_coeur_path = r'C:\Users\root\Data\sacre_coeur'

    # NEW DATA
    for path, subfolders, files in walk(new_data_path):
        print(path)

        for file in files:
            if file.endswith(".jpg"):
                patient = path.split("\\")[-5]
                angle = path.split("\\")[-1]
                destination_path = base_destination_path + "\\" + patient + "\\export\\" + angle
                if exists(destination_path):
                    copyfile(path + "\\" + file, destination_path + "\\" + file)
                    print(file + " copied")

    # MITCHEL DATA
    for path, subfolders, files in walk(mitchel_path):
        print(path)

        for file in files:
            if file.endswith(".jpg") and "_Frame" in file:
                patient = path.split("\\")[-3].split(", ")[-1]
                angle = str(int(file.split("_")[0][4:]))
                destination_path = base_destination_path + "\\" + patient + "\\export\\" + angle
                if exists(destination_path):
                    copyfile(path + "\\" + file, destination_path + "\\" + file)
                    print(file + " copied")

    # SEVERINE DATA
    for path, subfolders, files in walk(severine_path):
        print(path)

        for file in files:
            if file.endswith(".jpg") and "-Frame" in file:
                patient = path.split("\\")[-3]
                angle = path.split("\\")[-1]
                destination_path = base_destination_path + "\\" + patient + "\\export\\" + angle
                if exists(destination_path):
                    copyfile(path + "\\" + file, destination_path + "\\" + file)
                    print(file + " copied")

    # SACRE-COEUR DATA
    for path, subfolders, files in walk(sacre_coeur_path):
        print(path)

        # Rename folders to fit the ones created by the copy_r-peaks_files script
        # for subfolder in subfolders:
        #     if subfolder.startswith("Patient-"):
        #         rename(path + "\\" + subfolder, path + "\\" + "pt" + subfolder[-3:])

        for file in files:
            if file.endswith(".jpg"):
                patient = path.split("\\")[-3]
                angle = str(int(path.split("\\")[-1].split("_")[1]))
                destination_path = base_destination_path + "\\" + patient + "\\export\\" + angle
                if exists(destination_path):
                    copyfile(path + "\\" + file, destination_path + "\\" + file)
                    print(file + " copied")
