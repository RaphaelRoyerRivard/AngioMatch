from os import walk, rename
from os.path import exists
from shutil import copyfile


if __name__ == '__main__':
    base_destination_path = r'C:\Users\root\Data\Angiographie\sacre_coeur'
    input_folder = r'C:\Users\root\Data\sacre_coeur\export'

    for path, subfolders, files in walk(input_folder):
        print(path)

        # Rename folders to fit the ones created by the copy_r-peaks_files script
        # for subfolder in subfolders:
        #     if subfolder.startswith("Patient-"):
        #         rename(path + "\\" + subfolder, path + "\\" + "pt" + subfolder[-3:])

        for file in files:
            # C:\Users\root\Data\sacre_coeur\export\pt002\Study__XA_C.T.O.[20200527]\Series_001_Art. cor. gche 15 i_s éco\I1_Frame1.jpg
            if file.endswith(".jpg"):
                patient = path.split("\\")[-3]
                angle = str(int(path.split("\\")[-1].split("_")[1]))
                destination_path = base_destination_path + "\\" + patient + "\\export\\" + angle
                if exists(destination_path):
                    copyfile(path + "\\" + file, destination_path + "\\" + file)
                    print(file + " copied")
