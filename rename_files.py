from os import walk, mkdir, rename
from shutil import copy2, copytree

if __name__ == '__main__':
    source_path = r'C:\Users\root\Data\donnees_Mitchel_Benovoy'
    destination_path = r'C:\Users\root\Data\Angiographie'
    for path, subfolders, files in walk(source_path):
        print(path)

        for file in files:
            if file.startswith("DIRW00") and not file.endswith(".dcm"):
                complete_filename = path + "\\" + file
                rename(complete_filename, complete_filename + ".dcm")

        # patient = ""
        # for subfolder in path.split('\\'):
        #     if len(subfolder) == 3 and subfolder[0] == 'P':
        #         patient = subfolder
        # folder_name = path.split('\\')[-1]
        # if folder_name[0:3] in ['RCA', 'LCA']:
        #     print(f"Copying {path} to {destination_path}\{patient}\export\{folder_name}")
        #     copytree(path, fr"{destination_path}\{patient}\export\{folder_name}")

        # if 'Series' in folder_name:
        #     folders = {}
        #     for file in files:
        #         name = file.split("Frame")[0][:-1]
        #         if name not in folders.keys():
        #             folders[name] = []
        #         folders[name].append(file)
        #     for folder, files in folders.items():
        #         mkdir(fr"{path}\{folder}")
        #         for file in files:
        #             copy2(fr"{path}\{file}", fr"{path}\{folder}\{file}")
