from os import walk

base_path = r'C:\Users\root\Data\Angiographie'
for path, subfolders, files in walk(base_path):
    if "relevant_frames.txt" in files:
        rfile = open(path + "/relevant_frames.txt", "r")
        line = rfile.readline()
        rfile.close()
        split_path = path.split("\\")
        first = int(line.split(";")[0])
        last = int(line.split(";")[1])
        print(split_path[-3], f"{split_path[-1]}:", first + (last - first) / 2)
