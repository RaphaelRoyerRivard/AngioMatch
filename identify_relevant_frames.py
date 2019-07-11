import cv2
import numpy as np
from os import walk
import re
from matplotlib import pyplot as plt

base_path = r'\\primnis.gi.polymtl.ca\dfs\cheriet\Images\Cardiologie\Angiographie'
for path, subfolders, files in walk(base_path):
    avg = []
    for filename in files:
        if not bool(re.search('.*Frame[0-9]+.jpg', filename)):
            continue
        id = int(filename.split("Frame")[1].split(".")[0])
        img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)

        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        edges_x = cv2.filter2D(img, cv2.CV_8U, kernelx)
        edges_y = cv2.filter2D(img, cv2.CV_8U, kernely)
        gradient = edges_x + edges_y
        avg.append((id, np.average(gradient)))
    if len(avg) == 0:
        continue
    print(path)
    avg.sort()
    avg = np.array([x[1] for x in avg])
    avg_avg = np.average(avg)
    avg = avg - avg_avg

    first = last = -1
    for i, value in enumerate(avg):
        # Several sequences start with a few frames that are more "grainy" and thus have more gradient even if there is no iodine solution in the vessels yet. We want to skip those frames.
        if value >= 0:
            if first < 0:
                first = i
            elif i == first + 10:  # We want to keep the "first" when there has been at least 10 successive frames with a gradient over the average
                break
        elif first >= 0:
            first = -1
    for i, value in reversed(list(enumerate(avg))):
        if value >= 0:
            if last < 0:
                last = i
            elif i == last - 10:
                break
        elif last >= 0:
            last = -1

    # plt.plot(avg, color='black')
    # plt.axvspan(first, last, color='g')
    # plt.axhline(0, color='r')
    # plt.title("avg gradient intensity")
    # plt.xlabel("frame # in the sequence")
    # plt.ylabel("gradient intensity")
    # plt.show()

    rfile = open(path + "/temp/seg/relevant_frames.txt", "r")
    line = rfile.readline()
    rfile.close()
    print(len(avg), "frames [", first, ",", last, "] vs", line)
    # print(len(avg), "frames [", first, ",", last, "]")
    wfile = open(path + "/relevant_frames.txt", "w+")
    wfile.write(str(first) + ";" + str(last))
    wfile.close()
