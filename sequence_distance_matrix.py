import cv2
import numpy as np
from os import walk
import re
from matplotlib import pyplot as plt
import time


base_path = r'\\primnis.gi.polymtl.ca\dfs\cheriet\Images\Cardiologie\Angiographie'
for path, subfolders, files in walk(base_path):
    frames = []
    gradients = []
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

        frames.append(img)
        gradients.append(gradient)

    if len(frames) == 0:
        continue

    print(len(frames), "frames in", path)

    frames = np.array(frames)

    # print(frames.shape)
    # print(frames.swapaxes(1, 2).shape)

    start_time = time.time()
    distances = []
    for i in range(len(frames)):
        print(i)
        distances.append(frames - frames[i])
        # distances_i = []
        # for j in range(len(frames)):
        #     if j < i:
        #         distances_i.append(distances[j][i])
        #     elif j == i:
        #         distances_i.append(0)
        #     else:
        #         val = np.sum(np.abs(frames[i] - frames[j]))
        #         distances_i.append(val)
        # distances.append(distances_i)
    distances = np.array(distances)

    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    plt.imshow(distances)
    plt.colorbar()
    plt.show()
