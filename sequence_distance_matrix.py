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

    frames = np.array(frames, dtype=np.float32)

    # print(frames.shape)
    # print(frames.swapaxes(1, 2).shape)

    # ones = np.ones((2, 2))
    # ones[0, 0] = 0
    # twos = np.ones((2, 2)) * 2
    # twos[0, 1] = 0
    # threes = np.ones((2, 2)) * 3
    # threes[1, 0] = 0
    # frames = np.array([ones, twos, threes])
    # frames = np.random.rand(228, 10, 10)

    start_time = time.time()
    distances = []
    manual_distances = []
    for i in range(len(frames)):
        print(i*100/len(frames), "%")
        # with broadcast
        # distance_matrix = np.abs(frames - frames[i])
        # distance_vector = np.sum(distance_matrix, axis=(1, 2))
        # distances.append(distance_vector)

        # without broadcast
        distances_i = []
        for j in range(len(frames)):
            if j < i:
                distances_i.append(manual_distances[j][i])
            elif j == i:
                distances_i.append(0)
            else:
                val = np.sum(np.abs(frames[i] - frames[j]))
                distances_i.append(val)
        manual_distances.append(distances_i)
    distances = np.array(distances)
    manual_distances = np.array(manual_distances)

    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    # plt.subplot(1, 3, 1)
    # plt.imshow(distances)
    # plt.colorbar()
    # plt.title("With broadcast")

    # plt.subplot(1, 3, 2)
    plt.imshow(manual_distances)
    plt.colorbar()
    plt.title("Without broadcast")
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.abs(manual_distances - distances))
    # plt.colorbar()
    # plt.title("Diff")

    plt.show()
