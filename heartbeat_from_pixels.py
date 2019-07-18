import cv2
import numpy as np
from matplotlib import pyplot as plt

avg = []
std = []
for i in range(1, 229):
    img_path = rf"\\primnis.gi.polymtl.ca\dfs\cheriet\Images\Cardiologie\Angiographie\AA-4\export\LCA_30LAO25CAU\DIRW0019-Frame{i}.jpg"
    # img_path = rf"\\primnis.gi.polymtl.ca\dfs\cheriet\Images\Cardiologie\Angiographie\AC-1\export\LCA_LAT\DIRW0013-Frame{i}.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    avg.append(np.average(img))
    std.append(np.std(img))

    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()

    # kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    # kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    # edges_x = cv2.filter2D(img, cv2.CV_8U, kernelx)
    # edges_y = cv2.filter2D(img, cv2.CV_8U, kernely)
    # gradient = edges_x + edges_y
    # cv2.imshow('Gradients', gradient)
    # cv2.waitKey(0)

    # avg.append(np.average(gradient))
    # std.append(np.std(gradient))

avg = np.array(avg)
std = np.array(std)

avg_avg = np.average(avg)
avg = avg - avg_avg
avg_std = np.average(std)
std = std - avg_std

N = 20
averaged_std_moving_average = np.convolve(std, np.ones((N,))/N, mode='same')
smoothed_std = std - averaged_std_moving_average


local_minimas = []
local_maximas = []
current_local_minima = np.inf
current_local_maxima = -np.inf
current_local_minima_index = current_local_maxima_index = -1
for i, value in enumerate(smoothed_std):
    # minima
    if value < current_local_minima:
        current_local_minima_index = i
        current_local_minima = value
    elif i == current_local_minima_index + 10:
        local_minimas.append(current_local_minima_index)
        current_local_minima_index = i
        current_local_minima = value

    # maxima
    if value > current_local_maxima:
        current_local_maxima_index = i
        current_local_maxima = value
    elif i == current_local_maxima_index + 10:
        local_maximas.append(current_local_maxima_index)
        current_local_maxima_index = i
        current_local_maxima = value


# plt.subplot(1, 2, 1)
# plt.axvspan(20, 60, color='y')
# plt.axvspan(60, 124, color='g')
# plt.axvspan(124, 170, color='y')
# plt.plot(avg, color='black')
# plt.title("avg")
# plt.ylabel("pixel intensity")
#
# plt.subplot(1, 2, 2)
# plt.axvspan(20, 60, color='y')
# plt.axvspan(60, 124, color='g')
# plt.axvspan(124, 170, color='y')
for local_minima in local_minimas:
    plt.axvline(local_minima, color='pink')
for local_maxima in local_maximas:
    plt.axvline(local_maxima, color='cyan')
plt.axhline(0, color='r')
plt.plot(std, color='black', label='Pixel intensity std')
plt.plot(averaged_std_moving_average, color='b', label=f'Moving average (N={N})')
plt.plot(smoothed_std, color='orange', label='Std with subtracted moving average')
plt.title("Global pixel intensity std")
plt.xlabel("frame # in sequence")
plt.ylabel("pixel intensity std (avg std subtracted)")
plt.legend()


plt.show()
