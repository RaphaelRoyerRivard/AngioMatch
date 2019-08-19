import cv2
import numpy as np
from os import walk
import re
from matplotlib import pyplot as plt
from sympy import mobius


def get_video_statistics(files):
    video_statistics = []
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

        video_statistics.append((id, np.average(gradient), np.std(img)))

    return video_statistics


def get_frames_with_good_gradient(gradient_avg):
    first = last = -1
    for i, value in enumerate(gradient_avg):
        # Several sequences start with a few frames that are more "grainy" and thus have more gradient even if there is no iodine solution in the vessels yet. We want to skip those frames.
        if value >= 0:
            if first < 0:
                first = i
            elif i == first + 10:  # We want to keep the "first" when there has been at least 10 successive frames with a gradient over the average
                break
        elif first >= 0:
            first = -1
    for i, value in reversed(list(enumerate(gradient_avg))):
        if value >= 0:
            if last < 0:
                last = i
            elif i == last - 10:
                break
        elif last >= 0:
            last = -1

    return first, last


def remove_outliers(values):
    neighbor_size = 4  # Needs to be even
    for i in range(len(values)):
        neighbor = []
        # Find offset for values at the extremities
        offset = 0
        if i < neighbor_size // 2:
            offset = neighbor_size // 2 - i
        elif i > len(values) - 1 - (neighbor_size // 2):
            offset = len(values) - 1 - i - neighbor_size // 2
        # Find the right neighbors
        for j in range(neighbor_size + 1):
            neighbor_offset = j - neighbor_size // 2 + offset
            if neighbor_offset == 0:
                continue  # current i
            neighbor.append(values[i + neighbor_offset])
        # Get the average of the neighbors
        avg = np.array(neighbor).mean()
        # if abs(values[i] - values[compare_1]) >= 1 and abs(values[i] - values[compare_2]) >= 1:
        if abs(values[i] - avg) >= 1:
            print(f"Frame {i+1} is an outlier! avg is {avg} while frame is {values[i]}")
            values[i] = avg


def gaussian_kernel_1d(n, sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    return [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]


def get_moving_average(values, N):
    gaussian_kernel = np.array(gaussian_kernel_1d(N, sigma=4))
    moving_average = np.convolve(values, gaussian_kernel/gaussian_kernel.sum(), mode='same')
    return moving_average


def get_heartbeat_frequency(title, intensity_std, good_gradient):
    extrema_minimum_frame_distance = 10
    moving_average_N = 19
    averaged_std_moving_average = get_moving_average(intensity_std, moving_average_N)
    smoothed_std = intensity_std - averaged_std_moving_average

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
        elif i == current_local_minima_index + extrema_minimum_frame_distance:
            local_minimas.append(current_local_minima_index)
            current_local_minima_index = i
            current_local_minima = value

        # maxima
        if value > current_local_maxima:
            current_local_maxima_index = i
            current_local_maxima = value
        elif i == current_local_maxima_index + extrema_minimum_frame_distance:
            local_maximas.append(current_local_maxima_index)
            current_local_maxima_index = i
            current_local_maxima = value

    print('good gradient', good_gradient)
    print('min', local_minimas)
    print('max', local_maximas)
    filtered_local_minimas = [x for x in local_minimas if good_gradient[0] <= x <= good_gradient[1]]
    filtered_local_maximas = [x for x in local_maximas if good_gradient[0] <= x <= good_gradient[1]]
    print('cleaned min', filtered_local_minimas)
    print('cleaned max', filtered_local_maximas)
    diff_min = [local_minimas[i] - local_minimas[i-1] for i in range(1, len(local_minimas))]
    diff_max = [local_maximas[i] - local_maximas[i-1] for i in range(1, len(local_maximas))]
    filtered_diff_min = [filtered_local_minimas[i] - filtered_local_minimas[i-1] for i in range(1, len(filtered_local_minimas))]
    filtered_diff_max = [filtered_local_maximas[i] - filtered_local_maximas[i-1] for i in range(1, len(filtered_local_maximas))]
    print('diff min', filtered_diff_min)
    print('diff max', filtered_diff_max)
    print('mean diff min', np.array(filtered_diff_min).mean())
    print('mean diff max', np.array(filtered_diff_max).mean())
    merged_diff = np.array(diff_min + diff_max)
    merged_filtered_diff = np.array(filtered_diff_min + filtered_diff_max)
    std = round(merged_diff.std() * 100) / 100
    filtered_std = round(merged_filtered_diff.std() * 100) / 100
    heartbeat_frequency = round(merged_diff.mean())
    filtered_heartbeat_frequency = round(merged_filtered_diff.mean())
    print('heartbeat_frequency', filtered_heartbeat_frequency)

    # plt.clf()
    # plt.axvspan(good_gradient[0], good_gradient[1], color='green')
    # for local_minima in local_minimas:
    #     plt.axvline(local_minima, color='pink')
    # for local_maxima in local_maximas:
    #     plt.axvline(local_maxima, color='cyan')
    # plt.axhline(0, color='r')
    # plt.plot(intensity_std, color='black', label='Pixel intensity std')
    # plt.plot(averaged_std_moving_average, color='b', label=f'Moving average (N={moving_average_N})')
    # plt.plot(smoothed_std, color='orange', label='Std with subtracted moving average')
    # plt.title(f"Heartbeat detection ({title})\nvalues: {len(merged_diff)}, std: {std}, mean: {heartbeat_frequency} vs values: {len(merged_filtered_diff)}, std: {filtered_std}, mean: {filtered_heartbeat_frequency}")
    # plt.xlabel("frame # in sequence")
    # plt.ylabel("pixel intensity std (avg std subtracted)")
    # plt.legend()
    # plt.savefig(f"./figures/{title.replace(',', '')}.png")
    # plt.show()

    return filtered_heartbeat_frequency


if __name__ == '__main__':
    i = 0
    hb_ground_truth = [21, 22, 22, 20, 21, 21, 20,  # AA-4
                       18, 18, 17, 17, 18, 18,  # ABL-5
                       12, 11, 11, 11, 12, 12,  # AC-1
                       10, 10, 10, 10, 9,  # ALR-2
                       8, 8, 9, 8, 8, 9, 9, 9, 9,  # G1
                       11, 11,  # G10
                       10, 9,  # G12
                       12, 13, 13, 12, 12, 13, 13, 13,  # G13
                       12, 11, 14, 17,  # G14
                       8, 8, 7, 8, 7, 8,  # G15
                       10, 11, 10, 10, 10, 10, 10, 10, 10, 10,  # G16
                       9, 9, 8, 8, 8, 8, 8, 8, 8,  # G17
                       17, 17, 17, 17, 17, 17, 18, 18, 17,  # G18
                       9, 9, 9, 9, 9,  # G2
                       8, 8, 8, 8, 8, 8, 8,  # G3
                       9, 9, 9, 9, 9, 9, 9, 8,  # G5
                       8, 8, 8, 8, 8, 8,  # G6
                       16, 16, 17, 17, 15, 16, 16,  # G8
                       11, 11, 11, 11, 10, 11,  # G9
                       11, 11, 11, 11, 11, 11,  # JEL-10
                       14, 14, 14, 14, 15, 15,  # KC-3
                       9, 9, 9, 9, 9, 9, 9,  # KR-11
                       11, 11, 11, 12,  # MAL-8
                       16, 13, 14, 15, 15,  # MB-12
                       14, 14,  # MJY-9
                       10, 10]  # SB-6
    base_path = r'C:\Users\root\Data\Angiographie\G17\export\6'
    for path, subfolders, files in walk(base_path):

        video_statistics = get_video_statistics(files)

        if len(video_statistics) == 0:
            continue

        print(path)

        split_path = path.split("\\export\\")
        angle = split_path[1]
        patient = split_path[0].split("\\Angiographie\\")[1]

        print(patient, angle)

        video_statistics.sort()

        gradient_avg = np.array([x[1] for x in video_statistics])
        avg_avg = np.mean(gradient_avg)
        gradient_avg = gradient_avg - avg_avg
        remove_outliers(gradient_avg)
        gradient_moving_average = get_moving_average(gradient_avg, 19)

        first, last = get_frames_with_good_gradient(gradient_moving_average)

        intensity_std = np.array([x[2] for x in video_statistics])
        std_avg = np.average(intensity_std)
        intensity_std = intensity_std - std_avg

        remove_outliers(intensity_std)
        frequency = get_heartbeat_frequency(f"{patient}, {angle}", intensity_std, (first, last))

        plt.plot(gradient_avg, color='black')
        plt.plot(gradient_moving_average, color='gray')
        plt.axvspan(first, last, color='g')
        plt.axhline(0, color='r')
        plt.title("avg gradient intensity")
        plt.xlabel("frame # in the sequence")
        plt.ylabel("gradient intensity")
        plt.show()

        # rfile = open(path + "/temp/seg/relevant_frames.txt", "r")
        # line = rfile.readline()
        # rfile.close()
        # # print(len(gradient_avg), "frames [", first, ",", last, "] vs", line)
        wfile = open(path + "/relevant_frames.txt", "w+")
        # wfile.write(str(first) + ";" + str(last) + ";" + str(frequency))
        # wfile.write(str(first) + ";" + str(last) + ";" + str(hb_ground_truth[i]))
        wfile.close()
        i += 1
        print(len(video_statistics), f"frames [{first}, {last}] @{frequency}")
