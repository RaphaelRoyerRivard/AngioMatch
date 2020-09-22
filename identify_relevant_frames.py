import cv2
import numpy as np
from os import walk
import re
# from matplotlib import pyplot as plt
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

        video_statistics.append((id, np.average(gradient), np.std(img), img))

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
    if len(values <= neighbor_size):
        return
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
        if abs(values[i] - avg) >= 1:
            print(f"Frame {i+1} is an outlier! avg is {avg} while frame is {values[i]}")
            values[i] = avg


def gaussian_kernel_1d(n, sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    return [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]


def get_moving_average(values, N, use_gaussian_kernel=True, mode='same', pad_values=False):
    if use_gaussian_kernel:
        kernel = np.array(gaussian_kernel_1d(N, sigma=4))
    else:
        kernel = np.ones(N)
    if mode == 'same':
        # We manually add padding and use valid mode, otherwise numpy will use 0 padding
        values = np.append(np.ones(int(N/2)) * values[0], values)
        values = np.append(values, np.ones(int(N/2)) * values[-1])
        mode = 'valid'
    moving_average = np.convolve(values, kernel/kernel.sum(), mode=mode)
    if pad_values and values.shape[0] > moving_average.shape[0]:
        if (values.shape[0] - moving_average.shape[0]) % 2 > 0:
            raise Exception(f"Size difference between values ({values.shape[0]}) and moving average ({moving_average.shape[0]}) is odd when it should be even.")
        half_size_diff = int((values.shape[0] - moving_average.shape[0]) / 2)
        moving_average = np.append(np.ones(half_size_diff) * moving_average[0], moving_average)
        moving_average = np.append(moving_average, np.ones(half_size_diff) * moving_average[-1])
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
    # i = 0
    # hb_ground_truth = [
    #     22, 22.5, 22, 20, 20, 22, 23,  # AA-4
    #     18.5, 18, 17, 17.5, 18.5, 18.5,  # ABL-5
    #     11.5, 11.5, 11, 11, 12, 12,  # AC-1
    #     10.5, 9.5, 10, 10, 9.5,  # ALR-2
    #     8, 8, 9, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,  # G1
    #     11, 11,  # G10
    #     9.5, 9.5,  # G12
    #     11.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12, 12.5,  # G13
    #     16.5, 11.5, 11.5, 14.5,  # G14
    #     7.5, 7.5, 7, 7.5, 7.5, 7,  # G15
    #     9, 10, 10, 10, 10, 10.5, 10.4, 10, 10, 10,  # G16
    #     8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5,  # G17
    #     17, 16.5, 17, 16.5, 17.5, 17.5, 18, 17.5, 17,  # G18
    #     9, 9, 9, 9, 9,  # G2
    #     8, 8, 8, 8, 8, 8, 8,  # G3
    #     9, 9, 9, 8.5, 8.5, 8.5, 8.5, 8.5,  # G5
    #     8, 8, 8, 8, 8, 8, 8,  # G6
    #     15.5, 16, 16, 16, 16, 16.5, 16.5,  # G8
    #     11, 11, 11, 11, 10, 11,  # G9
    #     11, 10.5, 11, 11, 11, 11,  # JEL-10
    #     14, 14.5, 14, 14, 15, 15,  # KC-3
    #     9, 9, 9, 9.5, 9.5, 9, 9,  # KR-11
    #     11, 11, 11.5, 11.5,  # MAL-8
    #     15.5, 14, 14, 15.5, 15.5,  # MB-12
    #     14.5, 14.5,  # MJY-9
    #     18, 17.5, 18, 18, 17, 18.5, 16, 19, 16.5,  # P20
    #     10, 10, 9.5, 9.5, 9.5, 9.25, 10.5, 11.5, 10.5, 11.5,  # P21
    #     16, 17, 15, 18, 15.5, 17, 16, 20, 15, 18, 13.5, 14, 18, 13.5, 14, 18,  # P22
    #     14, 14, 15.5, 14.5, 15, 16, 15.5, 14,  # P23
    #     8.5, 8, 8.5, 8.5, 8.5, 9, 9,  # P24
    #     20, 20, 19, 19, 21, 18, 19, 17, 17.5, 19.5, 17,  # P25
    #     9, 9, 9, 9, 9, 9, 9, 8.5, 9, 9,  # P26
    #     16, 15, 15, 15, 15, 14.5, 14.5,  # P27
    #     9.5, 5.5, 5.5, 9.5, 9, 11.5, 12.5, 11.5, 9, 10, 12, 11, 11,  # P28
    #     10.5, 10.5, 10.5, 10, 11, 11,  # P29
    #     9.5, 10, 10, 9.5, 9.5, 9.5, 10.5, 10,  # P30
    #     12, 12, 11.5, 11.5, 11.5, 11.5,  # P31
    #     11, 11.5, 11.5, 11, 13.5, 12, 10.5, 10.5,  # P32
    #     10, 10  # SB-6
    # ]
    # contracted_ground_truth = [
    #     75, 110, 90, 90, 89, 103, 82,  # AA-4
    #     75, 80, 80, 80, 69, 69,  # ABL-5
    #     40, 55, 52, 62, 64, 52,  # AC-1
    #     38, 47, 47, 37, 55,  # ALR-2
    #     47, 46.5, 43, 34.5, 45, 47, 44, 44, 37,  # G1
    #     41, 30,  # G10
    #     41, 40,  # G12
    #     36, 37, 37, 42, 53, 40, 14, 43,  # G13 (10, 11, 12, 13, 6, 7, 8, 9)
    #     49, 55, 43, 56,  # G14
    #     46, 61, 34, 34, 30, 34.5,  # G15
    #     30, 42, 43, 37, 30, 30, 33.5, 21, 81, 90,  # G16 (10, 11, 12, 13, 4, 5, 6, 7, 8, 9)
    #     47.5, 38.5, 31, 39, 35, 28.5, 19, 68, 72.5,  # G17
    #     82.5, 91, 78.5, 75.5, 66, 65, 72, 88, 100,  # G18
    #     50, 58.5, 51, 51, 61.5,  # G2
    #     36, 28, 35, 35, 34, 33.5, 38,  # G3 (3, 4, 5, 6, 7, 8, 9)
    #     34, 25, 35, 44.5, 44, 37.5, 49, 56,  # G5
    #     36, 29, 35, 35, 26, 33, 38,  # G6 (3, 4, 5, 6, 7, 8, 9)
    #     80, 75.5, 65, 79.5, 79, 66, 82,  # G8
    #     56.5, 46, 34.5, 49, 50, 47.5,  # G9
    #     60, 50.5, 52, 51, 60, 60,  # JEL-10
    #     89, 109.5, 82, 81, 91.5, 91,  # KC-3
    #     46.5, 49, 44, 50, 58.5, 46.5, 46,  # KR-11
    #     53, 53, 44, 43.5,  # MAL-8
    #     66, 52, 45, 57, 57.5,  # MB-12
    #     44, 43,  # MJY-9
    #     80, 80, 73, 75, 80, 69, 93, 69, 125.5,  # P20
    #     54.5, 71.5, 45, 44, 53, 35, 71, 50.5, 60, 50.5,  # P21
    #     28, 41, 54, 50, 48.5, 57, 44, 46, 52, 51, 41, 39, 67, 40, 39, 67,  # P22
    #     45, 52, 57, 44, 53, 66, 42, 44,  # P23
    #     25, 32, 21, 26, 23.5, 21, 102,  # P24
    #     55, 66, 42, 54, 39, 98, 122, 105, 99, 142, 106,  # P25
    #     32, 27, 34, 25, 29, 28, 34, 25, 60, 50,  # P26
    #     58, 53, 70, 45, 69, 44, 43,  # P27
    #     27, 59, 52, 25, 33, 43, 34.5, 38, 42, 24, 34, 27, 60,  # P28
    #     25, 43, 43, 25, 50.5, 50,  # P29
    #     35, 30, 33, 40.5, 33.5, 40, 44.5, 35,  # P30
    #     37, 55, 41, 55, 37, 37,  # P31
    #     36, 47, 29.5, 45, 40, 29, 43, 32,  # P32
    #     41, 30.5  # SB-6
    # ]
    base_path = r'C:\Users\root\Data\Angiographie_new'
    for path, subfolders, files in walk(base_path):

        video_statistics = get_video_statistics(files)

        if len(video_statistics) == 0:
            continue

        print(path)

        split_path = path.split("\\export\\")
        angle = split_path[1]
        patient = split_path[0].split("\\")[-1]

        print(patient, angle)

        video_statistics.sort()

        gradient_avg = np.array([x[1] for x in video_statistics])
        avg_avg = np.mean(gradient_avg)
        gradient_avg = gradient_avg - avg_avg
        remove_outliers(gradient_avg)
        gradient_moving_average = get_moving_average(gradient_avg, 19)

        first, last = get_frames_with_good_gradient(gradient_moving_average)

        # intensity_std = np.array([x[2] for x in video_statistics])
        # std_avg = np.average(intensity_std)
        # intensity_std = intensity_std - std_avg
        #
        # remove_outliers(intensity_std)
        #
        # max_std_frame = np.argmax(intensity_std)

        # plt.suptitle("Frame with max intensity std")
        # plt.subplot(1, 2, 1)
        # plt.plot(intensity_std)
        # plt.axvline(max_std_frame, color='orange')
        # plt.xlabel("frame # in the sequence")
        # plt.ylabel("intensity std")
        # plt.subplot(1, 2, 2)
        # plt.imshow(video_statistics[max_std_frame][3])
        # plt.show()

        # frequency = get_heartbeat_frequency(f"{patient}, {angle}", intensity_std, (first, last))

        # plt.plot(gradient_avg, color='black')
        # plt.plot(gradient_moving_average, color='gray')
        # plt.axvspan(first, last, color='g')
        # plt.axhline(0, color='r')
        # plt.title("avg gradient intensity")
        # plt.xlabel("frame # in the sequence")
        # plt.ylabel("gradient intensity")
        # plt.show()

        # rfile = open(path + "/temp/seg/relevant_frames.txt", "r")
        # line = rfile.readline()
        # rfile.close()
        # # print(len(gradient_avg), "frames [", first, ",", last, "] vs", line)
        if 0 <= first < last:
            # if not 0.35 < (contracted_ground_truth[i] - first) / (last - first) < 0.65:
            #     print(f"{patient} {angle}: key frame {contracted_ground_truth[i]} is not centered between {first} and {last}, should be around {first + (last - first) / 2}")
            wfile = open(path + "/relevant_frames.txt", "w+")
            # wfile.write(str(first) + ";" + str(last) + ";" + str(hb_ground_truth[i]) + ";" + str(contracted_ground_truth[i]))
            wfile.write(str(first) + ";" + str(last))
            wfile.close()
            # print(len(video_statistics), f"frames [{first}, {last}] @{frequency} max {max_std_frame}")
        else:
            print("Sequence is invalid")
        # i += 1
