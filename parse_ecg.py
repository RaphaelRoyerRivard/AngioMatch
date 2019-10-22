import numpy as np
from os import walk
from matplotlib import pyplot as plt
from identify_relevant_frames import get_moving_average


MIN_PEAK_RATIO_COMPARED_TO_MEAN = 1.0
MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN = 1.1
MAX_DISTANCE_BETWEEN_PEAK_AND_GRADIENT_PEAK = 20
MIN_GRADIENT_VALUE_FOR_PEAK = 0.05
MAX_PEAK_WIDTH = 50
MIN_GRADIENT_PEAKS_DIFFERENCE = 0.3
MIN_DISTANCE_BETWEEN_PEAKS = 30


def standardize(values):
    return (values - values.min()) / (values.max() - values.min())


def find_peaks(standardized_values, filter=None, min_peak_ratio=None):
    if filter is not None:
        peaks = np.where(filter)[0].tolist()
    elif min_peak_ratio is not None:
        peaks = np.where(standardized_values > standardized_values.mean() * min_peak_ratio)[0].tolist()
    else:
        return []
    filtered_peaks = []
    last_max_index = -1
    last_index = -2
    # Find maximums of each peak
    for peak in peaks:
        if peak > last_index + 1:
            if last_max_index >= 0:
                filtered_peaks.append(last_max_index)
            last_max_index = peak
        if standardized_values[peak] > standardized_values[last_max_index]:
            last_max_index = peak
        last_index = peak
    if last_max_index >= 0:
        filtered_peaks.append(last_max_index)
    return filtered_peaks


def filter_peaks_based_on_derivative(peaks, derivative_positive_peaks, derivative_negative_peaks, derivative_values):
    filtered_peaks = []
    for peak in peaks:
        # Compute distances between the current peak and all the gradient peaks
        positive_peak_distances = peak - derivative_positive_peaks
        negative_peak_distances = derivative_negative_peaks - peak
        # Find the valid distances (positive peaks must be before and negative peaks must be after)
        valid_positive_peak_distances_indices = np.where(positive_peak_distances >= 0)[0]
        valid_negative_peak_distances_indices = np.where(negative_peak_distances >= 0)[0]
        # If there is no valid gradient positive peak or no valid gradient negative peak, the current peak is not valid
        if len(valid_positive_peak_distances_indices) == 0 or len(valid_negative_peak_distances_indices) == 0:
            continue
        # Filter out negative distances
        positive_peak_distances = positive_peak_distances[valid_positive_peak_distances_indices]
        negative_peak_distances = negative_peak_distances[valid_negative_peak_distances_indices]
        # If the closest gradient positive peak or closest gradient negative peak is too far, the current peak is not valid
        if positive_peak_distances.min() > MAX_DISTANCE_BETWEEN_PEAK_AND_GRADIENT_PEAK or negative_peak_distances.min() > MAX_DISTANCE_BETWEEN_PEAK_AND_GRADIENT_PEAK:
            continue
        # Compute the difference in value between the closest gradient positive and closest gradient negative peaks
        closest_gradient_positive_peak = derivative_positive_peaks[valid_positive_peak_distances_indices[positive_peak_distances.argmin()]]
        closest_gradient_negative_peak = derivative_negative_peaks[valid_negative_peak_distances_indices[negative_peak_distances.argmin()]]
        gradient_peaks_difference = derivative_values[closest_gradient_positive_peak] - derivative_values[closest_gradient_negative_peak]
        if gradient_peaks_difference < MIN_GRADIENT_PEAKS_DIFFERENCE:
            continue
        filtered_peaks.append(peak)
    return filtered_peaks


def filter_peaks_based_on_narrowness(data, peaks, max_width):
    filtered_peaks = []
    for peak in peaks:
        # Find local minima before peak
        current = peak - 1
        while current >= 0:
            # If the current value is lower than the next one or if they are almost identical but far enough from the peak
            if data[current] > data[current + 1] or (peak - current > 5 and data[current + 1] - data[current] < 0.001):
                break
            # Moving to the left
            current = current - 1
        if current < 0:
            if peak < 2 or np.gradient(data[0:peak]).max() < MIN_GRADIENT_VALUE_FOR_PEAK:
                if peak >= 2:
                    print("Early gradient", np.gradient(data[0:peak]))
                continue
        previous_min_index = current + 1

        # Find local minima after peak
        current = peak + 1
        while current < len(data):
            # If the current value is lower than the previous one or if they are almost identical but far enough from the peak
            if data[current] > data[current - 1] or (current - peak > 5 and data[current - 1] - data[current] < 0.001):
                break
            # Moving to the right
            current = current + 1
        if current >= len(data):
            if len(data) - peak < 2 or np.abs(np.gradient(data[peak:])).max() < MIN_GRADIENT_VALUE_FOR_PEAK:
                if len(data) - peak >= 2:
                    print("End gradient", np.gradient(data[peak:]))
                continue
        next_min_index = current + 1

        if abs(previous_min_index - next_min_index) <= max_width:
            filtered_peaks.append(peak)
    return filtered_peaks


def filter_close_peaks(data, peaks):
    invalid_peaks = []
    peaks = np.array(peaks)
    for i in range(1, len(peaks)):
        distances = peaks[i] - peaks[:i+1]
        close_peaks = np.where(distances < MIN_DISTANCE_BETWEEN_PEAKS)[0]
        if len(close_peaks) > 1:
            highest_peak = peaks[close_peaks[data[peaks[close_peaks]].argmax()]]
            for peak in peaks[close_peaks]:
                if peak != highest_peak:
                    invalid_peaks.append(peak)
    filtered_peaks = []
    for peak in peaks:
        if peak not in invalid_peaks:
            filtered_peaks.append(peak)
    return filtered_peaks


if __name__ == '__main__':
    print("Starting")
    base_path = r'C:\Users\root\Data\new_data'
    previous_peaks = None
    for path, subfolders, files in walk(base_path):
        print(path)
        for filename in files:
            if not filename.endswith("ecg.txt"):
                continue

            print(filename)
            # Read ECG file
            file = open(path + "\\" + filename, 'r')
            lines = file.readlines()
            file.close()
            ecg = np.array([int(line) for line in lines])

            # Smooth and standardize ECG values
            coarse_moving_average = get_moving_average(ecg, 201, use_gaussian_kernel=False, mode="valid", pad_values=True)
            finer_moving_average = get_moving_average(ecg, 51, use_gaussian_kernel=False, mode="valid", pad_values=True)
            over_moving_average = ecg > finer_moving_average
            standardized_ecg = standardize(ecg - coarse_moving_average)
            smoothed_ecg = get_moving_average(standardized_ecg, 3)
            standardized_ecg = standardize(smoothed_ecg)
            gradient = standardize(np.gradient(standardized_ecg))

            # Find and filter R-peaks
            # ecg_peaks = find_peaks(standardized_ecg, MIN_PEAK_RATIO_COMPARED_TO_MEAN)
            ecg_peaks = find_peaks(standardized_ecg, filter=over_moving_average)
            gradient_positive_peaks = np.array(find_peaks(gradient, min_peak_ratio=MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN))
            gradient_negative_peaks = np.array(find_peaks(1 - gradient, min_peak_ratio=MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN))
            filtered_peaks = filter_peaks_based_on_derivative(ecg_peaks, gradient_positive_peaks, gradient_negative_peaks, gradient)
            more_filtered_peaks = filter_peaks_based_on_narrowness(standardized_ecg, filtered_peaks, MAX_PEAK_WIDTH)
            more_filtered_peaks = np.array(filter_close_peaks(standardized_ecg, more_filtered_peaks))

            # Validate R-peaks
            # diffs = []
            # for i in range(1, len(more_filtered_peaks)):
            #     diffs.append(more_filtered_peaks[i] - more_filtered_peaks[i-1])
            # diffs = np.array(diffs)
            # print("Diff std & mean & ratio", diffs.std(), diffs.mean(), diffs.mean() / diffs.std())
            # if diffs.mean() / diffs.std() < 10:  # STD must not be too high in proportion with the mean
            #     from matplotlib import pyplot as plt
            #     plt.subplots_adjust(hspace=0.4)
            #     plt.subplot(5, 1, 1)
            #     plt.title("ECG and coarse moving average")
            #     plt.plot(ecg)
            #     plt.plot(coarse_moving_average)
            #     plt.subplot(5, 1, 2)
            #     plt.title("Unfiltered peaks")
            #     plt.vlines(ecg_peaks, 0, 1, 'red')
            #     plt.plot(standardized_ecg)
            #     plt.subplot(5, 1, 3)
            #     plt.title("Positive and negative peaks of gradient")
            #     plt.vlines(gradient_positive_peaks, 0, 1, 'red')
            #     plt.vlines(gradient_negative_peaks, 0, 1, 'green')
            #     plt.plot(gradient)
            #     plt.subplot(5, 1, 4)
            #     plt.title("R-peaks filtered with gradient")
            #     plt.vlines(filtered_peaks, 0, 1, 'red')
            #     plt.plot(standardized_ecg)
            #     plt.subplot(5, 1, 5)
            #     plt.title("R-peaks filtered with gradient and narrowness")
            #     plt.vlines(more_filtered_peaks, 0, 1, 'red')
            #     plt.plot(standardized_ecg)
            #     plt.show()
            #     answer = None
            #     while answer not in ["y", "n"]:
            #         answer = input("Standard deviation of distance between R-peaks is unusually high. Should we keep them? [y/n]\n")
            #     if answer == "n":
            #         raise Exception(f"Standard deviation of distance between R-peaks ({diffs.std()}) is too high compared to the mean of {diffs.mean()}")

            if previous_peaks is None or len(more_filtered_peaks) != len(previous_peaks) or (more_filtered_peaks - previous_peaks).std() > 0:
                plt.subplots_adjust(hspace=0.45)
                plt.suptitle(f"R-peaks for ({filename})")
                plt.subplot(5, 1, 1)
                plt.title("ECG and coarse moving average")
                plt.plot(ecg)
                plt.plot(coarse_moving_average)
                plt.plot(finer_moving_average)
                plt.subplot(5, 1, 2)
                plt.title("Unfiltered peaks")
                plt.vlines(ecg_peaks, 0, 1, 'red')
                plt.plot(standardized_ecg, zorder=1)
                plt.subplot(5, 1, 3)
                plt.title("Positive and negative peaks of gradient")
                plt.vlines(gradient_positive_peaks, 0, 1, 'red')
                plt.vlines(gradient_negative_peaks, 0, 1, 'green')
                plt.hlines(gradient.mean() * MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN, 0, len(standardized_ecg), 'purple', zorder=2)
                plt.plot(gradient)
                plt.subplot(5, 1, 4)
                plt.title("R-peaks filtered with gradient")
                plt.vlines(filtered_peaks, 0, 1, 'red')
                plt.plot(standardized_ecg)
                plt.subplot(5, 1, 5)
                plt.title("R-peaks filtered with gradient and narrowness")
                plt.vlines(more_filtered_peaks, 0, 1, 'red', zorder=2)
                plt.plot(standardized_ecg, zorder=1)
                plt.show()

            previous_peaks = more_filtered_peaks

            # Calculate R-peak positions in video percentage
            positions = more_filtered_peaks / len(ecg)

            # Save results
            file = open(path + "\\" + filename.replace("ecg.txt", "ecg.r-peaks.txt"), 'w')
            file.writelines(str(positions))
            file.close()
