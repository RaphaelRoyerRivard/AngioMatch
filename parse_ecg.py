import numpy as np
from os import walk
from matplotlib import pyplot as plt
from identify_relevant_frames import get_moving_average


MIN_PEAK_RATIO_COMPARED_TO_MEAN = 1.25
MAX_DISTANCE_BETWEEN_PEAK_AND_GRADIENT_PEAK = 10
MIN_GRADIENT_VALUE_FOR_PEAK = 0.05


def standardize(values):
    return (values - values.min()) / (values.max() - values.min())


def find_peaks(standardized_values):
    peaks = np.where(standardized_values > standardized_values.mean() * MIN_PEAK_RATIO_COMPARED_TO_MEAN)[0].tolist()
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


def filter_peaks_based_on_derivative(peaks, derivative_positive_peaks, derivative_negative_peaks):
    filtered_peaks = []
    for peak in peaks:
        positive_peak_distances = derivative_positive_peaks - peak
        negative_peak_distances = derivative_negative_peaks - peak
        if np.absolute(positive_peak_distances).min() < MAX_DISTANCE_BETWEEN_PEAK_AND_GRADIENT_PEAK and np.absolute(negative_peak_distances).min() < MAX_DISTANCE_BETWEEN_PEAK_AND_GRADIENT_PEAK:
            filtered_peaks.append(peak)
    return filtered_peaks


def filter_peaks_based_on_narrowness(data, peaks, max_width):
    filtered_peaks = []
    for peak in peaks:
        # Find local minima before peak
        current = peak - 1
        while current >= 0:
            if data[current] > data[current + 1]:
                break
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
            if data[current] > data[current - 1]:
                break
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


if __name__ == '__main__':
    print("Starting")
    base_path = r'C:\Users\root\Data\new_data'
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
            coarse_moving_average = get_moving_average(ecg, 201, use_gaussian_kernel=False, mode="valid")
            if (ecg.shape[0] - coarse_moving_average.shape[0]) % 2 > 0:
                raise Exception(f"Size difference between ECG ({ecg.shape[0]}) and coarse moving average ({coarse_moving_average.shape[0]}) is odd when it should be even.")
            half_size_diff = int((ecg.shape[0] - coarse_moving_average.shape[0]) / 2)
            coarse_moving_average = np.append(np.ones(half_size_diff) * coarse_moving_average[0], coarse_moving_average)
            coarse_moving_average = np.append(coarse_moving_average, np.ones(half_size_diff) * coarse_moving_average[-1])
            standardized_ecg = standardize(ecg - coarse_moving_average)
            smoothed_ecg = get_moving_average(standardized_ecg, 5)
            standardized_ecg = standardize(smoothed_ecg)
            gradient = standardize(np.gradient(standardized_ecg))

            # Find and filter R-peaks
            ecg_peaks = find_peaks(standardized_ecg)
            gradient_positive_peaks = np.array(find_peaks(gradient))
            gradient_negative_peaks = np.array(find_peaks(1-gradient))
            filtered_peaks = filter_peaks_based_on_derivative(ecg_peaks, gradient_positive_peaks, gradient_negative_peaks)
            more_filtered_peaks = filter_peaks_based_on_narrowness(standardized_ecg, filtered_peaks, 40)

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

            plt.subplots_adjust(hspace=0.45)
            plt.suptitle(f"R-peaks for ({filename})")
            plt.subplot(5, 1, 1)
            plt.title("ECG and coarse moving average")
            plt.plot(ecg)
            plt.plot(coarse_moving_average)
            plt.subplot(5, 1, 2)
            plt.title("Unfiltered peaks")
            plt.vlines(ecg_peaks, 0, 1, 'red')
            plt.hlines(standardized_ecg.mean() * MIN_PEAK_RATIO_COMPARED_TO_MEAN, 0, len(standardized_ecg), 'purple', zorder=2)
            plt.plot(standardized_ecg, zorder=1)
            plt.subplot(5, 1, 3)
            plt.title("Positive and negative peaks of gradient")
            plt.vlines(gradient_positive_peaks, 0, 1, 'red')
            plt.vlines(gradient_negative_peaks, 0, 1, 'green')
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

            # Calculate R-peak positions in video percentage
            positions = np.array(filtered_peaks) / len(ecg)

            # Save results
            file = open(path + "\\" + filename.replace("ecg.txt", "ecg.r-peaks.txt"), 'w')
            file.writelines(str(positions))
            file.close()
