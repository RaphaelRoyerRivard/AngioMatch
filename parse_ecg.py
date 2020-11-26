import numpy as np
from os import walk, remove
from matplotlib import pyplot as plt
from identify_relevant_frames import get_moving_average
import argparse


MIN_PEAK_RATIO_COMPARED_TO_MEAN = 1.0
MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN = 1.1
MAX_DISTANCE_BETWEEN_PEAK_AND_GRADIENT_PEAK = 20
MIN_GRADIENT_VALUE_FOR_PEAK = 0.05
MAX_PEAK_WIDTH = 50
MIN_GRADIENT_PEAKS_DIFFERENCE = 0.3
MIN_DISTANCE_BETWEEN_PEAKS = 30
MOVING_AVERAGE_SIZE = 201
FINER_MOVING_AVERAGE_SIZE = 51


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='.', help='input folder in which to search recursively for ecg files (end with .ecg.txt)')
    parser.add_argument('--intermediate_folder_count', default=0, help='number of folders between the patient folder and the sequence files')
    parser.add_argument('--sequence_number_index', default=-1, help='index of the sequence number in the file name. if left to default, will use the whole file name')
    parser.add_argument('--save_files', default="True", help='whether to save the r-peaks files or not (default True)')
    args = parser.parse_args()

    input_folder = args.input_folder
    intermediate_folder_count = int(args.intermediate_folder_count)
    sequence_number_index = int(args.sequence_number_index)
    save_files = args.save_files == "True"

    show_sequences = [
        ("test", '0'),
    ]

    previous_peaks = None  # Used to prevent to show 2 identical ECG signals in a row (happens with biplan sequences)
    for path, subfolders, files in walk(input_folder):
        print(path)
        for filename in files:
            if not filename.endswith("ecg.txt"):
                continue

            split_path = path.split("\\")
            patient = split_path[-(intermediate_folder_count+1)]
            print("patient:", patient)
            print(filename)
            split_filename = filename.split(".")
            sequence = split_filename[0] if sequence_number_index < 0 else int(split_filename[0][sequence_number_index:])
            print("sequence:", sequence)
            # Read ECG file
            file = open(path + "\\" + filename, 'r')
            lines = file.readlines()
            file.close()
            frame_count = int(lines[0])
            ecg = np.array([int(line) for line in lines[1:]])

            # Smooth and standardize ECG values
            moving_average_size = min(ecg.shape[0], MOVING_AVERAGE_SIZE)
            if moving_average_size % 2 == 0:
                moving_average_size -= 1
            finer_moving_average_size = min(ecg.shape[0], FINER_MOVING_AVERAGE_SIZE)
            if finer_moving_average_size % 2 == 0:
                finer_moving_average_size -= 1
            coarse_moving_average = get_moving_average(ecg, moving_average_size, use_gaussian_kernel=False, mode="valid", pad_values=True)
            finer_moving_average = get_moving_average(ecg, finer_moving_average_size, use_gaussian_kernel=False, mode="valid", pad_values=True)
            over_moving_average = ecg > finer_moving_average
            standardized_ecg = standardize(ecg - coarse_moving_average)
            smoothed_ecg = get_moving_average(standardized_ecg, 3)
            standardized_ecg = standardize(smoothed_ecg)
            gradient = standardize(np.gradient(standardized_ecg))

            # Find and filter R-peaks
            # ecg_peaks = find_peaks(standardized_ecg, MIN_PEAK_RATIO_COMPARED_TO_MEAN)
            ecg_peaks = np.array(find_peaks(standardized_ecg, filter=over_moving_average))
            gradient_positive_peaks = np.array(find_peaks(gradient, min_peak_ratio=MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN))
            gradient_negative_peaks = np.array(find_peaks(1 - gradient, min_peak_ratio=MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN))
            filtered_peaks = filter_peaks_based_on_derivative(ecg_peaks, gradient_positive_peaks, gradient_negative_peaks, gradient)
            more_filtered_peaks = filter_peaks_based_on_narrowness(standardized_ecg, filtered_peaks, MAX_PEAK_WIDTH)
            more_filtered_peaks = np.array(filter_close_peaks(standardized_ecg, more_filtered_peaks))

            valid = True
            # Calculate the distances between peaks
            distances = []
            for i in range(1, len(more_filtered_peaks)):
                distances.append(more_filtered_peaks[i] - more_filtered_peaks[i-1])
            if len(distances) > 0:
                distances = np.array(distances)
                mean = distances.mean()
                # We need at least 3 peaks to compute the standard deviation
                if len(distances) > 1:
                    normalized_std = distances.std() * np.sqrt(len(distances)) / mean
                    if normalized_std > 0.6:
                        print(f"Normalized std between peaks is too high ({normalized_std})")
                        valid = False
                if more_filtered_peaks[0] - mean * 1.3 > 0 or more_filtered_peaks[-1] + mean * 1.3 < len(ecg):
                    print("Missing peak at the", "start" if more_filtered_peaks[0] - mean * 1.3 > 0 else "end")
                    valid = False
            else:
                print("Not enough peaks")
                valid = False

            # Show the R-peaks of selected sequences
            # if not valid:
            if (patient, sequence) in show_sequences:  # and (previous_peaks is None or len(more_filtered_peaks) != len(previous_peaks) or (more_filtered_peaks - previous_peaks).std() > 0):
                plt.subplots_adjust(hspace=0.45)
                plt.suptitle(f"R-peaks for {patient} {sequence}")
                plt.subplot(4, 1, 1)
                plt.title("ECG and coarse moving average")
                plt.plot(ecg)
                plt.plot(coarse_moving_average)
                plt.plot(finer_moving_average)
                plt.subplot(4, 1, 2)
                plt.title("Unfiltered peaks")
                plt.vlines(ecg_peaks, 0, 1, 'orange')
                plt.plot(standardized_ecg, zorder=1)
                plt.subplot(4, 1, 3)
                plt.title("Positive and negative peaks of gradient")
                plt.vlines(gradient_positive_peaks, 0, 1, 'green')
                plt.vlines(gradient_negative_peaks, 0, 1, 'red')
                # plt.hlines(gradient.mean() * MIN_GRADIENT_PEAK_RATIO_COMPARED_TO_MEAN, 0, len(standardized_ecg), 'purple', zorder=2)
                plt.plot(gradient)
                # plt.subplot(5, 1, 4)
                # plt.title("R-peaks filtered with gradient")
                # plt.vlines(filtered_peaks, 0, 1, 'red')
                # plt.plot(standardized_ecg)
                # plt.subplot(5, 1, 5)
                plt.subplot(4, 1, 4)
                plt.title("R-peaks filtered with gradient and narrowness")
                plt.vlines(more_filtered_peaks, 0, 1, 'red', zorder=2)
                # plt.plot(standardized_ecg, zorder=1)
                plt.plot(standardized_ecg)
                plt.show()

            # previous_peaks = more_filtered_peaks

            # Calculate R-peak positions in video percentage
            positions = more_filtered_peaks / len(ecg)

            # Save results
            if save_files:
                output_file_name = path + "\\" + filename.replace("ecg.txt", "ecg.r-peaks")
                if valid:
                    np.save(output_file_name, positions)  # This one creates a .npy file
                    file = open(output_file_name + ".txt", 'w')
                    file.writelines(str(positions * frame_count))
                    file.close()
                elif filename.replace("ecg.txt", "ecg.r-peaks.txt") in files:
                    print("Deleting r-peaks files that were already saved")
                    remove(output_file_name + ".npy")
                    remove(output_file_name + ".txt")
