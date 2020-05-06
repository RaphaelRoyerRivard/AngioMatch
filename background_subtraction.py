import argparse
import cv2
import numpy as np
import os
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
from matplotlib import pyplot as plt


def subtract_background_naively(background_image, input_image):
    background_image = 255 - background_image
    input_image = 255 - input_image
    image = input_image - background_image
    zeros = image < 0
    image[zeros] = 0
    return image


# Tried a bunch of image manipulation before optical flow but nothing helped
# def subtract_background_with_optical_flow(background_image, input_image, backward=False):
#     background_image = background_image / 255  # copies the image so we don't edit the original
#     input_image /= 255
#
#     subtracted_background = input_image - background_image
#     subtracted_background[subtracted_background < 0] = 0
#
#     background_minus_input = background_image - input_image
#     background_minus_input[background_minus_input < 0] = 0
#
#     background_input_diff = np.abs(background_image - input_image)
#
#     input_diff = input_image - subtracted_background
#     input_diff[input_diff < 0] = 0
#
#     # diff_add = input_diff + background_input_diff
#     cleaned_background = background_image + subtracted_background
#     cleaned_background[cleaned_background > 1] = 1
#
#     cleaned_input = input_image + background_minus_input
#     cleaned_input[cleaned_input > 1] = 1
#
#     plt.subplot(2, 4, 1)  # Rows, columns, index
#     plt.imshow(background_image, cmap='gray', vmin=0, vmax=1)
#     plt.title("background_image")
#
#     plt.subplot(2, 4, 2)
#     plt.imshow(background_minus_input, cmap='gray', vmin=0, vmax=1)
#     plt.title("background_minus_input")
#
#     plt.subplot(2, 4, 3)
#     plt.imshow(background_input_diff, cmap='gray', vmin=0, vmax=1)
#     plt.title("background_input_diff")
#
#     plt.subplot(2, 4, 4)
#     plt.imshow(cleaned_background, cmap='gray', vmin=0, vmax=1)
#     plt.title("cleaned_background")
#
#     plt.subplot(2, 4, 5)
#     plt.imshow(input_image, cmap='gray', vmin=0, vmax=1)
#     plt.title("input_image")
#
#     plt.subplot(2, 4, 6)
#     plt.imshow(subtracted_background, cmap='gray', vmin=0, vmax=1)
#     plt.title("subtracted_background")
#
#     plt.subplot(2, 4, 7)
#     plt.imshow(input_diff, cmap='gray', vmin=0, vmax=1)
#     plt.title("input_diff")
#
#     plt.subplot(2, 4, 8)
#     plt.imshow(cleaned_input, cmap='gray', vmin=0, vmax=1)
#     plt.title("cleaned_input")
#
#     test = cleaned_background - cleaned_input
#     print(test.min(), test.max())
#
#     plt.show()
#
#     nr, nc = input_image.shape  # Use the estimated optical flow for registration
#     row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
#
#     # Compute the optical flow
#     if backward:  # We find the transformation needed to fit the input image onto the background image and apply the opposite transformation to the background image
#         v, u = optical_flow_tvl1(input_image, background_image)
#         inverse_map = np.array([row_coords + v, col_coords + u])
#         v, u = optical_flow_tvl1(subtracted_background, background_minus_input)
#         inverse_map2 = np.array([row_coords - v, col_coords - u])
#
#     else:  # We find the transformation needed to fit the background image onto the input image and apply it to the background image
#         v, u = optical_flow_tvl1(background_image, input_image)
#         inverse_map = np.array([row_coords - v, col_coords - u])
#         v, u = optical_flow_tvl1(background_minus_input, subtracted_background)
#         inverse_map2 = np.array([row_coords - v, col_coords - u])
#
#     # Apply the transformation to the background image
#     warped_background = warp(background_image, inverse_map)
#     warped_background2 = warp(background_image, inverse_map2)
#
#     subtracted_background1 = warped_background - input_image  # To get a white on black image, the logic is reversed
#     subtracted_background1[subtracted_background1 < 0] = 0
#     subtracted_background2 = warped_background2 - input_image  # To get a white on black image, the logic is reversed
#     subtracted_background2[subtracted_background2 < 0] = 0
#
#     differences = np.abs(subtracted_background1 - subtracted_background2) > 0.05
#     combined = subtracted_background1.copy()
#     combined[differences] = 0
#
#     plt.subplot(2, 4, 1)
#     plt.title("Background image")
#     plt.imshow(background_image, cmap='gray')
#     plt.axis("off")
#
#     plt.subplot(2, 4, 2)
#     plt.title("Background minus input")
#     plt.imshow(background_minus_input, cmap='gray', vmin=0, vmax=1)
#     plt.axis("off")
#
#     plt.subplot(2, 4, 3)
#     plt.title("Warped background from raw images")
#     plt.imshow(warped_background, cmap='gray')
#     plt.axis("off")
#
#     plt.subplot(2, 4, 4)
#     plt.title("Optical flow on raw images")
#     plt.imshow(subtracted_background1, cmap='gray')
#     plt.axis("off")
#
#     plt.subplot(2, 4, 5)
#     plt.title("Input image")
#     plt.imshow(input_image, cmap='gray')
#     plt.axis("off")
#
#     plt.subplot(2, 4, 6)
#     plt.title("Input minus background")
#     plt.imshow(subtracted_background, cmap='gray', vmin=0, vmax=1)
#     plt.axis("off")
#
#     plt.subplot(2, 4, 7)
#     plt.title("Warped background from subtracted images")
#     plt.imshow(warped_background2, cmap='gray')
#     plt.axis("off")
#
#     plt.subplot(2, 4, 8)
#     plt.title("Optical flow on subtracted images")
#     plt.imshow(subtracted_background2, cmap='gray')
#     plt.axis("off")
#
#     plt.show()
#
#     subtracted_background1 *= 255
#     return subtracted_background1


def subtract_background_with_optical_flow(background_image, input_image, backward=False):
    background_image = background_image / 255  # copies the image so we don't edit the original
    input_image /= 255

    nr, nc = input_image.shape  # Use the estimated optical flow for registration
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

    # Compute the optical flow
    if backward:  # We find the transformation needed to fit the input image onto the background image and apply the opposite transformation to the background image
        v, u = optical_flow_tvl1(input_image, background_image)
        inverse_map = np.array([row_coords + v, col_coords + u])

    else:  # We find the transformation needed to fit the background image onto the input image and apply it to the background image
        v, u = optical_flow_tvl1(background_image, input_image)
        inverse_map = np.array([row_coords - v, col_coords - u])

    # Show the optical flow on the input image
    h, w = input_image.shape[0], input_image.shape[1]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(mag*4, 255)
    img = cv2.cvtColor((input_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.addWeighted(img, 0.5, rgb, 0.5, 0, rgb)
    cv2.imshow('frame2', rgb)
    cv2.waitKey()

    # Apply the transformation to the background image
    warped_background = warp(background_image, inverse_map)

    subtracted_background = warped_background - input_image  # To get a white on black image, the logic is reversed
    subtracted_background[subtracted_background < 0] = 0

    subtracted_background *= 255
    return subtracted_background


def subtract_background_combined(naive_image, optical_flow_image, optical_flow_backward_image):
    pixels = np.logical_and(naive_image > 0, optical_flow_image > 0)
    pixels = np.logical_and(pixels > 0, optical_flow_backward_image > 0)
    pixel_values = np.zeros(pixels.shape)
    pixel_values[pixels] = (naive_image[pixels] + optical_flow_image[pixels] + optical_flow_backward_image[pixels]) / 3
    return pixel_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, help='Path of the folder in which we can find the angiography sequence.')
    parser.add_argument('--output_folder', default=None, help='Path of the folder in which we want to save the result images. If empty, will create a subtracted_background folder in the input folder.')
    parser.add_argument('--subtraction_type', default='optical_flow', help='The method used for subtracting background. Can be \'naive\', \'optical_flow\' or \'combined\'. When using \'combined\', other methods must have been ran first (optical flow needs to be computed both in normal and backward mode).')
    parser.add_argument('--backward_optical_flow', default=0, help='When computing the optical flow, we can choose to find the transformation needed for the background image to fit the input image and apply it directly (0, normal mode) or the one needed for the input image to fit the background and apply the opposite to the background image (1, backward mode).')
    parser.add_argument('--threshold', default=0, help='All pixels with a value below threshold will be set to 0. Possible values range between 0 and 255.')
    parser.add_argument('--smoothing_kernel_size', default=0, help='If greater than 0, will apply gaussian blurring to the image before applying the threshold. The saved result will not be blurred.')
    parser.add_argument('--binarize', default=False, help='If set to True, all pixels above the threshold will be set to 255.')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    subtraction_type = args.subtraction_type
    backward_mode = args.backward_optical_flow in ['1', 1]
    threshold = int(args.threshold)
    smoothing_kernel_size = int(args.smoothing_kernel_size)
    binarize = bool(args.binarize)

    if input_folder is None:
        raise Exception("Input folder is mandatory")
    if not os.path.exists(input_folder):
        raise Exception("Input folder cannot be found")
    if subtraction_type not in ['naive', 'optical_flow', 'combined']:
        raise Exception("Subtraction type is invalid")
    if output_folder is None:
        output_folder = f"{input_folder}/subtracted_background_{subtraction_type}"
        if threshold > 0:
            output_folder += f"_{threshold}"
        if smoothing_kernel_size > 0:
            output_folder += f"_({smoothing_kernel_size},{smoothing_kernel_size})"
        if binarize:
            output_folder += "_binarized"

    background_img_path = None
    files = os.listdir(input_folder)
    for file in files:
        if file.endswith("Frame1.jpg"):
            background_img_path = f"{input_folder}/{file}"
            break

    if background_img_path is None:
        raise Exception("Could not find a file named '*Frame1.jpg'")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    background_img = cv2.imread(background_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    for file in files:
        if not file.endswith(".jpg"):
            continue

        # if file != "DIRW0019-Frame97.jpg":
        #     continue

        print(file)
        input_img_path = f"{input_folder}/{file}"
        input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if subtraction_type == 'combined':
            naive_img_path = f"{input_folder}/subtracted_background_naive/{file.split('.jpg')[0]}_subtracted_background.png"
            if not os.path.exists(naive_img_path):
                raise Exception(f"Naively subtracted background image not found at {naive_img_path}")
            optical_flow_img_path = f"{input_folder}/subtracted_background_optical_flow/{file.split('.jpg')[0]}_subtracted_background.png"
            if not os.path.exists(optical_flow_img_path):
                raise Exception(f"Subtracted background image from optical flow not found at {optical_flow_img_path}")
            optical_flow_backward_img_path = f"{input_folder}/subtracted_background_optical_flow_backward/{file.split('.jpg')[0]}_subtracted_background.png"
            if not os.path.exists(optical_flow_backward_img_path):
                raise Exception(f"Subtracted background image from optical flow backward not found at {optical_flow_backward_img_path}")
            naive_img = cv2.imread(naive_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            optical_flow_img = cv2.imread(optical_flow_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            optical_flow_backward_img = cv2.imread(optical_flow_backward_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            image = subtract_background_combined(naive_img, optical_flow_img, optical_flow_backward_img)

        elif subtraction_type == 'optical_flow':
            image = subtract_background_with_optical_flow(background_img, input_img, backward=backward_mode)

        else:  # naive
            image = subtract_background_naively(background_img, input_img)

        if smoothing_kernel_size > 0:
            # Apply a Gaussian blur to the image
            smoothed_image = cv2.GaussianBlur(image, (smoothing_kernel_size, smoothing_kernel_size), 0)
            # Apply the threshold
            image[smoothed_image < threshold] = 0

        else:
            # Apply the threshold
            image[image < threshold] = 0

        if binarize:
            image[image > 0] = 255

        output_image_name = f'{file.split(".jpg")[0]}_subtracted_background.png'
        cv2.imwrite(f'{output_folder}/{output_image_name}', image)
