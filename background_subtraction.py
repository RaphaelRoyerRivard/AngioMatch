import argparse
import cv2
import numpy as np
import os
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1


def subtract_background_naively(background_image, input_image):
    background_image = 255 - background_image
    input_image = 255 - input_image
    image = input_image - background_image
    zeros = image < 0
    image[zeros] = 0
    return image


def subtract_background_with_optical_flow(background_image, input_image, backwards=True):
    background_image = background_image / 255  # copies the image so we don't edit the original
    input_image /= 255

    nr, nc = input_image.shape  # Use the estimated optical flow for registration
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

    # Compute the optical flow
    if backwards:
        v, u = optical_flow_tvl1(input_image, background_image)
        inverse_map = np.array([row_coords + v, col_coords + u])
    else:
        v, u = optical_flow_tvl1(background_image, input_image)
        inverse_map = np.array([row_coords - v, col_coords - u])

    # Apply the transformation to the background image
    warped_background = warp(background_image, inverse_map)
    subtracted_background = warped_background - input_image  # To get a white on black image, the logic is reversed
    zeros = subtracted_background < 0
    subtracted_background[zeros] = 0
    subtracted_background *= 255
    return subtracted_background


# def subtract_background_with_optical_flow(background_image, input_image):
#     print("background_img", background_image.min(), background_image.max(), background_image.mean())
#     print("input_image", input_image.min(), input_image.max(), input_image.mean())
#     background_image = background_image / 255  # copies the image so we don't edit the original
#     input_image /= 255
#     print("background_img", background_image.min(), background_image.max(), background_image.mean())
#     print("input_image", input_image.min(), input_image.max(), input_image.mean())
#
#     v, u = optical_flow_tvl1(input_image, background_image)  # Compute the optical flow
#     # v, u = optical_flow_tvl1(background_image, input_image)  # Compute the optical flow
#     print("v", v.shape, v.min(), v.max(), v.mean(), v)
#     print("u", u.shape, u.min(), u.max(), u.mean(), u)
#
#     # Use the estimated optical flow for registration
#     nr, nc = input_image.shape
#     row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
#     print("row_coords", row_coords.shape, row_coords)
#     print("col_coords", col_coords.shape, col_coords)
#
#     inverse_map = np.array([row_coords + v, col_coords + u])
#     # inverse_map = np.array([row_coords - v, col_coords - u])
#     print("inverse_map", inverse_map.shape, inverse_map)
#
#     warped_background = warp(background_image, inverse_map, mode='nearest')
#
#     input_image_name = f'{file.split(".jpg")[0]}_input.png'
#     cv2.imwrite(f'{output_folder}/{input_image_name}', input_image * 255)
#     print("warped_background", warped_background.min(), warped_background.max(), warped_background.mean())
#     warped_background_image_name = f'{file.split(".jpg")[0]}_warped_background.png'
#     cv2.imwrite(f'{output_folder}/{warped_background_image_name}', warped_background * 255)
#
#     subtracted_background = warped_background - input_image
#     print("subtracted_background", subtracted_background.min(), subtracted_background.max(), subtracted_background.mean())
#     zeros = subtracted_background < 0
#     subtracted_background[zeros] = 0
#     print("subtracted_background", subtracted_background.min(), subtracted_background.max(), subtracted_background.mean())
#     subtracted_background *= 255
#     print("subtracted_background", subtracted_background.min(), subtracted_background.max(), subtracted_background.mean())
#     return subtracted_background


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, help='Path of the folder in which we can find the angiography sequence.')
    parser.add_argument('--output_folder', default=None, help='Path of the folder in which we want to save the result images. If empty, will create a subtracted_background folder in the input folder.')
    parser.add_argument('--subtraction_type', default='optical_flow', help='The method used for subtracting background. Either optical_flow or naive.')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    subtraction_type = args.subtraction_type

    if input_folder is None:
        raise Exception("Input folder is mandatory")
    if not os.path.exists(input_folder):
        raise Exception("Input folder cannot be found")
    if subtraction_type not in ['optical_flow', 'naive']:
        raise Exception("Subtraction type is invalid")
    if output_folder is None:
        output_folder = f"{input_folder}/subtracted_background"

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
        print(file)
        input_img_path = f"{input_folder}/{file}"
        input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if subtraction_type == 'optical_flow':
            image = subtract_background_with_optical_flow(background_img, input_img, backwards=False)
        else:  # naive
            image = subtract_background_naively(background_img, input_img)
        output_image_name = f'{file.split(".jpg")[0]}_subtracted_background.png'
        cv2.imwrite(f'{output_folder}/{output_image_name}', image)
