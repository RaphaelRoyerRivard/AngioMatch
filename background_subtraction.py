import argparse
import cv2
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, help='path of the folder in which we can find the angiography sequence')
    parser.add_argument('--output_folder', default=None, help='path of the folder in which we want to save the result images')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if input_folder is None:
        raise Exception("Input folder is mandatory")
    if not os.path.exists(input_folder):
        raise Exception("Input folder cannot be found")
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

    background_img = 255 - np.float32(cv2.imread(background_img_path, cv2.IMREAD_GRAYSCALE))
    for file in files:
        if not file.endswith(".jpg"):
            continue
        input_img_path = f"{input_folder}/{file}"
        input_img = 255 - np.float32(cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE))
        image = input_img - background_img
        zeros = image < 0
        image[zeros] = 0
        output_image_name = f'{file.split(".jpg")[0]}_subtracted_background.png'
        cv2.imwrite(f'{output_folder}/{output_image_name}', image)
