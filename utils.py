
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os

def get_filenames():
    """Read in car and non-car images"""
    cars, non_cars = [], []
    car_base = './vehicles/vehicles/'
    non_car_base = './non-vehicles/non-vehicles/'
    car_folders, non_car_folders = os.listdir(car_base), os.listdir(non_car_base)
    [cars.extend(glob.glob(car_base + folder + '/*.png')) for folder in car_folders]
    [non_cars.extend(glob.glob(non_car_base + folder + '/*.png')) for folder in non_car_folders]
    return cars, non_cars

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    [cv2.rectangle(draw_img, pt1, pt2, color, thick) for pt1, pt2 in bboxes]
    return draw_img # Change this line to return image copy with boxes

def show_image(orig_img, grad_img, titles=['Original Image', 'Transformed Image'], cmaps = [None, None],disp_flag = False):
    """Method to display pairs of images"""
    # Plot the result
    if disp_flag:
        f, (ax1, ax2) = plt.subplots(1, 2) #figsize=(18, 7)
        f.tight_layout()
        ax1.imshow(orig_img, cmap = cmaps[0])
        ax1.set_title(titles[0], fontsize=25)
        ax2.imshow(grad_img, cmap = cmaps[1])
        ax2.set_title(titles[1], fontsize=25)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()