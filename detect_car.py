import glob
import os
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from utils import *
from svm_classify import run_svm_classifier
import ProcessFrame

run_classifier = False

test_files = []
test_base = './test_images/'
test_files.extend(glob.glob(test_base + '/*.jpg'))
print(test_files)

# test_img = mpimg.imread(test_files[2])
# plt.figure()
# plt.imshow(test_img)
# plt.show()

process_frame = ProcessFrame.ProcessFrame()
params = process_frame.setup_params()

if run_classifier:
    run_svm_classifier(params)

svm_dict = pickle.load(open("svm_pickle.p", "rb"))
clf = svm_dict["clf"]
scaler = svm_dict["scaler"]

# x_start_stop=[None, None]
# y_start_stop=[400, 650]#[None, None]


for file in test_files:
    t1 = time.time()
    test_img = mpimg.imread(file)
    show_img = test_img.copy()
    #print("test img min and max", np.min(test_img), np.max(test_img))
    #slide_window_tup1 = (32, (10, 6), (400, 480))
    slide_window_tup1 = (64, (5, 4), (400, 550))
    slide_window_tup2 = (96, (5, 5), (400, 650))
    slide_window_params = [slide_window_tup1, slide_window_tup2]
    process_frame.find_cars(test_img, slide_window_params, svc=clf, X_scaler=scaler, params=params, show_search_windows=True)
    img_boxes, search_windows = process_frame.box_dict['img_boxes'], process_frame.box_dict['search_windows']
    heat_map = process_frame.box_dict['heat_map']
    print("img boxes", len(img_boxes))
    detect_car_img = draw_boxes(show_img, search_windows)

    t2 = time.time()
    print(round(t2 - t1, 3), 'Seconds to process image')
    #plt.imshow(detect_car_img)
    show_image(detect_car_img, heat_map, titles=['Original Image', 'Heat map'], cmaps=[None, 'hot'], disp_flag=True)
    plt.show()

