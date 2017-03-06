from collections import deque
from utils import *
from hog_classify import get_hog_features
from color_classify import bin_spatial, color_hist

class ProcessFrame():
    def __init__(self):
        self.box_dict = {}
        self.thresh_box_dict = {}
        # ring buffer to keep track of params for the last 35 frames
        self.result_buffer = deque(maxlen=35)
        self.apply_sliding_window_flag = False
        self.debug = False

    def setup_params(self):
        """Method to set up the params for the feature extraction"""
        params = {}
        # hog params
        params['orient'] = 9  # 9
        params['pix_per_cell'] = 12  # 6 #8 #12 works!!!!!!
        params['cell_per_block'] = 2  # 2
        params['hog_channel'] = "ALL"  # "ALL" # Can be 0, 1, 2, or "ALL"

        # for color and hist features
        params['colorspace'] = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        params['spatial'] = 16
        params['histbin'] = 32 #64  # 32

        # Flags for features
        params['spatial_feat'] = True
        params['hist_feat'] = True
        params['hog_feat'] = True
        return params

    def find_cars(self, img, slide_window_params, svc, X_scaler, params, show_search_windows = False):
        """Main method that perform sliding window search and predictions"""
        # get params from passed in dict
        orient = params['orient']
        pix_per_cell = params['pix_per_cell']
        cell_per_block = params['cell_per_block']
        hog_channel = params['hog_channel']

        # for color and hist features
        cspace = params['colorspace']
        spatial_size = (params['spatial'], params['spatial'])
        hist_bins = params['histbin']

        # Flags for features
        spatial_feat = params['spatial_feat']
        hist_feat = params['hist_feat']
        hog_feat = params['hog_feat'] # note - hog_feat must always be true

        # define a heat map
        heat_map = np.zeros_like(img[:, :, 0])
        img_boxes, search_windows = [], []

        # make seperate copy for draw image
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        for slide_window_tup in slide_window_params:
            w_size, overlap_pixels, y_start_stop = slide_window_tup
            #ystart, ystop = y_start_stop[0], y_start_stop[1]
            # compute scale
            scale = w_size / 64.
            ystart, ystop = y_start_stop[0], y_start_stop[1]
            #print("scale, ystart, ystop", scale, ystart, ystop)
            img_tosearch = img[ystart:ystop, :, :]
            ctrans_tosearch = self.convert_color_space(img_tosearch, cspace)
            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // pix_per_cell)
            nyblocks = (ch1.shape[0] // pix_per_cell)

            nfeat_per_block = orient * cell_per_block ** 2
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // pix_per_cell) - 1
            cells_per_step_x, cells_per_step_y = overlap_pixels
            nxsteps = (nxblocks) // cells_per_step_x
            nysteps = (nyblocks) // cells_per_step_y

            #print("Total windows, nx, ny", nxsteps * nysteps, nxsteps, nysteps)

            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb * cells_per_step_y
                    xpos = xb * cells_per_step_x
                    x_pos_end, y_pos_end = xpos + nblocks_per_window, ypos + nblocks_per_window
                    # adjust last sliding window if it overshoots
                    if y_pos_end >= hog1.shape[0]:
                        delta = y_pos_end - hog1.shape[0] + 1
                        ypos -= delta
                        y_pos_end = ypos + nblocks_per_window
                    if x_pos_end >= hog1.shape[1]:
                        delta = x_pos_end - hog1.shape[1] + 1
                        xpos -= delta
                        x_pos_end = xpos + nblocks_per_window
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:y_pos_end, xpos:x_pos_end].ravel()
                    hog_feat2 = hog2[ypos:y_pos_end, xpos:x_pos_end].ravel()
                    hog_feat3 = hog3[ypos:y_pos_end, xpos:x_pos_end].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos * pix_per_cell
                    ytop = ypos * pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                    spatial_features, hist_features = [], []
                    # Get color features
                    if spatial_feat:
                        spatial_features = bin_spatial(subimg, size=spatial_size)
                    if hist_feat:
                        hist_features = color_hist(subimg, nbins=hist_bins)

                    # Scale features and make a prediction
                    test_features = X_scaler.transform(
                        np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                    # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                    test_prediction = svc.predict(test_features)

                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    if show_search_windows:
                        search_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                        self.box_dict['search_windows'] = search_windows

                    if test_prediction == 1:
                        heat_map[ytop_draw + ystart: ytop_draw + win_draw + ystart, xbox_left: xbox_left + win_draw] += 1
                        img_boxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
        self.box_dict['img_boxes'] = img_boxes
        self.box_dict['heat_map'] = heat_map
        self.result_buffer.append(self.box_dict.copy())
        return #draw_img, heat_map

    def convert_color_space(self, image, cspace):
        """Method to convert between colorspaces"""
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        return feature_image

    def update_thresh_box(self):
        """Method to compute the running average of the values in the ring buffer"""
        # print(result_buffer)
        total = len(self.result_buffer)
        thresh = 0
        if total < 4:
            thresh = 2
        elif total < 8:
            thresh = 3
        else:
            thresh = 15 #12 #5

        thresh_heat_map = np.zeros_like(self.box_dict['heat_map'])
        for i in range(total):
            calc_dict = self.result_buffer[i]
            thresh_heat_map += calc_dict['heat_map']
        thresh_heat_map[thresh_heat_map < thresh] = 0
        self.thresh_box_dict['heat_map'] = thresh_heat_map
        return

    def draw_labeled_bboxes(self, img, labels):
        """Method to draw bounding boxes"""
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img




