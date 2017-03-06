import matplotlib.image as mpimg
from utils import *
from hog_classify import get_hog_features
from color_classify import bin_spatial, color_hist


def extract_features(image, filenames, params, hog_vis = False, single_image_flag = True):
    """Function to extract all features required for training a classifier"""
    if hog_vis and params['hog_channel']  == 'ALL' and single_image_flag:
        print('Hog vis is possible only for single channel')
        return

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
    hog_feat = params['hog_feat']

    # Create a list to append feature vectors to
    features = []
    if single_image_flag:
        filenames = ['single_image_mode']
    for file in filenames:
        # Read in each one by one
        if file != 'single_image_mode':
            image = mpimg.imread(file)
        feature_image = convert_color_space(image, cspace)
        spatial_features, hist_features, hog_features = [], [], []
        if spatial_feat:
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, size=spatial_size)

        if hist_feat:
            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append the new feature vector to the features list

        if hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                if hog_vis == True:
                    hog_features, hog_img = get_hog_features(feature_image[:,:,hog_channel], orient,
                                pix_per_cell, cell_per_block, vis=True, feature_vec=True)
                else:
                    hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                             pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    if hog_vis:
        return features, hog_img
    else:
        return features


def convert_color_space(image, cspace):
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