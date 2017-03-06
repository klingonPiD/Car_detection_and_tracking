
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *
from extract_features import extract_features
import pickle

def run_svm_classifier(params, pickle_flag = True):
    #sample_size = 5000
    # get data
    cars, non_cars = get_filenames()
    num_cars, num_non_cars = len(cars), len(non_cars)
    print("Number of cars, non-cars", num_cars, num_non_cars)

    # Reduce the sample size because HOG features are slow to compute
    #cars = cars[0:sample_size]
    #notcars = non_cars[0:sample_size]

    t=time.time()
    image = None
    car_features = extract_features(image, cars, params, single_image_flag=False )
    notcar_features = extract_features(image, non_cars, params, single_image_flag=False )
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    if params['spatial_feat']:
        print('Using spatial binning of:',params['spatial'])
    if params['hist_feat']:
        print('Using histogram bins:', params['histbin'])
    if params['hog_feat']:
        print('Using:',params['orient'],'orientations',params['pix_per_cell'],
            'pixels per cell and', params['cell_per_block'],'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    # t=time.time()
    # n_predict = 10
    # print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    # print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    # t2 = time.time()
    # print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    if pickle_flag:
        #save the classifier and scaler for reuse
        svm_dict = {}
        svm_dict['clf'] = svc
        svm_dict['scaler'] = X_scaler
        pickle.dump( svm_dict, open( "./svm_pickle.p", "wb" ) )
        print("Pickling done")