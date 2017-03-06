# Project 3: Vehicle detection and tracking
### Ajay Paidi

# Objective
The objective of this project is to develop a software pipeline to identify and track cars in a video from a front-facing camera on a car. The software pipeline will use machine learning and computer vision approaches to detect and track cars and should be reasonably robust enough to handle different situations (multiple cars, lighting conditions, etc.)

# File structure
- **ReadMe.md**: This file
- **color_classify.py**: Script with functions to extract spatial and color histogram features from images.
- **hog_classify.py**: Script with functions to extract hog features from images.
- **svm_classify.py**: Script that trains the linear SVC model.
- **extract_features.py**: Script called upon by svm_classify.py to extract features.
- **ProcessFrame.py**: Class with methods implementing sliding windows  to search, classify and track cars in an image.
- **utils.py**: Utility script for visualizing images, reading files and drawing bounding boxes.
- **detect_car.py**: Script that runs the classifier and sliding window search on the sample test images.
- **vehicle_detect_pipeline.py**: Main script that implements the software pipeline to detect and track cars on videos.
- **demo_vehicle_detection.ipynb**: Python notebook that demonstrates some of the concepts in the software pipeline.

# Approach

My approach to solving the problem can be broadly categorized into these steps

### Data visualization
This involved visually inspecting the provided dataset of cars and non-cars and identifying appropriate features that could help build a good classifier.

### Feature extraction
Appropriate features capturing various aspects of a car (shape, color, intensity, etc.) were extracted from the dataset.

### Training a linear SVC (Support Vector Classifier)
A linear SVC classifier was then trained on the identified features.

### Implementing a software pipeline to perform sliding window search
 The software pipeline essentially does these sets of activities on the incoming stream of images
 1. **Sliding window search**:  This involved running windows of different sizes through a valid region of the image and extracting pixels contained within these windows.
 2. **Feature extraction and prediction**: Desired features were then extracted from these pixels and fed into the trained linear classifier to get a prediction for the presence or absence of a car in the search window.
 3. **False postive removal and bounding box calculation**: A heatmap was generated for each prediction and integrated over several frames of the video. This information was then used to eliminate false positives and compute bounding boxes.

 All the above steps are illustrated with sutiable examples in the `demo_vehicle_detection.ipynb` notebook.

# Results

[![Project Video](https://img.youtube.com/vi/WdyI0JpjTDo/0.jpg)](https://youtu.be/WdyI0JpjTDo)


# Discussion

 The implemented pipeline seems to do a reasonable job on the project video. The two main cars get identified and tracked reasonably well throughout the video. And there are no false positives. The pipeline is also pretty efficient. It takes little less than 0.5 seconds to process a frame. And the whole video gets processed in less than 10 minutes (on a  windows 7 laptop with core i7 processor and 8 GB RAM).  There are still a few issues with the pipeline
 1. The pipeline fails to detect the really small cars near the top of the image and on the opposite lanes. I believe this issue can be addressed by doing a more exhaustive search through the frames using tiny windows. But this will come at a cost to performance and probably increase false positives as well.
 2. For a very brief moment in the video (between 26-28 seconds) the white car goes undetected. I am not really sure what could be causing this. I tried using smaller window sizes. I also tried training the classifier with different colorspaces and different hog features. These steps did not seem to make a difference.
 3. Finally, there are multiple factors that could potentially cause this pipeline to fail. Trucks, motorcycles, excessive lighting or lack of lighting (night time), etc. are just a few examples that could cause the pipeline to fail.

 Deep learning solutions seem to be making a lot of progress in this area. My next step would be to look at implementations for YOLO (You Look Only Once) and SSD (Single Shot MultiBox Detector).

# References

Most of the code in the Udacity lecture notes was used as a starting material for this project.
