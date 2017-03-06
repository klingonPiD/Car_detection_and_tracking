
import ProcessFrame
import pickle
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure

# Create a ProcessFrame object
process_frame = ProcessFrame.ProcessFrame()
params = process_frame.setup_params()
svm_dict = pickle.load(open("svm_pickle.p", "rb"))
clf = svm_dict["clf"]
scaler = svm_dict["scaler"]
# x_start_stop=[None, None]
# y_start_stop=[400, 650]
# slide_window_tup1 = (32, (10, 6), (400, 470))
slide_window_tup1 = (64, (5, 4), (400, 550))
slide_window_tup2 = (96, (5, 5), (400, 650))
slide_window_params = [slide_window_tup1, slide_window_tup2]

def vehicle_detect_pipeline(image):
    """Vehicle detection pipeline"""
    process_frame.find_cars(image, slide_window_params, svc=clf, X_scaler=scaler, params=params)
    process_frame.update_thresh_box()
    sel = generate_binary_structure(2, 2)
    labels = label(process_frame.thresh_box_dict['heat_map'], sel)
    detect_car_img = process_frame.draw_labeled_bboxes(image, labels)
    return detect_car_img


def process_image(frame):
    """Method that is invoked by movie py for every frame in a video"""
    result = vehicle_detect_pipeline(frame)# first_frame_flag)
    return result

from moviepy.editor import VideoFileClip
myclip = VideoFileClip('project_video.mp4')
mod_clip = myclip.fl_image(process_image)
mod_clip.write_videofile('project_video_output.mp4', audio=False)





