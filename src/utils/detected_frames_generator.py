import os
import shutil

from model import facenet_keras_model


def generate_detected_frames(frames_directory, detector, frames_folders_names):
    if detector == 'mtcnn':
        detector = 'resize'
    elif detector != 'dlib_hog' and detector != 'dlib_mmod':
        print("[ERROR: 'detector' parameter. Please, enter one of the following values: "
              "'mtcnn', 'dlib_hog' or 'dlib_mmod']")
        return -1

    directory = os.getenv('MAIN_ROUTE') + "/out/frames-generator/"
    copy_directory = os.getenv('MAIN_ROUTE') + "/out/" + frames_directory
    if not os.path.exists(copy_directory):
        os.makedirs(copy_directory)
    for frame_folder_name in frames_folders_names:
        original_frames_directory = directory + frame_folder_name
        detected_frames_directory = copy_directory + frame_folder_name
        if not os.path.exists(detected_frames_directory):
            os.makedirs(detected_frames_directory)
        face_detection_results = facenet_keras_model.load_pickle_results(frame_folder_name, embedding_type=detector)
        for face in face_detection_results:
            shutil.copy(original_frames_directory + face['frame_number'],
                        detected_frames_directory + face['frame_number'])

    print("[Detected frames have been stored in '" + copy_directory + "']")
    return 0
