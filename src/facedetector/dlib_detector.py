import os
import numpy
import dlib
import pickle
import random
import cv2
import imquality.brisque as brisque

from termcolor import cprint
from matplotlib import pyplot as plt
from PIL import Image


def get_false_negative_rate(false_negative_samples, true_positive_samples):
    cprint("False negative rate: " + str(false_negative_samples/(false_negative_samples + true_positive_samples)),
           'blue')


def get_false_positive_rate(false_positive_samples, true_negative_samples):
    cprint("False positive rate: " + str(false_positive_samples/(false_positive_samples + true_negative_samples)),
           'blue')


class DLIBDetector:

    def __init__(self, max_detections=10, pickle_results=True, image_results=False, force_update=False,
                 detector_type='mmod', confidence_range=0.02, frames_directory='/out/frames-generator/'):
        self.max_detections = max_detections
        self.pickle_results = pickle_results
        self.image_results = image_results
        self.force_update = force_update
        self.detector_type = detector_type
        self.confidence_range = confidence_range
        self.frames_directory = frames_directory
        self.predictor68 = dlib.shape_predictor("src/facedetector/shape_predictor_68_face_landmarks.dat")
        if detector_type == 'hog':
            self.detector = dlib.get_frontal_face_detector()
        elif detector_type == 'mmod':
            self.detector = dlib.cnn_face_detection_model_v1('src/facedetector/mmod_human_face_detector.dat')
        else:
            print("[WARNING: 'detector_type' parameter. Please, enter one of the following values: "
                  "'hog'(not recommended) or 'mmod']")

    def process_video(self, frames_folder_name):
        directory = os.getenv('MAIN_ROUTE') + self.frames_directory + frames_folder_name
        result_directory = os.getenv('MAIN_ROUTE') + '/out/face-detector/' + frames_folder_name
        print("[Detecting faces from '" + directory + "']")
        try:
            if not os.path.exists(result_directory):
                os.makedirs(result_directory)
            else:
                if self.force_update:
                    print("[WARNING: results already exists. Existing results will be updated]")
                else:
                    if (self.detector_type == 'hog' and
                        os.path.exists(result_directory + 'dlib-hog-data.pickle')) or \
                            (self.detector_type == 'mmod' and
                             os.path.exists(result_directory + 'dlib-mmod-data.pickle')):
                        print("[Results already exists. Existing results will not be updated]")
                        return 0
        except OSError:
            print('[Error creating detection results folder]')
            return -1

        video_frames = os.listdir(directory)
        detection_list = list()
        current_max_confidence = 1

        for index, frame in enumerate(video_frames):
            print("[Processing " + str(index) + "/" + str(len(video_frames)) + " frame]", end='\r')
            if frame.startswith("frame-") and frame.endswith(".jpg"):
                pixels_array = numpy.asarray(Image.open(directory + frame))
                detection = self.detect_face(pixels_array)
                if detection == -1:
                    return

                if detection is not None:
                    detection_box = detection[0]
                    detection = {'frame_number': frame,
                                 'box': [detection_box.tl_corner().x, detection_box.tl_corner().y,
                                         detection_box.width(), detection_box.height()],
                                 'confidence': detection[1]}

                    points = self.get_facial_landmarks(pixels_array, detection_box)

                    if points is not None:
                        face_matrix = self.prepare_face(detection, directory)
                        detection['keypoints'] = {'nose': points[2],
                                                  'mouth_right': points[1],
                                                  'right_eye': points[4],
                                                  'left_eye': points[3],
                                                  'mouth_left': points[0]}

                        gray_scale_image = cv2.cvtColor(face_matrix, cv2.COLOR_BGR2GRAY)
                        pil_gray_scale_image = Image.fromarray(gray_scale_image)
                        edges_image = cv2.Canny(gray_scale_image, 100, 200)
                        if self.detector_type == 'mmod':
                            detection['confidence'] = 0.15 * (10 * (numpy.sum(edges_image / 255) /
                                                                    (edges_image.shape[0] * edges_image.shape[1]))) \
                                                      + 0.7 * detection['confidence'] \
                                                      + 0.15 * (1 / (1 + brisque.score(pil_gray_scale_image)))
                        elif self.detector_type == 'hog':
                            detection['confidence'] = 0.25 * (10 * (numpy.sum(edges_image / 255) /
                                                                    (edges_image.shape[0] * edges_image.shape[1]))) \
                                                      + 0.5 * detection['confidence'] \
                                                      + 0.25 * (1 / (1 + brisque.score(pil_gray_scale_image)))
                        else:
                            print("[Error: 'detector_type' constructor parameter. Please, enter one of the following "
                                  "values: 'hog' or 'mmod'")
                            return

                        if detection['confidence'] > current_max_confidence:
                            current_max_confidence = detection['confidence']

                        detection_list.append(detection)
                    else:
                        continue

        detection_list = self.prepare_results_list(sorted(detection_list, key=lambda k: k['confidence'], reverse=True),
                                                   current_max_confidence)

        resized_detection_list = list()

        for index, detection in enumerate(detection_list):
            print("[Resizing " + str(index) + "/" + str(len(detection_list)) + " detection]", end='\r')
            eyes_x_axis_difference = detection['keypoints']['right_eye'][0] - detection['keypoints']['left_eye'][0]

            mouth_eye_y_axis_difference = ((detection['keypoints']['mouth_left'][1] +
                                            detection['keypoints']['mouth_right'][1]) / 2) - \
                                          ((detection['keypoints']['right_eye'][1] +
                                            detection['keypoints']['left_eye'][1]) / 2)

            detection['box'][0] = round(detection['keypoints']['left_eye'][0] - eyes_x_axis_difference)

            detection['box'][1] = round(((detection['keypoints']['right_eye'][1] +
                                          detection['keypoints']['left_eye'][1]) / 2) - mouth_eye_y_axis_difference)

            detection['box'][2] = round(detection['keypoints']['right_eye'][0] \
                                        + eyes_x_axis_difference - detection['box'][0])

            detection['box'][3] = round(((detection['keypoints']['mouth_left'][1] +
                                          detection['keypoints']['mouth_right'][1]) / 2) \
                                        + mouth_eye_y_axis_difference - detection['box'][1])

            resized_detection_list.append(detection)

        detection_list = resized_detection_list

        if self.image_results:
            for detection in detection_list:
                face_matrix = self.prepare_face(detection, directory)
                plt.imshow(face_matrix)
                plt.savefig(result_directory + detection['frame_number'])

        if self.pickle_results:
            self.save_pickle_results(detection_list, result_directory)

        return detection_list

    def get_facial_landmarks(self, pixels_array, face):
        points = self.predictor68(pixels_array, face)

        if points is not None:
            return [(points.part(49).x, points.part(49).y),
                    (points.part(55).x, points.part(55).y),
                    (points.part(34).x, points.part(34).y),
                    (points.part(38).x, points.part(38).y),
                    (points.part(45).x, points.part(45).y)]

        return None

    def prepare_results_list(self, detection_list, current_max_confidence):
        new_detection_list = list()
        not_chosen_detections = list()

        confidence = current_max_confidence

        while len(new_detection_list) < self.max_detections and confidence > 0:
            detection_set = [detection for detection in detection_list
                             if confidence - self.confidence_range <= detection['confidence'] <= confidence]

            if detection_set:
                new_detection_list.append(detection_set.pop(random.randrange(len(detection_set))))
                not_chosen_detections.extend(detection_set)

            confidence -= self.confidence_range

        if len(new_detection_list) < self.max_detections:
            new_detection_list.extend(not_chosen_detections[:self.max_detections - len(new_detection_list)])

        return new_detection_list

    def detect_face(self, image):
        if self.detector_type == 'hog':
            detection, scores = self.detector.run(image, 0, 0)[:2]
        elif self.detector_type == 'mmod':
            detection = self.detector(image, 0)
            if detection:
                scores = [detection[0].confidence]
                detection = [detection[0].rect]
        else:
            print("[ERROR: 'detector_type' parameter. Please, enter one of the following values: "
                  "'hog'(not recommended) or 'mmod']")
            return -1

        if detection and scores:
            return [detection[0], scores[0]]

        return None

    @staticmethod
    def prepare_face(face, directory):
        pixels_array = numpy.asarray(Image.open(directory + face['frame_number']))
        x1, y1, width, height = face['box']
        face_matrix = pixels_array[abs(y1):abs(y1) + height, abs(x1):abs(x1) + width]
        return numpy.asarray(Image.fromarray(face_matrix).resize((160, 160)))

    def save_pickle_results(self, pickle_data, pickle_directory):
        if not pickle_data:
            print("[ERROR: detection list is empty. Pickle file could not be generated]")
            return
        try:
            if self.detector_type == 'hog':
                pickle_file = open(pickle_directory + 'dlib-hog-data.pickle', 'wb+')
            elif self.detector_type == 'mmod':
                pickle_file = open(pickle_directory + 'dlib-mmod-data.pickle', 'wb+')
            else:
                print("[ERROR: 'detector_type' parameter. Please, enter one of the following values: "
                      "'hog'(not recommended) or 'mmod']")
                return
            pickle.dump(pickle_data, pickle_file)
            pickle_file.close()
            print("[Faces have been stored in '" + pickle_directory + "']")

        except OSError:
            if self.detector_type == 'hog':
                print("[File '" + pickle_directory + "dlib-hog-data.pickle' could not be created]")
            elif self.detector_type == 'mmod':
                print("[File '" + pickle_directory + "dlib-mmod-data.pickle' could not be created]")
            else:
                print("[ERROR: 'detector_type' parameter. Please, enter one of the following values: "
                      "'hog'(not recommended) or 'mmod']")
