import os
import numpy
import mtcnn
import pickle
import cv2
import random
import imquality.brisque as brisque

from termcolor import cprint
from matplotlib import pyplot as plt
from PIL import Image
from facedetector import FaceNormalizationUtils as faceutils


def get_false_negative_rate(false_negative_samples, true_positive_samples):
    cprint("False negative rate: " + str(false_negative_samples/(false_negative_samples + true_positive_samples)),
           'blue')


def get_false_positive_rate(false_positive_samples, true_negative_samples):
    cprint("False positive rate: " + str(false_positive_samples/(false_positive_samples + true_negative_samples)),
           'blue')


class MTCNNDetector:

    def __init__(self, max_detections=10, pickle_results=True, image_results=False, force_update=False,
                 normalize_face='None', confidence_range=0.00001, frames_directory='frames-generator/'):
        self.max_detections = max_detections
        self.pickle_results = pickle_results
        self.image_results = image_results
        self.force_update = force_update
        self.normalize_face = normalize_face
        self.confidence_range = confidence_range
        self.frames_directory = frames_directory
        self.detector = mtcnn.MTCNN()
        if self.normalize_face == 'basic_double_detection' or self.normalize_face == 'basic_cut':
            self.normalizatorHS = faceutils.Normalization()

    def process_video(self, frames_folder_name):
        directory = os.getenv('MAIN_ROUTE') + "/out/" + self.frames_directory + frames_folder_name
        result_directory = os.getenv('MAIN_ROUTE') + '/out/face-detector/' + frames_folder_name
        print("[Detecting faces from '" + directory + "']")
        try:
            if not os.path.exists(result_directory):
                os.makedirs(result_directory)
            else:
                if self.force_update:
                    print("[WARNING: results already exists. Existing results will be updated]")
                else:
                    if (self.normalize_face == 'basic_double_detection' and os.path.exists(
                            result_directory + 'data-basic-dd-norm.pickle')) or \
                            (self.normalize_face == 'basic_cut' and os.path.exists(
                                result_directory + 'data-basic-c-norm.pickle')) or \
                            (self.normalize_face == 'resize' and os.path.exists(
                                result_directory + 'data-resize.pickle')) or \
                            (self.normalize_face == 'None' and os.path.exists(result_directory + 'data.pickle')):
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

                if detection is not None:
                    detection['frame_number'] = frame
                    face_matrix = self.prepare_face(detection, directory)
                    gray_scale_image = cv2.cvtColor(face_matrix, cv2.COLOR_BGR2GRAY)
                    pil_gray_scale_image = Image.fromarray(gray_scale_image)
                    edges_image = cv2.Canny(gray_scale_image, 100, 200)
                    detection['confidence'] = 0.0125 * (10 * (numpy.sum(edges_image / 255) /
                                                              (edges_image.shape[0] * edges_image.shape[1]))) \
                                              + 0.975 * detection['confidence'] \
                                              + 0.0125 * (1 / (1 + brisque.score(pil_gray_scale_image)))

                    if detection['confidence'] > current_max_confidence:
                        current_max_confidence = detection['confidence']

                    detection_list.append(detection)

        detection_list = self.prepare_results_list(sorted(detection_list, key=lambda k: k['confidence'], reverse=True),
                                                   current_max_confidence)

        if self.normalize_face == 'basic_double_detection' or self.normalize_face == 'basic_cut':

            normalized_detection_list = list()

            for index, detection in enumerate(detection_list):

                print("[Normalizing " + str(index) + "/" + str(len(detection_list)) + " detection]", end='\r')

                img = numpy.asarray(Image.open(directory + detection['frame_number']))
                b, g, r = cv2.split(img)

                detection_left_eye = detection['keypoints']['left_eye']
                detection_right_eye = detection['keypoints']['right_eye']
                self.normalizatorHS.normalize_gray_img(b, detection_left_eye[0], detection_left_eye[1],
                                                       detection_right_eye[0], detection_right_eye[1],
                                                       faceutils.Kind_wraping.HS)
                b_norm = self.normalizatorHS.normf_image
                self.normalizatorHS.normalize_gray_img(g, detection_left_eye[0], detection_left_eye[1],
                                                       detection_right_eye[0], detection_right_eye[1],
                                                       faceutils.Kind_wraping.HS)
                g_norm = self.normalizatorHS.normf_image
                self.normalizatorHS.normalize_gray_img(r, detection_left_eye[0], detection_left_eye[1],
                                                       detection_right_eye[0], detection_right_eye[1],
                                                       faceutils.Kind_wraping.HS)
                r_norm = self.normalizatorHS.normf_image

                norm_bgr = cv2.merge((b_norm, g_norm, r_norm))

                Image.fromarray(norm_bgr).save(directory + "n-" + str(detection['frame_number']))

                if self.normalize_face == 'basic_double_detection':
                    new_detection = self.detect_face(norm_bgr)

                    if new_detection is not None:
                        new_detection['frame_number'] = "n-" + detection['frame_number']
                        normalized_detection_list.append(new_detection)

                elif self.normalize_face == 'basic_cut':
                    detection['box'] = [35, 39, 80, 80]
                    detection['frame_number'] = "n-" + detection['frame_number']
                    normalized_detection_list.append(detection)

            detection_list = normalized_detection_list

        elif self.normalize_face == 'resize':

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

        elif self.normalize_face != 'None':
            print("[ERROR: 'normalize_face' parameter. Please, enter one of the following values: "
                  "'basic_double_detection', 'basic_cut', 'resize', or 'None' (default)]")
            return None

        if self.image_results:
            for detection in detection_list:
                face_matrix = self.prepare_face(detection, directory)
                plt.imshow(face_matrix)
                plt.savefig(result_directory + detection['frame_number'])

        if self.pickle_results:
            self.save_pickle_results(detection_list, result_directory)

        return detection_list

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
        detection = self.detector.detect_faces(image)

        if detection:
            return detection[0]

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
            if self.normalize_face == 'basic_double_detection':
                pickle_file = open(pickle_directory + 'data-basic-dd-norm.pickle', 'wb+')
            elif self.normalize_face == 'basic_cut':
                pickle_file = open(pickle_directory + 'data-basic-c-norm.pickle', 'wb+')
            elif self.normalize_face == 'resize':
                pickle_file = open(pickle_directory + 'data-resize.pickle', 'wb+')
            elif self.normalize_face == 'None':
                pickle_file = open(pickle_directory + 'data.pickle', 'wb+')
            else:
                print("[ERROR: 'normalize_face' parameter. Please, enter one of the following values: " +
                      "'basic_double_detection', 'basic_cut', 'resize' or 'None' (default)]")
                return

            pickle.dump(pickle_data, pickle_file)
            pickle_file.close()
            print("[Faces have been stored in '" + pickle_directory + "']")

        except OSError:
            if self.normalize_face == 'basic_double_detection':
                print("[File '" + pickle_directory + "data-basic-dd-norm.pickle' could not be created]")
            elif self.normalize_face == 'basic_cut':
                print("[File '" + pickle_directory + "data-basic-c-norm,pickle' could not be created]")
            elif self.normalize_face == 'resize':
                print("[File '" + pickle_directory + "data-resize.pickle' could not be created]")
            elif self.normalize_face == 'None':
                print("[File '" + pickle_directory + "data.pickle' could not be created]")
            else:
                print("[ERROR: 'normalize_face' parameter. Please, enter one of the following values: " +
                      "'basic_double_detection', 'basic_cut', 'resize' or 'None' (default)]")
