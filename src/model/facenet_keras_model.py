import os
import pickle
import numpy

from keras.models import load_model
from PIL import Image
from sklearn.preprocessing import normalize


def load_pickle_results(pickle_directory, embedding_type='None'):
    try:
        if embedding_type == 'basic_double_detection':
            file_name = os.getenv('MAIN_ROUTE') + "/out/face-detector/" + pickle_directory + 'data-basic-dd-norm.pickle'
        elif embedding_type == 'basic_cut':
            file_name = os.getenv('MAIN_ROUTE') + "/out/face-detector/" + pickle_directory + 'data-basic-c-norm.pickle'
        elif embedding_type == 'resize':
            file_name = os.getenv('MAIN_ROUTE') + "/out/face-detector/" + pickle_directory + 'data-resize.pickle'
        elif embedding_type == 'dlib_hog':
            file_name = os.getenv('MAIN_ROUTE') + "/out/face-detector/" + pickle_directory + 'dlib-hog-data.pickle'
        elif embedding_type == 'dlib_mmod':
            file_name = os.getenv('MAIN_ROUTE') + "/out/face-detector/" + pickle_directory + 'dlib-mmod-data.pickle'
        elif embedding_type == 'None':
            file_name = os.getenv('MAIN_ROUTE') + "/out/face-detector/" + pickle_directory + 'data.pickle'
        else:
            print("[ERROR: 'embedding_type' parameter. Please, enter one of the following values: " +
                  "'basic_double_detection', 'basic_cut', 'resize', 'dlib_hog', 'dlib_mmod' or 'None' (default)]")
            return

        file = open(file_name, 'rb')

    except OSError:
        print("[File '" + file_name + "' does not exist]")
        return None

    pickle_data = pickle.load(file)
    file.close()

    return pickle_data


def save_pickle_results(pickle_data, pickle_directory, embedding_type='None'):
    try:
        if embedding_type == 'basic_double_detection':
            pickle_file = open(pickle_directory + 'data-basic-dd-norm.pickle', 'wb+')
        elif embedding_type == 'basic_cut':
            pickle_file = open(pickle_directory + 'data-basic-c-norm.pickle', 'wb+')
        elif embedding_type == 'resize':
            pickle_file = open(pickle_directory + 'data-resize.pickle', 'wb+')
        elif embedding_type == 'dlib_hog':
            pickle_file = open(pickle_directory + 'dlib-hog-data.pickle', 'wb+')
        elif embedding_type == 'dlib_mmod':
            pickle_file = open(pickle_directory + 'dlib-mmod-data.pickle', 'wb+')
        elif embedding_type == 'None':
            pickle_file = open(pickle_directory + 'data.pickle', 'wb+')
        else:
            print("[ERROR: 'embedding_type' parameter. Please, enter one of the following values: " +
                  "'basic_double_detection', 'basic_cut', 'resize', 'dlib_hog', 'dlib_mmod' or 'None' (default)]")
            return

        pickle.dump(pickle_data, pickle_file)
        pickle_file.close()

    except OSError:
        if embedding_type == 'basic_double_detection':
            print("[File '" + pickle_directory + "data-basic-dd-norm.pickle' could not be created]")
        elif embedding_type == 'basic_cut':
            print("[File '" + pickle_directory + "data-basic-c-norm.pickle' could not be created]")
        elif embedding_type == 'resize':
            print("[File '" + pickle_directory + "data-resize.pickle' could not be created]")
        elif embedding_type == 'dlib_hog':
            print("[File '" + pickle_directory + "dlib-hog-data.pickle' could not be created]")
        elif embedding_type == 'dlib_mmod':
            print("[File '" + pickle_directory + "dlib-mmod-data.pickle' could not be created]")
        elif embedding_type == 'None':
            print("[File '" + pickle_directory + "data.pickle' could not be created]")


def save_csv_results(pickle_data, pickle_directory, embedding_type='None'):
    if embedding_type == 'basic_double_detection':
        numpy.savetxt(pickle_directory + 'data-basic-dd-norm.csv', pickle_data, delimiter=',')
    elif embedding_type == 'basic_cut':
        numpy.savetxt(pickle_directory + 'data-basic-c-norm.csv', pickle_data, delimiter=',')
    elif embedding_type == 'resize':
        numpy.savetxt(pickle_directory + 'data-resize.csv', pickle_data, delimiter=",")
    elif embedding_type == 'dlib_hog':
        numpy.savetxt(pickle_directory + 'dlib-hog-data.csv', pickle_data, delimiter=",")
    elif embedding_type == 'dlib_mmod':
        numpy.savetxt(pickle_directory + 'dlib-mmod-data.csv', pickle_data, delimiter=",")
    elif embedding_type == 'None':
        numpy.savetxt(pickle_directory + 'data.csv', pickle_data, delimiter=',')
    else:
        print("[ERROR: 'embedding_type' parameter. Please, enter one of the following values: " +
              "'basic_double_detection', 'basic_cut', 'resize', 'dlib_hog', 'dlib_mmod' or 'None' (default)]")


class FacenetKerasModel:

    def __init__(self, norm_type='l2', embedding_type='None', pickle_results=True, csv_results=False,
                 print_model_info=False, force_update=False):
        self.pickle_results = pickle_results
        self.csv_results = csv_results
        self.print_model_info = print_model_info
        self.force_update = force_update
        self.norm_type = norm_type
        self.embedding_type = embedding_type
        self.model = self.get_model()

    def get_model(self):
        model = load_model('src/model/facenet_keras.h5', compile=False)

        if self.print_model_info:
            model.summary()
            print(model.inputs)
            print(model.outputs)

        return model

    def predict_data(self, frames_folder_name, face_detection_results):
        directory = os.getenv('MAIN_ROUTE') + "/out/frames-generator/" + frames_folder_name
        result_directory = os.getenv('MAIN_ROUTE') + "/out/model/" + frames_folder_name
        print("[Getting Face Embeddings from video '" + frames_folder_name + "']")

        try:
            if not os.path.exists(result_directory):
                os.makedirs(result_directory)
            else:
                if self.force_update:
                    print("[WARNING: results already exists. Existing results will be updated]")
                else:
                    if (self.embedding_type == 'basic_double_detection' and os.path.exists(
                            result_directory + 'data-basic-dd-norm.pickle')) or \
                            (self.embedding_type == 'basic_cut' and os.path.exists(
                                result_directory + 'data-basic-c-norm.pickle')) or \
                            (self.embedding_type == 'resize' and os.path.exists(
                                result_directory + 'data-resize.pickle')) or \
                            (self.embedding_type == 'dlib_hog' and os.path.exists(
                                result_directory + 'dlib-hog-data.pickle')) or \
                            (self.embedding_type == 'dlib_mmod' and os.path.exists(
                                result_directory + 'dlib-mmod-data.pickle')) or \
                            (self.embedding_type == 'None' and os.path.exists(result_directory + 'data.pickle')):
                        print("[Results already exists. Existing results will not be updated]")
                        return 0

        except OSError:
            print('[Error creating detection results folder]')
            return -1

        vectorized_faces = []

        if face_detection_results:
            prepared_faces_matrix = self.prepare_face(face_detection_results[0], directory)
            for face in face_detection_results[1:]:
                prepared_faces_matrix = numpy.vstack((prepared_faces_matrix, self.prepare_face(face, directory)))

            vectorized_faces = self.model.predict(prepared_faces_matrix)
            if self.norm_type == 'l2':
                vectorized_faces = normalize(vectorized_faces, norm='l2')
            elif self.norm_type == 'l1':
                vectorized_faces = normalize(vectorized_faces, norm='l1')
            elif self.norm_type != 'None':
                print("[ERROR: 'norm type' parameter. " +
                      "Please, enter one of the following values: 'l2', 'l1' or 'None']")
                return None

        if self.pickle_results:
            save_pickle_results(vectorized_faces, result_directory, self.embedding_type)

        if self.csv_results:
            save_csv_results(vectorized_faces, result_directory, self.embedding_type)

        print("[Face embeddings have been stored in '" + result_directory + "']")
        return vectorized_faces

    @staticmethod
    def prepare_face(face, directory):
        pixels_array = numpy.asarray(Image.open(directory + face['frame_number']))
        x1, y1, width, height = face['box']
        face_matrix = pixels_array[abs(y1):abs(y1) + height, abs(x1):abs(x1) + width]
        return numpy.divide(numpy.asarray(Image.fromarray(face_matrix).resize((160, 160))), 255).reshape(
            (1, 160, 160, 3))
