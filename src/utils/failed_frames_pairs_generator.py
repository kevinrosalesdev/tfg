import numpy
import itertools

from model import facenet_keras_model


def get_failed_frames_pairs(id_video_1, id_video_2, same_person, distance_matrix, threshold, used_frames, detector):
    if detector == 'mtcnn':
        detector = 'resize'
    elif detector == 'dlib_hog':
        print("[WARNING: DLIB HOG detector can cause errors as it is not able to get the desired number of detections"
              " sometimes]")
    elif detector != 'dlib_mmod':
        print("[ERROR: 'detector' parameter. Please, enter one of the following values: "
              "'mtcnn', 'dlib_hog' or 'dlib_mmod']")
        return -1

    distance_matrix = distance_matrix.reshape((used_frames, used_frames))
    if same_person:
        row, column = numpy.where(distance_matrix > threshold)
    else:
        row, column = numpy.where(distance_matrix < threshold)

    failed_frames_pairs = list()
    face_detection_results_video_1 = facenet_keras_model.load_pickle_results(id_video_1, embedding_type=detector)
    face_detection_results_video_2 = facenet_keras_model.load_pickle_results(id_video_2, embedding_type=detector)
    for sample in zip(row, column):
        failed_frames_pairs.append((face_detection_results_video_1[sample[0]]['frame_number'],
                                    face_detection_results_video_2[sample[1]]['frame_number']))
    return failed_frames_pairs


def get_nn_failed_frames_pairs(id_video_1, id_video_2, same_person, predict_results,
                               used_frames_detector_1, detector_1,
                               used_frames_detector_2=None, detector_2=None,
                               use_one_detector=False, use_euclidean_distances=False):
    if detector_1 == 'mtcnn':
        detector_1 = 'resize'
    elif detector_1 == 'dlib_hog':
        print("[WARNING: DLIB HOG detector can cause errors as it is not able to get the desired number of detections"
              " sometimes]")
    elif detector_1 != 'dlib_mmod':
        print("[ERROR: 'detector_1' parameter. Please, enter one of the following values: "
              "'mtcnn', 'dlib_hog' or 'dlib_mmod']")
        return -1
    if not use_one_detector:
        if detector_2 == 'mtcnn':
            detector_2 = 'resize'
        elif detector_2 == 'dlib_hog':
            print("[WARNING: DLIB HOG detector can cause errors as it is not able to get the desired number of detections"
                  " sometimes]")
        elif detector_2 != 'dlib_mmod':
            print("[ERROR: 'detector_2' parameter. Please, enter one of the following values: "
                  "'mtcnn', 'dlib_hog' or 'dlib_mmod']")
            return -1

    if not use_euclidean_distances:
        if same_person:
            failed_indices = numpy.where(predict_results == 1)[0]
        else:
            failed_indices = numpy.where(predict_results == 0)[0]
    else:
        if same_person:
            failed_indices = numpy.where(predict_results >= 0.5)[0]
        else:
            failed_indices = numpy.where(predict_results < 0.5)[0]

    if not use_one_detector:
        available_indices = list(itertools.product(list(range(used_frames_detector_1)), list(range(used_frames_detector_2))))
        embeddings_number = 4
    else:
        available_indices = list(range(used_frames_detector_1))
        embeddings_number = 2

    available_indices = numpy.array(list(itertools.product(available_indices, available_indices)))[failed_indices]
    available_indices = available_indices.reshape(available_indices.shape[0], embeddings_number)

    failed_frames_pairs = list()
    face_detection_results_video_1_detector_1 = facenet_keras_model.load_pickle_results(id_video_1,
                                                                                        embedding_type=detector_1)
    face_detection_results_video_2_detector_1 = facenet_keras_model.load_pickle_results(id_video_2,
                                                                                        embedding_type=detector_1)

    if not use_one_detector:
        face_detection_results_video_1_detector_2 = facenet_keras_model.load_pickle_results(id_video_1,
                                                                                            embedding_type=detector_2)
        face_detection_results_video_2_detector_2 = facenet_keras_model.load_pickle_results(id_video_2,
                                                                                            embedding_type=detector_2)
    for sample in available_indices:
        if not use_one_detector:
            failed_frames_pairs.append((face_detection_results_video_1_detector_1[sample[0]]['frame_number'],
                                        face_detection_results_video_1_detector_2[sample[1]]['frame_number'],
                                        face_detection_results_video_2_detector_1[sample[2]]['frame_number'],
                                        face_detection_results_video_2_detector_2[sample[3]]['frame_number']))
        else:
            failed_frames_pairs.append((face_detection_results_video_1_detector_1[sample[0]]['frame_number'],
                                        face_detection_results_video_2_detector_1[sample[1]]['frame_number']))
    return failed_frames_pairs

