import pickle
import numpy
import os
import datetime
import re
import pandas as pd


def load_panda_embeddings(pickle_file_name):
    try:
        file_name = os.getenv('MAIN_ROUTE') + "/out/embeddings-assembler/" + pickle_file_name
        file = open(file_name, 'rb')

    except OSError:
        print("[File '" + file_name + "' does not exist]")
        return None

    pickle_data = pickle.load(file)
    file.close()

    if len(pickle_data) == 0:
        return None

    return pickle_data


def load_pickle_results(pickle_directory, embedding_type='None'):
    try:
        if embedding_type == 'basic_double_detection':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'data-basic-dd-norm.pickle'
        elif embedding_type == 'basic_cut':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'data-basic-c-norm.pickle'
        elif embedding_type == 'resize':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'data-resize.pickle'
        elif embedding_type == 'dlib_hog':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'dlib-hog-data.pickle'
        elif embedding_type == 'dlib_mmod':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'dlib-mmod-data.pickle'
        elif embedding_type == 'None':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'data.pickle'
        else:
            print("[ERROR: 'normalized_faces' parameter. Please, enter one of the following values: " +
                  "'basic_double_detection', 'basic_cut', 'resize', 'dlib_hog', 'dlib_mmod' or 'None' (default)]")
            return

        file = open(file_name, 'rb')

    except OSError:
        print("[File '" + file_name + "' does not exist]")
        return None

    pickle_data = pickle.load(file)
    file.close()

    if len(pickle_data) == 0:
        return None

    pickle_metadata = re.findall('\d+', pickle_directory)

    floor = numpy.ones((len(pickle_data), 1))
    floor.fill(pickle_metadata[0])

    person_id = numpy.ones((len(pickle_data), 1))
    person_id.fill(pickle_metadata[1])

    return [person_id, floor, pickle_data]


def save_pickle_results(dataframe, name, embedding_type='None'):
    if name is None:
        name = ''
    else:
        name += '-'

    result_directory = os.getenv('MAIN_ROUTE') + "/out/embeddings-assembler/" + \
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f-") + name

    try:
        if embedding_type == 'basic_double_detection':
            pickle_file = open(result_directory + 'data-basic-dd-norm.pickle', 'wb+')
        elif embedding_type == 'basic_cut':
            pickle_file = open(result_directory + 'data-basic-c-norm.pickle', 'wb+')
        elif embedding_type == 'resize':
            pickle_file = open(result_directory + 'data-resize.pickle', 'wb+')
        elif embedding_type == 'dlib_hog':
            pickle_file = open(result_directory + 'dlib-hog-data.pickle', 'wb+')
        elif embedding_type == 'dlib_mmod':
            pickle_file = open(result_directory + 'dlib-mmod-data.pickle', 'wb+')
        elif embedding_type == 'None':
            pickle_file = open(result_directory + 'data.pickle', 'wb+')
        else:
            print("[ERROR: 'normalized_faces' parameter. Please, enter one of the following values: " +
                  "'basic_double_detection', 'basic_cut', 'resize', 'dlib_hog', 'dlib_mmod' or 'None' (default)]")
            return

        pickle.dump(dataframe, pickle_file)
        pickle_file.close()

    except OSError:
        if embedding_type == 'basic_double_detection':
            print("[File '" + result_directory + "data-basic-dd-norm.pickle' could not be created]")
        elif embedding_type == 'basic_cut':
            print("[File '" + result_directory + "data-basic-c-norm.pickle' could not be created]")
        elif embedding_type == 'resize':
            print("[File '" + result_directory + "data-resize.pickle' could not be created]")
        elif embedding_type == 'dlib_hog':
            print("[File '" + result_directory + "dlib-hog-data.pickle' could not be created]")
        elif embedding_type == 'dlib_mmod':
            print("[File '" + result_directory + "dlib-mmod-data.pickle' could not be created]")
        elif embedding_type == 'None':
            print("[File '" + result_directory + "data.pickle' could not be created]")


def save_csv_results(dataframe, name, embedding_type='None'):
    if name is None:
        name = ''
    else:
        name += '-'

    result_directory = os.getenv('MAIN_ROUTE') + "/out/embeddings-assembler/" + \
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f-") + name

    if embedding_type == 'basic_double_detection':
        dataframe.to_csv(result_directory + 'data-basic-dd-norm.csv', index=False, header=True)
    elif embedding_type == 'basic_cut':
        dataframe.to_csv(result_directory + 'data-basic-c-norm.csv', index=False, header=True)
    elif embedding_type == 'resize':
        dataframe.to_csv(result_directory + 'data-resize.csv', index=False, header=True)
    elif embedding_type == 'dlib_hog':
        dataframe.to_csv(result_directory + 'dlib-hog-data.csv', index=False, header=True)
    elif embedding_type == 'dlib_mmod':
        dataframe.to_csv(result_directory + 'dlib-mmod-data.csv', index=False, header=True)
    elif embedding_type == 'None':
        dataframe.to_csv(result_directory + 'data.csv', index=False, header=True)
    else:
        print("[ERROR: 'normalized_faces' parameter. Please, enter one of the following values: " +
              "'basic_double_detection', 'basic_cut', 'resize', 'dlib_hog', 'dlib_mmod' or 'None' (default)]")


def assemble_embeddings(frames_folders_name_list, embedding_type='None', pickle_results=True, csv_results=True,
                        name=None):
    [person_id, floor, first_video_embedding] = load_pickle_results(frames_folders_name_list[0], embedding_type)
    videos_ids_list = [frames_folders_name_list[0]] * person_id.shape[0]
    face_embeddings = numpy.hstack((person_id, floor, first_video_embedding))
    for frames_folder_name in frames_folders_name_list[1:]:
        [person_id, floor, video_embedding] = load_pickle_results(frames_folder_name, embedding_type)
        videos_ids_list.extend([frames_folder_name] * person_id.shape[0])
        face_embeddings = numpy.vstack((face_embeddings, numpy.hstack((person_id, floor, video_embedding))))

    face_embeddings_df = create_pandas_df(face_embeddings, videos_ids_list)

    if pickle_results:
        save_pickle_results(face_embeddings_df, name, embedding_type)

    if csv_results:
        save_csv_results(face_embeddings_df, name, embedding_type)

    print("[Embeddings Pandas DataFrame has been stored in '" + os.getenv('MAIN_ROUTE')
          + "/out/embeddings-assembler/']")

    return face_embeddings_df


def create_pandas_df(face_embeddings, videos_ids_list):
    header = ['id', 'floor']
    header.extend(list(range(0, face_embeddings.shape[1]-len(header))))
    df = pd.DataFrame(face_embeddings, columns=header)

    df.insert(0, "video_id", videos_ids_list)

    return df
