import os
import pickle
import numpy

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from deprecated import deprecated


@deprecated(reason='Model results are loaded in model.embeddings_assembler')
def load_pickle_results(pickle_directory, normalized_faces='None'):
    try:
        if normalized_faces == 'basic_double_detection':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'data-basic-dd-norm.pickle'
        elif normalized_faces == 'basic_cut':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'data-basic-c-norm.pickle'
        elif normalized_faces == 'None':
            file_name = os.getenv('MAIN_ROUTE') + "/out/model/" + pickle_directory + 'data.pickle'
        else:
            print("[ERROR: 'normalized_faces' parameter. Please, enter one of the following values: " +
                  "'basic_double_detection', 'basic_cut' or 'None' (default)]")
            return

        file = open(file_name, 'rb')

    except OSError:
        print("[File '" + file_name + "' does not exist]")
        return None

    pickle_data = pickle.load(file)
    file.close()

    if len(pickle_data) == 0:
        return None

    return pickle_data


def get_different_video_range(embedding_matrix_1, embedding_matrix_2, distance_type='euclidean',
                              min_percentile=10, max_percentile=90):
    assert distance_type == 'euclidean' or distance_type == 'cosine', "Parameter 'distance_type' cannot be different " \
                                                                      "from 'euclidean' or 'cosine'."
    if distance_type == 'euclidean':
        distance_matrix = euclidean_distances(embedding_matrix_1, embedding_matrix_2)
    elif distance_type == 'cosine':
        distance_matrix = cosine_distances(embedding_matrix_1, embedding_matrix_2)

    return [numpy.percentile(distance_matrix, min_percentile, interpolation='nearest'),
            numpy.percentile(distance_matrix, max_percentile, interpolation='nearest'),
            distance_matrix.reshape(-1)]


@deprecated(reason='Euclidean distances are calculated using same or different videos. Use '
                   'get_different_video_range(embedding_matrix_1, embedding_matrix_2, min_percentile, max_percentile) '
                   'or get_same_video_range(embedding_matrix, min_percentile, max_percentile) instead.')
def get_same_person_range(embedding_matrix_list, min_percentile=10, max_percentile=90):
    euclidean_distance_matrix = euclidean_distances(numpy.vstack(embedding_matrix_list))
    upper_part_distance_matrix = euclidean_distance_matrix[numpy.triu_indices(euclidean_distance_matrix.shape[0], k=1)]
    return [numpy.percentile(upper_part_distance_matrix, min_percentile, interpolation='nearest'),
            numpy.percentile(upper_part_distance_matrix, max_percentile, interpolation='nearest'),
            upper_part_distance_matrix]


def get_same_video_range(embedding_matrix, distance_type='euclidean', min_percentile=10, max_percentile=90):
    assert distance_type == 'euclidean' or distance_type == 'cosine', "Parameter 'distance_type' cannot be different " \
                                                                      "from 'euclidean' or 'cosine'."
    if distance_type == 'euclidean':
        distance_matrix = euclidean_distances(embedding_matrix)
    elif distance_type == 'cosine':
        distance_matrix = cosine_distances(embedding_matrix)

    upper_part_distance_matrix = distance_matrix[numpy.triu_indices(distance_matrix.shape[0], k=1)]
    if len(upper_part_distance_matrix) > 0:
        return [numpy.percentile(upper_part_distance_matrix, min_percentile, interpolation='nearest'),
                numpy.percentile(upper_part_distance_matrix, max_percentile, interpolation='nearest'),
                upper_part_distance_matrix]
    else:
        return None, None
