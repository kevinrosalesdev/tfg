import numpy
from termcolor import cprint
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def get_false_acceptance_rate(false_acceptances_number, total_samples):
    return false_acceptances_number / total_samples


def get_false_rejection_rate(false_rejections_number, total_samples):
    return false_rejections_number / total_samples


def print_errors(false_rejection_rate, false_acceptance_rate):
    cprint("FAR: ", 'cyan', None, end='')
    cprint(round(false_acceptance_rate, 3), 'yellow')
    cprint("FRR: ", 'cyan', None, end='')
    cprint(round(false_rejection_rate, 3), 'yellow')


def measure_threshold(threshold, original_embedding_matrix, comparing_embedding_matrix, total_columns_with_same_person,
                      distance_type='euclidean', print_results=False):
    assert (total_columns_with_same_person < comparing_embedding_matrix.shape[0]), \
        "The number of columns with the same person cannot be >= " + str(comparing_embedding_matrix.shape[0])

    assert distance_type == 'euclidean' or distance_type == 'cosine', "Parameter 'distance_type' cannot be different " \
                                                                      "from 'euclidean' or 'cosine'."

    if distance_type == 'euclidean':
        distance_matrix = euclidean_distances(original_embedding_matrix, comparing_embedding_matrix)
    elif distance_type == 'cosine':
        distance_matrix = cosine_distances(original_embedding_matrix, comparing_embedding_matrix)

    fa_number = numpy.where(distance_matrix[:, total_columns_with_same_person:] < threshold)[0].shape[0]
    fr_number = numpy.where(distance_matrix[:, :total_columns_with_same_person] > threshold)[0].shape[0]

    genuine_samples = distance_matrix.shape[0] * total_columns_with_same_person
    impostor_samples = distance_matrix.shape[0] * \
                       (distance_matrix.shape[1] - total_columns_with_same_person)

    false_acceptance_rate = get_false_acceptance_rate(fa_number, impostor_samples)
    false_rejection_rate = get_false_rejection_rate(fr_number, genuine_samples)

    if print_results:
        print_errors(false_rejection_rate, false_acceptance_rate)
        print("Frames number with same person:", total_columns_with_same_person)
        print("Frames number with different person:",
              distance_matrix.shape[1] - total_columns_with_same_person)

    return [false_acceptance_rate, false_rejection_rate, genuine_samples,
            impostor_samples, fa_number, fr_number]
