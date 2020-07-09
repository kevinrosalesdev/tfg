import itertools

from verifier.verifier import Verifier


def get_same_person_videos_ids_pairs(videos_embeddings):
    id_pairs = set()

    for possible_combination in itertools.combinations(videos_embeddings.iterrows(), 2):
        if possible_combination[0][1]['id'] == possible_combination[1][1]['id'] and \
                possible_combination[0][1]['floor'] != possible_combination[1][1]['floor']:

            id_pairs.add((possible_combination[0][1]['video_id'], possible_combination[1][1]['video_id']))

    return list(id_pairs)


def get_different_people_videos_ids_pairs(videos_embeddings):
    id_pairs = set()

    for possible_combination in itertools.combinations(videos_embeddings.iterrows(), 2):
        if possible_combination[0][1]['id'] != possible_combination[1][1]['id'] and \
           possible_combination[0][1]['floor'] != possible_combination[1][1]['floor']:

            id_pairs.add((possible_combination[0][1]['video_id'], possible_combination[1][1]['video_id']))

    return list(id_pairs)


def get_accuracy(videos_embeddings, ids_pairs_list, distance_type='euclidean', confidence=0.5, threshold=1.193,
                 print_iterations=False):

    verifier = Verifier(videos_embeddings, distance_type, confidence, threshold)
    success_samples = 0

    for index, ids_pair in enumerate(ids_pairs_list):

        if print_iterations:
            print('[Getting accuracy (sample ' + str(index) + '/' + str(len(ids_pairs_list)) + ")]", end='\r')

        first_id = videos_embeddings[videos_embeddings['video_id'] == ids_pair[0]]['id'].unique()
        second_id = videos_embeddings[videos_embeddings['video_id'] == ids_pair[1]]['id'].unique()

        verifier_result = verifier.is_same_person(ids_pair[0], ids_pair[1])
        if first_id == second_id and verifier_result['is_same_person']:
            success_samples += 1
        elif first_id != second_id and not verifier_result['is_same_person']:
            success_samples += 1

    return round(success_samples*100/len(ids_pairs_list), 3)

