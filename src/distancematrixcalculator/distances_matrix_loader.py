from distancematrixcalculator import distances_matrix_calculator


def get_same_video_distance_matrices(videos_embeddings, distance_type='euclidean'):
    assert distance_type == 'euclidean' or distance_type == 'cosine', "Parameter 'distance_type' cannot be different " \
                                                                      "from 'euclidean' or 'cosine'."
    same_video_results_list = list()
    same_video_euclidean_distance_list = list()

    ids = videos_embeddings['id'].unique()

    for person_id in ids:
        person_videos_embeddings = videos_embeddings[videos_embeddings['id'] == person_id]

        same_video_result_1st_floor = distances_matrix_calculator.get_same_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 1)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )

        same_video_results_list.append(same_video_result_1st_floor[:2])
        same_video_euclidean_distance_list.append(same_video_result_1st_floor[2])

        same_video_result_2nd_floor = distances_matrix_calculator.get_same_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 2)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )

        same_video_results_list.append(same_video_result_2nd_floor[:2])
        same_video_euclidean_distance_list.append(same_video_result_2nd_floor[2])

        same_video_result_3rd_floor = distances_matrix_calculator.get_same_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 3)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )

        same_video_results_list.append(same_video_result_3rd_floor[:2])
        same_video_euclidean_distance_list.append(same_video_result_3rd_floor[2])

    return [same_video_results_list, same_video_euclidean_distance_list]


def get_same_person_distance_matrices(videos_embeddings, distance_type='euclidean'):
    assert distance_type == 'euclidean' or distance_type == 'cosine', "Parameter 'distance_type' cannot be different " \
                                                                      "from 'euclidean' or 'cosine'."
    same_person_results_list = list()
    same_person_euclidean_distance_list = list()

    ids = videos_embeddings['id'].unique()

    for person_id in ids:
        person_videos_embeddings = videos_embeddings[videos_embeddings['id'] == person_id]

        same_person_result_1_vs_23 = distances_matrix_calculator.get_different_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 1)].iloc[:, 3:].to_numpy(),
            person_videos_embeddings[(person_videos_embeddings['floor'] != 1)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )
        same_person_results_list.append(same_person_result_1_vs_23[:2])
        same_person_euclidean_distance_list.append(same_person_result_1_vs_23[2])

        same_person_result_2_vs_13 = distances_matrix_calculator.get_different_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 2)].iloc[:, 3:].to_numpy(),
            person_videos_embeddings[(person_videos_embeddings['floor'] != 2)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )
        same_person_results_list.append(same_person_result_2_vs_13[:2])
        same_person_euclidean_distance_list.append(same_person_result_2_vs_13[2])

        same_person_result_3_vs_12 = distances_matrix_calculator.get_different_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 3)].iloc[:, 3:].to_numpy(),
            person_videos_embeddings[(person_videos_embeddings['floor'] != 3)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )
        same_person_results_list.append(same_person_result_3_vs_12[:2])
        same_person_euclidean_distance_list.append(same_person_result_3_vs_12[2])

    return [same_person_results_list, same_person_euclidean_distance_list]


def get_different_people_distance_matrices(videos_embeddings, distance_type='euclidean'):
    assert distance_type == 'euclidean' or distance_type == 'cosine', "Parameter 'distance_type' cannot be different " \
                                                                      "from 'euclidean' or 'cosine'."
    different_people_results_list = list()
    different_people_euclidean_distance_list = list()

    ids = videos_embeddings['id'].unique()

    for person_id in ids:
        person_videos_embeddings = videos_embeddings[videos_embeddings['id'] == person_id]
        other_people_videos_embeddings = videos_embeddings[videos_embeddings['id'] != person_id]

        different_people_result_1_vs_23 = distances_matrix_calculator.get_different_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 1)].iloc[:, 3:].to_numpy(),
            other_people_videos_embeddings[(other_people_videos_embeddings['floor'] != 1)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )
        different_people_results_list.append(different_people_result_1_vs_23[:2])
        different_people_euclidean_distance_list.append(different_people_result_1_vs_23[2])

        different_people_result_2_vs_13 = distances_matrix_calculator.get_different_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 2)].iloc[:, 3:].to_numpy(),
            other_people_videos_embeddings[(other_people_videos_embeddings['floor'] != 2)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )
        different_people_results_list.append(different_people_result_2_vs_13[:2])
        different_people_euclidean_distance_list.append(different_people_result_2_vs_13[2])

        different_people_result_3_vs_12 = distances_matrix_calculator.get_different_video_range(
            person_videos_embeddings[(person_videos_embeddings['floor'] == 3)].iloc[:, 3:].to_numpy(),
            other_people_videos_embeddings[(other_people_videos_embeddings['floor'] != 3)].iloc[:, 3:].to_numpy(),
            distance_type=distance_type
        )
        different_people_results_list.append(different_people_result_3_vs_12[:2])
        different_people_euclidean_distance_list.append(different_people_result_3_vs_12[2])

    return [different_people_results_list, different_people_euclidean_distance_list]
