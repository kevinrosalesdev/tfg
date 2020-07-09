from distancematrixcalculator import distances_matrix_calculator
import numpy


class Verifier:
    def __init__(self, videos_embeddings, distance_type='euclidean', confidence=0.5, threshold=1.193):
        self.videos_embeddings = videos_embeddings
        self.confidence = confidence
        self.threshold = threshold
        self.distance_type = distance_type

    def is_same_person(self, id_video_1, id_video_2):
        embedding_1 = self.videos_embeddings[self.videos_embeddings['video_id'] == id_video_1].iloc[:, 3:].to_numpy()
        embedding_2 = self.videos_embeddings[self.videos_embeddings['video_id'] == id_video_2].iloc[:, 3:].to_numpy()
        distance_matrix = distances_matrix_calculator.get_different_video_range(embedding_1,
                                                                                embedding_2,
                                                                                distance_type=self.distance_type)[2]
        same_person_samples = numpy.where(distance_matrix < self.threshold)[0].shape[0]
        same_person_rate = same_person_samples / distance_matrix.shape[0]

        if same_person_rate >= self.confidence:
            return {'is_same_person': True,
                    'confidence': round(same_person_rate, 3),
                    'distance_matrix': distance_matrix}

        return {'is_same_person': False,
                'confidence': round(1 - same_person_rate, 3),
                'distance_matrix': distance_matrix}
