import numpy

from matplotlib import pyplot as plt
from thresholdvalidator import threshold_validator


class ThresholdPlotter:
    def __init__(self, videos_embeddings, distance_type='euclidean', threshold_min=0.01, threshold_max=2):

        assert (threshold_min < threshold_max), \
            "The threshold range cannot be (" + str(threshold_min) + "," + str(threshold_max) + ")"

        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.videos_embeddings = videos_embeddings
        self.distance_type = distance_type
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = plt.axes()
        self.far = list()
        self.frr = list()
        self.threshold_list = numpy.linspace(self.threshold_min, self.threshold_max,
                                             round(self.threshold_max / self.threshold_min))

    def draw_plot(self):
        plt.title("FAR vs FRR")
        self.ax.grid(color='b', ls='-.', lw=0.25)
        plt.ylabel("Error")
        plt.xlabel("Threshold")

        self.draw_errors()
        plt.legend(['FAR', 'FRR'], loc='upper left')

        plt.show()

    def draw_errors(self):

        ids = self.videos_embeddings['id'].unique()

        for index, threshold in enumerate(self.threshold_list):
            print("Calculating thresholds results... (" + str(index) + "/" + str(len(self.threshold_list)) + ")",
                  end='\r')

            false_acceptances_rates_list = list()
            false_rejections_rates_list = list()

            for person_id in ids:
                person_videos_embeddings = self.videos_embeddings[self.videos_embeddings['id'] == person_id]
                other_people_videos_embeddings = self.videos_embeddings[self.videos_embeddings['id'] != person_id]

                for floor in [1, 2, 3]:
                    [floor_far, floor_frr] = self.get_far_and_frr_from_person_in_floor(person_videos_embeddings,
                                                                                       other_people_videos_embeddings,
                                                                                       threshold,
                                                                                       floor)

                    false_acceptances_rates_list.append(floor_far)
                    false_rejections_rates_list.append(floor_frr)

            self.far.append(numpy.mean(false_acceptances_rates_list))
            self.frr.append(numpy.mean(false_rejections_rates_list))

        self.ax.plot(self.threshold_list, self.far, color='blue')
        self.ax.plot(self.threshold_list, self.frr, color='red')

    def calculate_best_threshold_and_eer(self):
        index = self.calculate_index()
        threshold_eer = self.threshold_list[index]
        eer = self.far[index]
        return [round(threshold_eer, 3), round(eer, 3)]

    def calculate_index(self):
        return int(numpy.argmin(numpy.absolute(numpy.subtract(self.far, self.frr))))

    def get_far_and_frr_from_person_in_floor(self, person_videos_embeddings, other_people_videos_embeddings,
                                             threshold, floor):
        original_embedding_matrix = person_videos_embeddings[(person_videos_embeddings['floor'] == floor)].iloc[:, 3:].to_numpy()
        comparing_embedding_matrix = person_videos_embeddings[(person_videos_embeddings['floor'] != floor)].iloc[:, 3:].to_numpy()
        total_columns_with_same_person = comparing_embedding_matrix.shape[0]
        comparing_embedding_matrix = numpy.vstack((comparing_embedding_matrix,
                                                   other_people_videos_embeddings[(other_people_videos_embeddings['floor'] != floor)].iloc[:, 3:].to_numpy()))

        return threshold_validator.measure_threshold(threshold,
                                                     original_embedding_matrix,
                                                     comparing_embedding_matrix,
                                                     total_columns_with_same_person,
                                                     distance_type=self.distance_type,
                                                     print_results=False)[:2]
