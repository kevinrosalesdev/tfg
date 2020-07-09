import numpy

from matplotlib import pyplot as plt
from accuracymeter import accuracy_meter


class ConfidencePlotter:
    def __init__(self, videos_embeddings, same_person_videos_id_pairs_list, different_person_videos_id_pairs_list,
                 distance_type='euclidean', threshold=0.5):

        self.sp_ids_pairs = same_person_videos_id_pairs_list
        self.dp_ids_pairs = different_person_videos_id_pairs_list
        self.mixed_pairs = self.sp_ids_pairs.copy()
        self.mixed_pairs.extend(self.dp_ids_pairs[:len(self.sp_ids_pairs)])
        self.videos_embeddings = videos_embeddings
        self.distance_type = distance_type
        self.threshold = threshold
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = plt.axes()
        self.same_person_accuracy = list()
        self.different_people_accuracy = list()
        self.mixed_samples_accuracy = list()
        self.confidence_rate_list = numpy.linspace(0, 1, 100)

    def draw_plot(self):
        plt.title("Same person accuracy vs Different people accuracy vs Mixed samples accuracy")
        self.ax.grid(color='b', ls='-.', lw=0.25)
        plt.ylabel("Accuracy")
        plt.xlabel("Confidence rate")

        self.draw_accuracy()
        plt.legend(['Same person accuracy', 'Different people accuracy', 'Mixed samples accuracy'], loc='upper left')

        plt.show()

    def draw_accuracy(self):
        for index, confidence_rate in enumerate(self.confidence_rate_list):
            print("Calculating confidence rate results... (" + str(index) + "/" + str(len(self.confidence_rate_list))
                  + ")", end='\r')

            self.same_person_accuracy.append(accuracy_meter.get_accuracy(self.videos_embeddings,
                                                                         self.sp_ids_pairs,
                                                                         distance_type=self.distance_type,
                                                                         confidence=confidence_rate,
                                                                         threshold=self.threshold))

            self.different_people_accuracy.append(accuracy_meter.get_accuracy(self.videos_embeddings,
                                                                              self.dp_ids_pairs,
                                                                              distance_type=self.distance_type,
                                                                              confidence=confidence_rate,
                                                                              threshold=self.threshold))

            self.mixed_samples_accuracy.append(accuracy_meter.get_accuracy(self.videos_embeddings,
                                                                           self.mixed_pairs,
                                                                           distance_type=self.distance_type,
                                                                           confidence=confidence_rate,
                                                                           threshold=self.threshold))

        self.ax.plot(self.confidence_rate_list, self.same_person_accuracy, color='blue')
        self.ax.plot(self.confidence_rate_list, self.different_people_accuracy, color='red')
        self.ax.plot(self.confidence_rate_list, self.mixed_samples_accuracy, color='green')

    def calculate_best_confidence_rate_and_accuracy(self):
        index = self.calculate_index()
        best_confidence_rate = self.confidence_rate_list[index]
        best_confidence_rate_same_person_accuracy = self.same_person_accuracy[index]
        best_confidence_rate_different_people_accuracy = self.different_people_accuracy[index]
        best_confidence_rate_mixed_samples_accuracy = self.mixed_samples_accuracy[index]
        return [round(best_confidence_rate, 3),
                round(best_confidence_rate_same_person_accuracy, 3),
                round(best_confidence_rate_different_people_accuracy, 3),
                round(best_confidence_rate_mixed_samples_accuracy, 3)]

    def calculate_index(self):
        return int(numpy.argmin(numpy.absolute(numpy.subtract(self.same_person_accuracy,
                                                              self.different_people_accuracy))))
