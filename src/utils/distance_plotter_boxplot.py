import numpy

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class DistancePlotterBoxplot:

    def __init__(self, same_person_euclidean_distance_list, different_people_euclidean_distance_list):

        #self.sv = same_video_euclidean_distance_list
        self.sp = same_person_euclidean_distance_list
        self.dp = different_people_euclidean_distance_list
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = plt.axes()
        # self.firstAxvlineLeft = 99
        # self.firstAxvlineRight = 110
        # self.secondAxvlineLeft = 124
        # self.secondAxvlineRight = 140

    def draw_plot(self):
        plt.title("Distances Ranges using Boxplot")
        self.ax.grid(color='b', ls='-.', lw=0.25)
        self.ax.get_xaxis().set_visible(False)
        self.draw_legend()
        plt.ylabel("Distance")

        # self.plot_same_video_results()
        self.plot_same_person_results()
        self.plot_different_people_results()

        self.draw_separators()

        plt.show()

    """
    Same video results are not plotted in last project version.
    def plot_same_video_results(self, x_range=(0, 49)):
        self.firstAxvlineLeft = x_range[1]
        if isinstance(self.sv, list):
            x_values = numpy.linspace(x_range[0], x_range[1], len(self.sv))
            bp = self.ax.boxplot(self.sv, sym='gs', positions=x_values, widths=0.75)
        else:
            x_values = [(x_range[0] + x_range[1])/2]
            bp = self.ax.boxplot(self.sv, sym='gs', positions=x_values, widths=7.5)

        for box in bp['boxes']:
            box.set(color='green')
    """

    def plot_same_person_results(self, x_range=(0, 75)):
        self.firstAxvlineRight = x_range[1]
        if isinstance(self.sp, list):
            x_values = numpy.linspace(x_range[0], x_range[1], len(self.sp))
            bp = self.ax.boxplot(self.sp, sym='bs', positions=x_values, widths=0.75)
        else:
            x_values = [(x_range[0] + x_range[1])/2]
            bp = self.ax.boxplot(self.sp, sym='bs', positions=x_values, widths=7.5)

        for box in bp['boxes']:
            box.set(color='blue')

    def plot_different_people_results(self, x_range=(85, 170)):
        self.firstAxvlineLeft = x_range[0]
        self.ax.set(xlim=(-5, x_range[1]))
        if isinstance(self.dp, list):
            x_values = numpy.linspace(x_range[0], x_range[1], len(self.dp))
            bp = self.ax.boxplot(self.dp, sym='rs', positions=x_values, widths=0.75)
        else:
            x_values = [(x_range[0] + x_range[1])/2]
            bp = self.ax.boxplot(self.dp, sym='rs', positions=x_values, widths=7.5)

        for box in bp['boxes']:
            box.set(color='red')

    def draw_legend(self):
        legend_lines = [# Line2D([0], [0], color='green', lw=2),
                        Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='red', lw=2)]

        self.ax.legend(legend_lines,
                       [# 'Frames from same video',
                        'Frames from a person on a floor VS same person on other floors',
                        'Frames from a person on a floor VS other people on other floors'],
                       loc='upper left',
                       prop={'size': 15})

    def draw_separators(self):
        self.ax.axvline((self.firstAxvlineRight + self.firstAxvlineLeft) / 2, color='k', ls='-', lw=2)
        # self.ax.axvline((self.secondAxvlineRight + self.secondAxvlineLeft) / 2, color='k', ls='-', lw=2)
