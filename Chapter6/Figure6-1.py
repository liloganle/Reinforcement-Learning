# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def draw_image_MC(labels, predicted_time, real_outcome):
    """
    to plot the figure of the driving home example by Monte Carlo methods
    :param labels: states of driving home
    :param predicted_time: predicted total time
    :param real_outcome: the actual outcome
    :return:
    """
    print("\nStarting drawing......")
    plt.plot(labels, predicted_time, "-o")
    plt.plot(real_outcome, "--")
    for i in np.arange(len(real_outcome)):
        plt.annotate(s="", xytext=(i, predicted_time[i]), xy=(i, real_outcome[i]),
                     arrowprops=dict(arrowstyle="->"))

    plt.xlabel("Situation")
    plt.ylabel("Predicted Total Time")
    plt.savefig("./images/Figure6-1-1.png")
    plt.show()
    plt.close()
    print("Completed !!! You can check it in the 'images' directory!")

def draw_image_TD(labels, predicted_time):
    """
    to plot the figure of the driving home example by Temporal-Difference methods
    :param labels: states of driving home
    :param predicted_time: predicted total time
    :return:
    """
    iteration = (len(labels) - 1) * 2
    print("\nStarting drawing......")
    plt.plot(labels, predicted_time, "-o")
    for i in np.arange(iteration):
        if i % 2 == 0:
            i = int(i/2)
            plt.annotate(s="", xytext=(i, predicted_time[i]), xy=(i, predicted_time[i+1]),
                         arrowprops=dict(arrowstyle="->"))
        else:
            start = int((i - 1)/2)
            end = int((i + 1)/2)
            plt.plot(np.arange(start, end+1), predicted_time[end]*np.ones(end-start+1), "r--")

    plt.xlabel("Situation")
    plt.ylabel("Predicted Total Time")
    plt.savefig("./images/Figure6-1-2.png")
    plt.show()
    plt.close()
    print("Completed !!! You can check it in the 'images' directory!")



if __name__ == "__main__":
    x_label = ["leaving \noffice", "reach \ncar", "exiting \nhighway", "secondary \nroad",
               "home \nstreet", "arrive \nhome"]
    predicted_total_time = np.array([30, 40, 35, 40, 43, 43])
    actual_outcome = 43 * np.ones(len(predicted_total_time))

    draw_image_MC(x_label, predicted_total_time, actual_outcome)
    draw_image_TD(x_label, predicted_total_time)