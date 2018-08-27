# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def draw_image(num_action=10, num_problem=2000):
    """
    to draw figure 2.1
    :param num_action: the number of actions
    :param num_problem: the number of problems
    :return:None
    """
    print("Start Drawing the figure......\n")

    data = np.random.randn(num_problem, num_action) + np.random.randn(num_action)
    plt.violinplot(dataset=data, showmeans=True, showextrema=False)
    plt.xlim(0, num_action+1)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig("./images/figure2_1.png")  # to save the figure
    plt.show()
    plt.close()

    print("Completed!")

if __name__ == "__main__":
    draw_image()