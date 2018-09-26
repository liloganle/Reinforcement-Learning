# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_rms_error(branch_factor):
    """
    to compute the root mean square error in value estimate
    :param branch_factor: the size of the branching factor
    :return: errors
    """
    true_branch = np.random.randn(branch_factor)    # to generate the branching
    true_value = np.mean(true_branch)               # the true value of the current state

    samples = np.zeros(2*branch_factor)
    errors = np.zeros(samples.shape)

    for idx in range(2*branch_factor):
        sampling = np.random.choice(true_branch)
        samples[idx] = sampling
        errors[idx] = np.abs(np.mean(samples[:idx+1]) - true_value)

    return errors


if __name__ == "__main__":
    branching_factor = [2, 10, 100, 1000, 10000]        # the list of a variety of branching factor
    runs = 100      # the number of runs

    for branch in branching_factor:
        rms_errors = np.zeros((runs, 2*branch))     # to store the error of each run
        for run in tqdm(range(runs)):
            rms_errors[run] = compute_rms_error(branch)
        # average the root mean square error
        rms_errors = np.mean(rms_errors, axis=0)
        x_axis = (np.arange(len(rms_errors)) + 1)/branch
        plt.plot(x_axis, rms_errors, label=r"b=%d" % branch)

    expected_error = np.ones(len(rms_errors))
    expected_error[x_axis >= 1] = 0
    plt.plot(x_axis, expected_error, label=r"expected")

    plt.xlabel("Number of MAX Computation")
    plt.xticks([0, 1, 2], ["0", "1b", "2b"])
    plt.ylabel("RMS Error in Value Estimate")
    plt.legend()
    plt.savefig("./images/Figure8-7.png")
    plt.show()
    plt.close()
    print("Completed!!! You can check it in the 'images' directory")


