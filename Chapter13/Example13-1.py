# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def true_state_value(probability):
    """
    compute the true value of the first state according to dynamics
    :param probability: the probability of right action
    :return: the true value

    V(S1)=p*1*(-1+V(S2))+(1-p)*1*(-1+V(S1))
    V(S2)=p*1*(-1+V(S1))+(1-p)*1*(-1+V(S3))
    V(S3)=p*1*(-1+0)+(1-p)*1*(-1+V(S2))
    p is the probability of right action
    ===>       V(S1) = 2*(p-2)/(p*(1-p))
    """
    return 2*(probability-2)/(probability*(1-probability))


if __name__ == "__main__":
    epsilon = 0.1           # the epsilon-greedy
    probability_range = np.linspace(0.03, 0.98, 100)        # set the probability of right action
    state_value = true_state_value(probability_range)       # get the true state value

    idx_max = np.argmax(state_value)            # get the index of the maximum of state value

    plt.figure(1)
    plt.plot(probability_range, state_value, "k")
    plt.plot(probability_range[idx_max], state_value[idx_max], "ro", label=r"optimal stochastic policy")
    plt.plot(epsilon/2, true_state_value(epsilon/2), "bo", label=r"$\epsilon$-greedy left")
    plt.plot(1-epsilon/2, true_state_value(1-epsilon/2), "go", label=r"$\epsilon$-greedy right")
    plt.ylim(-100)
    plt.xlabel("Probability of right action")
    plt.ylabel("True Value of First State")
    plt.legend()
    plt.savefig("./images/Example13-1.png")
    plt.show()
    plt.close()

