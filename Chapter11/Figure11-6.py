# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BairdCounterExample(object):
    def __init__(self, alpha, num_state=7, discount=0.99):
        self.alpha = alpha                      # the step size
        self.num_state = num_state              # the number of states
        self.discount = discount                # the size of discounting

        self.states = np.arange(num_state)       # all the states: 0-6
        self.upper_states = self.states[:num_state - 1]     # the upper states: 0-5
        self.lower_state = self.states[num_state - 1]        # the lower state: 6

        self.dashed_action = 0
        self.solid_action = 1
        self.actions = [self.dashed_action, self.solid_action]

        self.reward = 0                         # the reward is 0 on all transitions
        self.feature_size = 8                   # the size of feature vector
        self.features = np.zeros((num_state, self.feature_size))        # shape:7x8
        for row in self.upper_states:
            self.features[row, row] = 2
            self.features[row, self.feature_size-1] = 1
        self.features[self.lower_state, self.lower_state] = 1
        self.features[self.lower_state, self.feature_size-1] = 2

    def compute_ratio(self, state):
        """compute the importance sampling ratio"""
        if state == self.lower_state:
            return 1/(1/self.num_state)         # the target policy always takes the solid action, so 1/(1/7)
        else:
            return 0                            # the target policy always takes the solid action, so 0/(1/7)

    def emphatic_td(self, weight, emphasis, interest=1):
        """the one-step Emphatic-TD algorithm"""
        update_term = 0
        expect_emphasis = 0         # record the emphasis trajectory
        for stat in self.states:
            # get the importance sampling ratio
            ratio = self.compute_ratio(stat)
            # calculate the next emphasis
            next_emphasis = self.discount*ratio*emphasis + interest
            expect_emphasis += next_emphasis
            delta = self.reward + self.discount*np.dot(self.features[self.lower_state, :], weight
                                                       ) - np.dot(self.features[stat, :], weight)
            update_term += 1/self.num_state * next_emphasis * delta * self.features[stat, :]
        weight += self.alpha * update_term
        return expect_emphasis/self.num_state


def compute_value_error(_class_, weight, state_distribution):
    """compute the root mean square value error"""
    true_value = np.zeros(_class_.num_state)            # the true state value
    value_error = np.dot(np.power(np.dot(_class_.features, weight) - true_value, 2), state_distribution)
    return np.sqrt(value_error)


if __name__ == "__main__":
    alpha_float = 0.03
    sweeps = 1000
    example_class = BairdCounterExample(alpha=alpha_float)
    distribution = np.ones(example_class.num_state)/example_class.num_state

    weight_vector = np.ones(example_class.feature_size)
    weight_vector[6] = 10
    weight_array = np.zeros((example_class.feature_size, sweeps))
    error_array = np.zeros(sweeps)

    emphasis_init = 0
    for ith in tqdm(range(sweeps)):
        emphasis_init = example_class.emphatic_td(weight_vector, emphasis_init)
        weight_array[:, ith] += weight_vector
        error_array[ith] += compute_value_error(example_class, weight_vector, distribution)

    plt.figure(1)
    for ith in range(example_class.feature_size):
        plt.plot(weight_array[ith, :], label=r"$\omega_{%d}$" % (ith + 1))
    plt.plot(error_array, label=r"$\sqrt{\overline{VE}}$")
    plt.xlabel("Sweeps")
    plt.title("Emphatic-TD")
    plt.legend()
    plt.savefig("./images/Figure11-6.png")
    plt.show()
    plt.close()
    print("Completed!!!You can check it in 'images' directory")




