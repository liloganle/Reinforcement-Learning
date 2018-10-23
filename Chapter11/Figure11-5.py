# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BairdCounterExample(object):
    def __init__(self, alpha, beta, num_state=7, discount=0.99):
        self.alpha = alpha                      # the step size of the primary learning process
        self.beta = beta                        # the step size of the secondary learning process
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

    def behavior_policy(self, state):
        """choose an action according to the behavior policy"""
        if np.random.binomial(1, 1/self.num_state):             # take the solid action
            return self.solid_action
        else:                                   # take the dashed action
            return self.dashed_action

    def target_policy(self, state):
        """choose an action according to the target policy"""
        return self.solid_action                # always take the solid action

    def find_next_state(self, state, action):
        """take @action in @state, and return next state"""
        if action == self.dashed_action:
            return np.random.choice(self.upper_states)
        else:
            return self.lower_state

    def compute_ratio(self, action):
        """compute the importance sampling ratio"""
        if action == self.dashed_action:
            return 0                    # the target policy always takes the solid action, so 0/(6/7)=0
        else:
            return 1/(1/self.num_state)     # the target policy always takes the solid action, so 1/(1/7)

    def td_gradient_correction(self, state, v_estimate, weight):
        """The algorithm is known as TD(0) with gradient correction (TDC) or GTD(0)"""
        action = self.behavior_policy(state)            # choose an action
        next_state = self.find_next_state(state, action)            # get the next state
        ratio = self.compute_ratio(action)              # compute the importance sampling ratio
        delta = self.reward + self.discount*np.dot(self.features[next_state, :],
                                                   weight) - np.dot(self.features[state, :], weight)
        # update the second learned vector V
        v_estimate += self.beta*ratio*(delta - np.dot(v_estimate, self.features[state, :]))*self.features[state, :]
        # update the weight vector W
        weight += self.alpha*ratio*(delta*self.features[state, :] - self.discount*self.features[next_state, :] *
                                    np.dot(self.features[state, :], v_estimate))
        return next_state

    def expected_td_correction(self, v_estimate, weight):
        """the expected behavior of TDC"""
        for stat in self.states:
            action = self.target_policy(stat)           # choose an action
            next_stat = self.find_next_state(stat, action)          # get the next state
            # calculate the delta
            delta = self.reward + self.discount*np.dot(self.features[next_stat, :], weight) - \
                    np.dot(self.features[stat, :], weight)
            # get the importance sampling ratio
            ratio = self.compute_ratio(action)
            expected_v = 1/self.num_state*ratio*(delta - np.dot(self.features[stat, :],
                                                                v_estimate)) * self.features[stat, :]
            v_estimate += self.beta/self.num_state * expected_v
            expected_weight = 1/self.num_state*ratio*(delta * self.features[stat, :] - self.discount *
                                                      self.features[next_stat, :] *
                                                      np.dot(self.features[stat, :], v_estimate))
            weight += self.alpha/self.num_state * expected_weight


def compute_projection_matrix(_class_, state_distribution):
    """compute the projection matrix over the Bellman Error"""
    diag_matrix = np.matrix(np.diag(state_distribution))            # shape: 7x7
    inverse_matrix = np.linalg.pinv(np.matrix(_class_.features.T)*diag_matrix*np.matrix(_class_.features))   # shape:8x8
    # compute the projected matrix
    projection_matrix = np.matrix(_class_.features)*inverse_matrix*np.matrix(_class_.features.T)*diag_matrix
    return np.asarray(projection_matrix)                #


def compute_value_error(_class_, weight, state_distribution):
    """compute the root mean square value error"""
    true_value = np.zeros(_class_.num_state)            # the true state value
    value_error = np.dot(np.power(np.dot(_class_.features, weight) - true_value, 2), state_distribution)
    return np.sqrt(value_error)


def compute_bellman_error(_class_, weight, state_distribution, projection_matrix):
    """compute the root mean square projected Bellman error"""
    delta = np.zeros(_class_.num_state)                 # record the bellman errors
    for stat in _class_.states:
        delta[stat] += _class_.reward + _class_.discount*np.dot(_class_.features[_class_.lower_state, :],
                                                                weight) - np.dot(_class_.features[stat, :], weight)
    projected_bellman_error = np.dot(projection_matrix, delta)          # projected Bellman Error
    return np.sqrt(np.dot(np.power(projected_bellman_error, 2), state_distribution))


if __name__ == "__main__":
    alpha_float = 0.005             # the size of alpha
    beta_float = 0.05               # the size of beta
    episodes = 1000
    example_class = BairdCounterExample(alpha=alpha_float, beta=beta_float)         # create an class

    tdc_v_value = np.zeros(example_class.feature_size)                  # the v value
    tdc_weight_vector = np.ones(example_class.feature_size)             # the weight vector
    tdc_weight_vector[6] = 10
    expected_v_value = tdc_v_value.copy()
    expected_weight_vector = tdc_weight_vector.copy()

    tdc_weight_array = np.zeros((example_class.feature_size, episodes))
    tdc_value_error = np.zeros(episodes)        # record the value error
    tdc_bell_error = np.zeros(episodes)         # record the Bellman error
    expected_weight_array = tdc_weight_array.copy()
    expected_value_error = tdc_value_error.copy()
    expected_bell_error = tdc_bell_error.copy()

    distribution = np.ones(example_class.num_state)/example_class.num_state             # uniform distribution
    projection_mat = compute_projection_matrix(example_class, distribution)             # projection distribution

    init_state = np.random.choice(example_class.states)          # initialize the state
    for ith in tqdm(range(episodes)):
        # the TDC algorithm
        init_state = example_class.td_gradient_correction(init_state, tdc_v_value, tdc_weight_vector)
        tdc_weight_array[:, ith] += tdc_weight_vector
        tdc_value_error[ith] += compute_value_error(example_class, tdc_weight_vector, distribution)
        tdc_bell_error[ith] += compute_bellman_error(example_class, tdc_weight_vector, distribution, projection_mat)
        # the Expected TDC algorithm
        example_class.expected_td_correction(expected_v_value, expected_weight_vector)
        expected_weight_array[:, ith] += expected_weight_vector
        expected_value_error[ith] += compute_value_error(example_class, expected_weight_vector, distribution)
        expected_bell_error[ith] += compute_bellman_error(example_class, expected_weight_vector, distribution,
                                                          projection_mat)

    title = ["TDC", "Expected TDC"]
    x_label = ["Steps", "Sweeps"]
    weight_mat = [tdc_weight_array, expected_weight_array]
    value_error = [tdc_value_error, expected_value_error]
    bellman_error = [tdc_bell_error, expected_bell_error]

    for ith in range(len(title)):
        plt.figure(ith + 1)
        for jth in range(example_class.feature_size):
            plt.plot(weight_mat[ith][jth, :], label=r"$\omega_{%d}$" % (jth + 1))
        plt.plot(value_error[ith], label=r"$\sqrt{\bar{VE}}$")
        plt.plot(bellman_error[ith], label=r"$\sqrt{\bar{PBE}}$")
        plt.xlabel(x_label[ith])
        plt.title(title[ith])
        plt.legend()
        plt.savefig("./images/Figure11-5-"+str(ith + 1)+".png")
        plt.show()
        plt.close()
    print("Completed!!!You can check it in 'images' directory")

