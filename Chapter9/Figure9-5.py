# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk(object):
    def __init__(self, num_states=1000, groups=10, alpha=2e-5, order=5, flag=None):
        self.num_states = num_states        # the number of states
        self.groups = groups                # the number of groups
        self.alpha = alpha                  # the step size

        self.states = np.arange(1, num_states+1)        # all states except terminal state
        self.start_state = int(num_states / 2)          # the start state
        self.end_state = [0, num_states + 1]            # the terminal states
        self.action = [-1, 1]                           # right:1, left:-1
        self.neighbors = 100                            # the neighboring states

        self.order = order          # Fourier and polynomial bases of order
        self.weights = np.zeros(order + 1)
        self.bases = []             # bases function
        if flag == "Polynomial":
            for ith in range(order + 1):
                self.bases.append(lambda stat, i=ith: np.power(stat, i))
        if flag == "Fourier":
            for ith in range(order + 1):
                self.bases.append(lambda stat, i=ith: np.cos(np.pi * i * stat))

    def select_action(self):
        """to select randomly an action"""
        if np.random.binomial(1, 0.5):
            return self.action[1]  # select right action
        else:
            return self.action[0]  # select left action

    def find_next_state(self, state, action):
        """to get the next state and reward"""
        move_step = np.random.randint(1, self.neighbors + 1)  # the step size of moving
        move_step *= action
        next_state = state + move_step  # the next state
        next_state = max(min(next_state, self.end_state[1]), 0)

        if next_state == self.end_state[0]:  # terminating on the left
            reward = -1
        elif next_state == self.end_state[1]:  # terminating on the right
            reward = 1
        else:
            reward = 0
        return next_state, reward

    def get_state_value(self, state):
        """to get the state value"""
        state /= self.num_states                # normalize the state
        feature_vector = np.asarray([base(state) for base in self.bases])       # the feature vector
        return np.dot(self.weights, feature_vector)

    def update_group_value(self, state, delta):
        """to update the group value"""
        state /= self.num_states                # normalize the state
        gradient_vector = np.asarray([base(state) for base in self.bases])      # the gradient vector
        self.weights += delta * gradient_vector

    def gradient_monte_carlo(self):
        """ the gradient-descent version of Monte Carlo state-value prediction"""
        state = self.start_state  # initialize the state
        trajectory = [state]  # track the transition state

        while state not in self.end_state:
            action = self.select_action()  # select an action
            next_state, reward = self.find_next_state(state, action)  # get the next state and reward
            trajectory.append(next_state)  # record the transition state
            state = next_state

        for stat in trajectory[:-1]:
            delta = self.alpha * (reward - self.get_state_value(stat))
            self.update_group_value(stat, delta)


def dp_compute_value(test_class):
    """using Dynamic programming to find the true state values"""
    value = np.arange(-test_class.end_state[1], test_class.end_state[1] + 1, 2) / test_class.end_state[1]
    print("Starting computing......")
    while True:
        value_temp = value.copy()
        for state in test_class.states:
            value[state] = 0
            for act in test_class.action:
                for step in range(1, test_class.neighbors + 1):
                    step *= act
                    next_state = state + step
                    next_state = max(min(next_state, test_class.end_state[1]), 0)
                    # update the value
                    value[state] += 1 / (2 * test_class.neighbors) * value[next_state]
        if np.linalg.norm(value - value_temp) < 0.001:
            break
    print("Completed!!!")
    return value


def compute_rms_error(true_value, orders, alphas, flags, episodes):
    """to compute the mean square root error"""

    error_rms = np.zeros((len(flags), len(orders), episodes))
    for row in range(len(flags)):
        for col in range(len(orders)):
            test_class = RandomWalk(alpha=alphas[row], order=orders[col], flag=flags[row])
            for itr in tqdm(range(episodes)):
                test_class.gradient_monte_carlo()        # gradient Monte Carlo algorithm
                state_value = np.asarray([test_class.get_state_value(stat) for stat in test_class.states])
                # calculate the Mean Square Root Error
                error_rms[row, col, itr] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_value, 2)))
    return error_rms


if __name__ == "__main__":
    episodes_int = 5000
    orders_list = [5, 10, 20]
    alphas_list = [1e-4, 5e-5]
    flags_list = ["Polynomial", "Fourier"]
    labels_list = ["Polynomial basis", "Fourier basis"]

    runs = 10
    errors = 0
    test_exam = RandomWalk()
    real_value = dp_compute_value(test_class=test_exam)
    for run in range(runs):
        errors += compute_rms_error(real_value, orders_list, alphas_list, flags_list, episodes_int)
    errors /= runs

    plt.figure(1)
    for i in range(len(flags_list)):
        for j in range(len(orders_list)):
            plt.plot(errors[i, j, :], label="%s,%d order" % (labels_list[i], orders_list[j]))
    plt.xlabel("Episodes")
    plt.ylabel("Mean Square Root Error")
    plt.legend()
    plt.savefig("./images/Figure9-5.png")
    plt.show()
    plt.close()
    print("Completed!!!You can check it in 'images' directory")


