# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
#######################################################################


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk(object):
    def __init__(self, num_states=1000):
        self.num_states = num_states  # the number of states
        self.states = np.arange(1, num_states + 1)  # all states except terminal state
        self.start_state = int(num_states / 2)  # the start state
        self.end_state = [0, num_states + 1]  # the terminal states
        self.action = [-1, 1]  # right:1, left:-1
        self.neighbors = 100  # the neighboring states

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


class SingleTiling(object):
    def __init__(self, num_groups=10, num_states=1000):
        self.num_groups = num_groups                    # the number of groups
        # self.num_states = num_states                    # the number of states except for terminal states
        self.group_size = int(num_states / num_groups)  # the size of each group
        self.group_value = np.zeros(num_groups)         # the value of groups

    def get_state_value(self, state):
        """to get the state value except for terminal states"""
        group_idx = (state - 1) // self.group_size
        return self.group_value[group_idx]

    def update_state_value(self, state, delta):
        """to update the group value"""
        group_idx = (state - 1) // self.group_size
        self.group_value[group_idx] += delta


class MultipleTiling(object):
    def __init__(self, num_tilings=50, width_tiling=200, offset=4, num_states=1000):
        self.num_tilings = num_tilings          # the number of tilings
        self.width_tiling = width_tiling        # the width of tiling
        self.offset = offset                    # the size of offset

        self.size_tiling = num_states // width_tiling + 1
        self.value = np.zeros((num_tilings, self.size_tiling))
        self.tilings = np.arange(-width_tiling + 1, 0, offset)

    def get_state_value(self, state):
        """to get the state value except for terminal states"""
        state_value = 0
        for idx in range(len(self.tilings)):
            index_tiling = (state - self.tilings[idx]) // self.width_tiling
            state_value += self.value[idx, index_tiling]
        return state_value

    def update_state_value(self, state, delta):
        """to update the group value"""
        delta /= self.num_tilings
        for idx in range(len(self.tilings)):
            index_tiling = (state - self.tilings[idx]) // self.width_tiling
            self.value[idx, index_tiling] += delta


def gradient_monte_carlo(tiling, alpha):
    """ the gradient-descent version of Monte Carlo state-value prediction"""
    model = RandomWalk()
    state = model.start_state  # initialize the state
    trajectory = [state]  # track the transition state

    while state not in model.end_state:
        action = model.select_action()  # select an action
        next_state, reward = model.find_next_state(state, action)  # get the next state and reward
        trajectory.append(next_state)  # record the transition state
        state = next_state

    for stat in trajectory[:-1]:
        delta = alpha * (reward - tiling.get_state_value(stat))
        tiling.update_state_value(stat, delta)


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
                    value[state] += 1/(2*test_class.neighbors)*value[next_state]
        if np.linalg.norm(value - value_temp) < 0.001:
            break
    print("Completed!!!")
    return value


if __name__ == "__main__":
    runs = 10
    episodes = 5000
    # alpha_list = [1e-4, 2e-5]
    tiling_list = [SingleTiling(), MultipleTiling()]
    label_list = ["the single tiling", "the 50 tilings"]

    test_random_walk = RandomWalk()
    true_value = dp_compute_value(test_random_walk)
    errors = np.zeros((len(tiling_list), episodes))

    for run in range(runs):
        for ith in range(len(tiling_list)):
            for jth in tqdm(range(episodes)):
                alpha_temp = 1 / (10*(jth + 1))
                gradient_monte_carlo(tiling_list[ith], alpha_temp)
                val_estimating = np.asarray([tiling_list[ith].get_state_value(stat) for stat in
                                             test_random_walk.states])
                errors[ith, jth] += np.sqrt(np.mean(np.power(true_value[1: -1] - val_estimating, 2)))
    errors /= runs

    plt.figure(1)
    for itr in range(len(label_list)):
        plt.plot(errors[itr, :], label=label_list[itr])
    plt.xlabel("Episodes")
    plt.ylabel("Mean Square Root Value Error")
    plt.legend()
    plt.savefig("./images/Figure9-10.png")
    plt.show()
    plt.close()
    print("Completed!!!You can check it in 'images' directory")


