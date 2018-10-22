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

    def semi_gradient_td(self, state, weight):
        """the semi-gradient off-policy TD algorithm"""
        action = self.behavior_policy(state)            # choose an action in behavior policy
        next_state = self.find_next_state(state, action)        # find the next state
        # get the importance sampling ratio
        ratio = self.compute_ratio(action)
        delta = self.alpha*ratio*(self.reward + self.discount*np.dot(self.features[next_state, :], weight) -
                                  np.dot(self.features[state, :], weight))
        # update the weight vector
        weight += delta*self.features[state, :]
        return next_state

    def semi_gradient_dp(self, weight):
        """the semi-gradient Dynamic Programming"""
        delta = 0           # track the difference between expected value and estimate value
        for stat in self.states:
            returns = 0     # track the expected returns
            # compute Bellman error
            for next_stat in self.states:
                if next_stat == self.lower_state:
                    # calculate returns
                    returns += self.reward + self.discount*np.dot(self.features[next_stat, :], weight)
            error = returns - np.dot(self.features[stat, :], weight)
            delta += error*self.features[stat, :]
        # update the weight vector
        weight += self.alpha/len(self.states)*delta


if __name__ == "__main__":
    _alpha_ = 0.01          # the step size
    episodes = 1000         # the size of episodes
    example_class = BairdCounterExample(alpha=_alpha_)

    td_init_weight = np.ones(example_class.feature_size)     # the weight vector in semi-gradient TD
    td_init_weight[6] = 10
    td_weight_mat = np.zeros((example_class.feature_size, episodes))   # the weight matrix in semi-gradient TD
    dp_init_weight = td_init_weight.copy()                  # the weight vector in semi-gradient DP
    dp_weight_mat = td_weight_mat.copy()                    # the weight matrix in semi-gradient DP

    init_state = np.random.choice(example_class.states)     # initialize the state
    for ith in tqdm(range(episodes)):
        init_state = example_class.semi_gradient_td(init_state, td_init_weight)
        td_weight_mat[:, ith] = td_init_weight

        example_class.semi_gradient_dp(dp_init_weight)
        dp_weight_mat[:, ith] = dp_init_weight

    # starting to figure
    weight_mat = [td_weight_mat, dp_weight_mat]
    x_label = ["Steps", "Sweeps"]
    title = ["Semi-gradient Off-policy TD", "Semi-gradient DP"]
    for i in range(len(weight_mat)):
        plt.figure(i+1)
        for j in range(example_class.feature_size):
            plt.plot(weight_mat[i][j, :], label=r"$\omega_{%d}$" % (j + 1))
        plt.xlabel(x_label[i])
        plt.ylabel("Weight Value")
        plt.title(title[i])
        plt.legend()
        plt.savefig("./images/Figure11-"+str(i + 1)+".png")
        plt.show()
    plt.close()
    print("Completed!!!You can check it in 'images' directory")

