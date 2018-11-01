# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def soft_max(preference):
    """an exponential soft-max distribution"""
    var = np.exp(preference - np.mean(preference))          # normalization
    return var/np.sum(var)


class ShortCorridor(object):
    def __init__(self, alpha, gamma):
        self.alpha = alpha              # the step size
        self.gamma = gamma              # the discounted factor
        self.epsilon = 0.1              # the epsilon-greedy

        self.num_state = 4              # the number of states
        self.num_action = 2             # the number of actions
        self.states = np.arange(self.num_state)     # all states
        self.actions = np.arange(self.num_action)   # 0:left ,1:right
        self.start_state = self.states[0]           # start state:0
        self.end_state = self.states[-1]            # terminal state:3

        self.feature_vector = np.array([[0, 1],
                                        [1, 0]])    # the feature vector for preference function
        self.init_theta = np.zeros(len(self.feature_vector[:, 0]))  # initialize the policy parameters
        self.action_list = []           # record the transition action
        self.reward_list = []                # the transition reward

    def compute_policy(self):
        """get the pi, the action distribution"""
        h_function = np.dot(self.init_theta, self.feature_vector)       # the preference in linear form
        pi_function = soft_max(h_function)              # the soft max distribution

        return pi_function

    def select_action(self):
        """choose the action"""
        pi_func = self.compute_policy()
        max_val = np.max(pi_func)                # take the left action
        return np.random.choice([act for act, val in enumerate(pi_func) if val == max_val])

    def find_next_state(self, state, action):
        """find the next state according ro the right flag"""
        reward = -1             # set all rewards default is -1
        if state == self.states[0] or state == self.states[2]:
            if action == 1:                     # take right action
                next_state = state + 1
            else:                               # take left action
                next_state = max(0, state - 1)
        else:
            if action == 1:
                next_state = state - 1
            else:
                next_state = state + 1

        if next_state == self.end_state:            # reach the terminal state
            reward = 0

        return next_state, reward

    def monte_carlo_policy_gradient(self):
        """the Monte Carlo policy gradient"""
        reward_sum = 0
        init_state = self.start_state           # the initialized state
        while init_state != self.end_state:
            action = self.select_action()       # choose an action according to the preference
            next_state, reward = self.find_next_state(init_state, action)  # find the next state
            self.action_list.append(action)          # add new action to the list of action
            self.reward_list.append(reward)          # add new reward to the list of reward
            reward_sum += reward
            init_state = next_state             # set initialized state equal to the next state

        returns = np.zeros(len(self.reward_list))  # set default return is zeros
        returns[-1] = self.reward_list[-1]
        for idx in range(2, len(self.reward_list) + 1):
            returns[-idx] = self.reward_list[-idx] + self.gamma*returns[-idx + 1]  # update the returns of this episode

        gamma_power = 1
        for idx in range(len(returns)):
            row = self.action_list[idx]
            pi_func = self.compute_policy()             # get the pi distribution
            gradient = self.feature_vector[:, row] - np.dot(pi_func, self.feature_vector)
            self.init_theta += self.alpha * gamma_power * returns[idx] * gradient  # update the theta
            gamma_power *= self.gamma
        return reward_sum

    def reset(self):
        """reset the list of reward and action"""
        self.reward_list = []
        self.action_list = []


if __name__ == "__main__":
    runs = 50             # the size of running
    episodes = 1000         # the number of episodes
    alpha_list = [1.25e-4]
    var_gamma = 1

    rewards = np.zeros((len(alpha_list), runs, episodes))
    for alpha_idx in range(len(alpha_list)):
        problem = ShortCorridor(alpha=alpha_list[alpha_idx], gamma=var_gamma)
        for run in tqdm(range(runs)):
            for epi in range(episodes):
                rewards[alpha_idx, run, epi] = problem.monte_carlo_policy_gradient()
                problem.reset()
                # print("the "+str(epi+1)+"-th theta is ", method.init_theta)
    # print("The reward array's shape is ", rewards[0].shape)
    x_labels = ["r", "b", "g"]
    plt.figure(1)
    plt.plot(np.arange(episodes)+1, -11.6 * np.ones(episodes), "k-", label="-11.6")
    for i in range(len(alpha_list)):
        plt.plot(np.arange(episodes)+1, rewards[i].mean(axis=0), x_labels[i])
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward on Episode")
    plt.legend()
    plt.savefig("./images/Figure13-1.png")
    plt.show()
    plt.close()


