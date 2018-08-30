# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

'''
class Ten_Armed_Bandit(object):
    def __init__(self, arms=10, epsilons=0.):
        self.k = arms
        self.index = np.arange(arms)
        self.epsilon = epsilons
        self.average_reward = 0

    def initialize(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimate = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)

    def select_action(self):
        rand_num = np.random.rand()
        if rand_num < self.epsilon:
            return np.random.choice(self.index)
        else:
            return np.argmax(self.q_estimate)

    def feedback_reward(self, action):
        reward = self.q_true[action] + np.random.randn()
        self.action_count[action] += 1
        self.q_estimate[action] += (reward - self.q_estimate[action])/self.action_count[action]

        return reward
'''


def initialize():
    """
    to initialize the parameters
    :return:
    """
    num_actions = 10

    q_true = np.random.randn(num_actions)
    q_estimation = np.zeros(num_actions)
    action_count = np.zeros(num_actions)
    action_index = np.arange(num_actions)
    best_action = np.argmax(q_true)

    stat = {"q_true": q_true, "q_estimation": q_estimation, "action_count": action_count,
            "action_index": action_index, "best_action": best_action}
    return stat


def select_action(stat, epsilon):
    """
    to select a action
    :param stat: type:dict
    :param epsilon:
    :return:
    """
    rand_num = np.random.rand()
    if rand_num < epsilon:
        return np.random.choice(stat["action_index"])
    else:
        return np.argmax(stat["q_estimation"])


def calculate_reward(stat, action):
    """
    to find action-reward value
    :param stat: type:dict
    :param action:
    :return:
    """
    reward = stat["q_true"][action] + np.random.randn()
    stat["action_count"][action] += 1
    stat["q_estimation"][action] += (reward - stat["q_estimation"][action])/stat["action_count"][action]

    return reward


if __name__ == "__main__":
    runs = 2000
    times = 1000
    epsilons = [0, 0.01, 0.1]

    average_reward = np.zeros((len(epsilons), runs, times))
    optimal_action = np.zeros(average_reward.shape)
    print("Starting......")

    for i in range(len(epsilons)):
        for run in np.arange(runs):
            stat = initialize()
            for time in np.arange(times):
                action = select_action(stat, epsilons[i])
                average_reward[i, run, time] = calculate_reward(stat, action)
                if action == stat["best_action"]:
                    optimal_action[i, run, time] = 1
    print("Completed !!!")
    print("The true Q value is:\n", stat["q_true"])
    print("The estimation Q value is:\n", stat["q_estimation"])


    for eps, rewards in zip(epsilons, average_reward.mean(axis=1)):
        plt.plot(rewards, label=r"$\epsilon$ = %.2f" % (eps))
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("./images/figure2-2-1.png")
    plt.show()


    for eps, counts in zip(epsilons, optimal_action.mean(axis=1)):
        plt.plot(counts, label="$\epsilon$ = %.2f" % (eps))
    plt.xlabel("Steps")
    plt.ylabel("Optimal action percent")
    plt.legend()
    plt.savefig("./images/figure2-2-2.png")
    plt.show()

    plt.close()

