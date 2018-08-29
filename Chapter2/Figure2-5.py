# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Ten_Armed_Bandit(object):
    def __init__(self, k_armed=10, alpha=0, initial_reward=4, baseline_flag=False):
        self.k = k_armed
        self.alpha = alpha
        self.initial_reward = initial_reward
        self.baseline_flag = baseline_flag
        self.time = 0
        self.average_reward = 0

    def init_parameter(self):
        self.q_true = np.random.randn(self.k) + self.initial_reward
        self.q_estimate = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.action_index = np.arange(self.k)

    def select_action(self):
        h_value = np.exp(self.q_estimate)
        self.action_prob = h_value / np.sum(h_value)
        return np.random.choice(self.action_index, p=self.action_prob)


    def back_reward(self, action):
        reward = self.q_true[action] + np.random.randn()
        self.time += 1
        self.average_reward += (reward - self.average_reward)/self.time
        self.action_count[action] += 1

        ones = np.zeros(self.k)
        ones[action] = 1
        if self.baseline_flag:
            baseline = self.average_reward
        else:
            baseline = 0
        self.q_estimate += self.alpha*(reward - baseline)*(ones - self.action_prob)

        return reward

def update(problems, runs=2000, steps=1000):
    optimal_action = np.zeros((len(problems), runs, steps))
    average_reward = np.zeros(optimal_action.shape)

    print("Starting......")
    for i, problem in enumerate(problems):
        for run in np.arange(runs):
            problem.init_parameter()
            for step in np.arange(steps):
                action = problem.select_action()
                reward = problem.back_reward(action)
                average_reward[i, run, step] = reward
                if action == problem.best_action:
                    optimal_action[i, run, step] = 1
    print("Completed !!!")
    return average_reward.mean(axis=1), optimal_action.mean(axis=1)

if __name__ == "__main__":
    problems = []
    problems.append(Ten_Armed_Bandit(alpha=0.1, baseline_flag=True))
    problems.append(Ten_Armed_Bandit(alpha=0.4, baseline_flag=True))
    problems.append(Ten_Armed_Bandit(alpha=0.1, baseline_flag=False))
    problems.append(Ten_Armed_Bandit(alpha=0.4, baseline_flag=False))

    _, optimal_action = update(problems)
    labels = [r"$\alpha$=0.1 with baseline", r"$\alpha$=0.4 with baseline",
              r"$\alpha$=0.1 without baseline", r"$\alpha$=0.4 without baseline"]

    for iter in np.arange(len(labels)):
        plt.plot(optimal_action[iter], label=labels[iter])
    plt.xlabel("Steps")
    plt.ylabel("Optimal action percent")
    plt.legend()
    plt.savefig("./images/figure2-5.png")
    plt.show()

    plt.close()

