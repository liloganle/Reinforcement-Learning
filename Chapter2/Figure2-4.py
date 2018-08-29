# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Ten_Armed_Bandit(object):
    def __init__(self, k_armed=10, epsilon=0, sample_average=False, ucb_degree=0, ucb_flag=False):
        self.k = k_armed
        self.epsilon = epsilon
        self.degree = ucb_degree
        self.time = 0
        self.sample_average = sample_average
        self.ucb_flag = ucb_flag

    def init_parameter(self):
        self.q_true = np.random.randn(self.k)
        self.q_estimate = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.action_index = np.arange(self.k)

    def select_action(self):
        if self.sample_average:
            rand_num = np.random.rand()
            if rand_num < self.epsilon:
                return np.random.choice(self.action_index)
            else:
                return np.argmax(self.q_estimate)
        if self.ucb_flag:
            temp = self.q_estimate + self.degree*np.sqrt(np.log(self.time + 1)/(self.action_count+1e-6))
            temp_max = np.max(temp)
            return np.random.choice([action for action, value in enumerate(temp) if value == temp_max])


    def back_reward(self, action):
        reward = self.q_true[action] + np.random.randn()
        self.time += 1
        self.action_count[action] += 1
        self.q_estimate[action] += (reward - self.q_estimate[action])/self.action_count[action]

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
    problems.append(Ten_Armed_Bandit(epsilon=0.1, sample_average=True))
    problems.append(Ten_Armed_Bandit(ucb_degree=2, ucb_flag=True))

    average_reward, _ = update(problems)
    plt.plot(average_reward[0], label=r"$\epsilon$-greedy,$\epsilon$=0.1")
    plt.plot(average_reward[1], label="UCB, c=2")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("./images/figure2-4.png")
    plt.show()

    plt.close()

