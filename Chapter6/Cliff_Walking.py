# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Cliff_Walk(object):
    def __init__(self, epsilon=0.1, step_size=0.5, discount=1):
        self.walking_range = (4, 12)   # the range of walking
        self.action_num = 4    # the number of actions
        self.actions = np.arange(self.action_num)    # up:0, down:1, left:2, right:3

        self.epsilon = epsilon
        self.step_size = step_size
        self.discount = discount

        self.start = np.array([3, 0])
        self.end = np.array([3, 11])

    def epsilon_greedy(self, q_value, state):
        rand_num = np.random.rand()
        if rand_num < self.epsilon:
            return np.random.choice(self.actions)
        else:
            value = q_value[:, state[0], state[1]]
            return np.random.choice([act for act, val in enumerate(value) if val == np.max(value)])

    def find_next_state(self, state, action):
        row, col = state
        reward = -1
        # take the up action
        if action == self.actions[0]:
            temp = max(row - 1, 0)
            next_state = np.array([temp, col])
        # take the down action
        elif action == self.actions[1]:
            temp = min(row + 1, self.walking_range[0] - 1)
            next_state = np.array([temp, col])
        # take the left action
        elif action == self.actions[2]:
            temp = max(col - 1, 0)
            next_state = np.array([row, temp])
        # take the right action
        elif action == self.actions[3]:
            temp = min(col + 1, self.walking_range[1]-1)
            next_state = np.array([row, temp])

        if next_state[0] == self.walking_range[0]-1 and 0 < next_state[1] < self.walking_range[1]-1:
            next_state = self.start
            reward = -100

        return next_state, reward

    def sarsa_algorithm(self, q_value):
        state = self.start    # the initial state
        action = self.epsilon_greedy(q_value, state)    # to choose an action
        sum_reward = 0.0
        # to update until reach the destination
        while any(state != self.end):
            # according to current state and action to find the next state
            next_state, reward = self.find_next_state(state, action)
            # according to the next state to select the next action
            next_action = self.epsilon_greedy(q_value, next_state)
            sum_reward += reward
            # using Sarsa equation to update the action value
            q_value[action, state[0], state[1]] += \
                self.step_size*(reward + self.discount*q_value[next_action, next_state[0], next_state[1]] - 
                                q_value[action, state[0], state[1]])
            state = next_state
            action = next_action

        return sum_reward

    def q_learning_algorithm(self, q_value):
        state = self.start
        sum_reward = 0
        # to update until reach the destination
        while any(state != self.end):
            # according to the current state to choose an action
            action = self.epsilon_greedy(q_value, state)
            # to find the next state and reward
            next_state, reward = self.find_next_state(state, action)
            sum_reward += reward

            # using Q-learning equation to update the action value
            max_q_value = np.max(q_value[:, next_state[0], next_state[1]])
            q_value[action, state[0], state[1]] += self.step_size*(reward + self.discount*max_q_value -
                                                                   q_value[action, state[0], state[1]])
            state = next_state

        return sum_reward

if __name__ == "__main__":
    episodes = 500
    runs = 100

    cliff_walking = Cliff_Walk()
    sum_reward_sarsa = np.zeros(episodes)
    sum_reward_q_learning = np.zeros(episodes)

    for run in tqdm(range(runs)):
        q_value_sarsa = np.zeros((cliff_walking.action_num, cliff_walking.walking_range[0],
                                  cliff_walking.walking_range[1]))
        q_value_q_learning = np.zeros(q_value_sarsa.shape)
        for idx in range(episodes):
            sum_reward_sarsa[idx] += cliff_walking.sarsa_algorithm(q_value_sarsa)
            sum_reward_q_learning[idx] += cliff_walking.q_learning_algorithm(q_value_q_learning)

    # to average the runs
    # print("The sum of rewards during episode: \n", sum_reward_sarsa)
    # print("The average sum of rewards:\n", sum_reward_sarsa/runs)
    # sum_reward_q_learning /= runs


    # to plot the figure
    plt.figure(1)
    plt.plot(sum_reward_sarsa/runs, label=r"Sarsa")
    plt.plot(sum_reward_q_learning/runs, label=r"Q-learning")
    plt.ylim([-250, 0])
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.legend()
    plt.savefig("./images/Example6-6.png")
    plt.show()
    print("Completed!!! You can check it in the 'images' directory")

        
        





