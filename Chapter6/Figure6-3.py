# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Cliff_Walking(object):
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
            reward = -10
        return next_state, reward

    def sarsa_algorithm(self, q_value):
        state = self.start    # the initial state
        action = self.epsilon_greedy(q_value, state)    # to choose an action
        sum_reward = 0
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

    def expected_sarsa(self, q_value):
        state = self.start
        action = self.epsilon_greedy(q_value, state)
        sum_reward = 0

        # to update until reach the destination
        while any(state != self.end):
            next_state, reward = self.find_next_state(state, action)
            next_action = self.epsilon_greedy(q_value, next_state)
            sum_reward += reward

            # calculate the expected value
            expected_value = 0
            next_state_value = q_value[:, next_state[0], next_state[1]]
            optimal_action = np.argwhere(next_state_value == np.max(next_state_value))    # to find the max action
            for act in self.actions:
                if act in optimal_action:
                    expected_value += (self.epsilon/self.action_num + (1-self.epsilon)/len(optimal_action)) *\
                                      q_value[act, next_state[0], next_state[1]]
                else:
                    expected_value += self.epsilon/self.action_num*q_value[act, next_state[0], next_state[1]]
            # using the expected sarsa equation to update the action value
            q_value[action, state[0], state[1]] += self.step_size*(reward + self.discount*expected_value -
                                                                   q_value[action, state[0], state[1]])
            state = next_state
            action = next_action
        return sum_reward

if __name__ == "__main__":
    step_sizes = np.arange(0.1, 1.05, 0.05)
    number = 3
    ex_sarsa = 0
    sarsa = 1
    q_learning = 2

    runs = 10
    episodes = 1000

    asymptotic_performance = np.zeros((number, len(step_sizes)))
    interim_performance = np.zeros((number, len(step_sizes)))
    # q_learning_performance = np.zeros(sarsa_performance.shape)

    for run in tqdm(range(runs)):
        for idx, alpha in enumerate(step_sizes):
            example = Cliff_Walking(step_size=alpha)
            expected_sarsa_q_value = np.zeros((example.action_num, example.walking_range[0], example.walking_range[1]))
            sarsa_q_value = np.zeros(expected_sarsa_q_value.shape)
            q_learning_q_value = np.zeros(expected_sarsa_q_value.shape)
            for count in range(episodes):
                expected_sarsa_reward = example.expected_sarsa(q_value=expected_sarsa_q_value)
                sarsa_reward = example.sarsa_algorithm(q_value=sarsa_q_value)
                q_learning_reward = example.q_learning_algorithm(q_value=q_learning_q_value)
                asymptotic_performance[ex_sarsa, idx] += expected_sarsa_reward
                asymptotic_performance[sarsa, idx] += sarsa_reward
                asymptotic_performance[q_learning, idx] += q_learning_reward

                # to count interim performance
                if count < 100:
                    interim_performance[ex_sarsa, idx] += expected_sarsa_reward
                    interim_performance[sarsa, idx] += sarsa_reward
                    interim_performance[q_learning, idx] += q_learning_reward

    np.savetxt("asymptotic_performance.txt", asymptotic_performance)
    np.savetxt("interim_performance.txt", interim_performance)
    # asymptotic_performance = asymptotic_performance / (runs*episodes)
    # interim_performance = interim_performance / 100
    asy_labels = [r"Asymptotic Expected Sarsa", r"Asymptotic Sarsa", r"Asymptotic Q-Learning"]
    asy_style = [r"x-", r"v-", r"s-"]
    inter_labels = [r"Interim Expected Sarsa", r"Interim Sarsa", r"Interim Q-Learning"]
    inter_style = [r"x--", r"v--", r"s--"]

    plt.figure(1)
    for row in range(number):
        plt.plot(step_sizes, asymptotic_performance[row, :]/(runs*episodes), asy_style[row], label=asy_labels[row])
        plt.plot(step_sizes, interim_performance[row, :]/(100*runs), inter_style[row], label=inter_labels[row])
    plt.xlim([0.1, 1])

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"Sum of Rewards per episode")
    plt.legend()
    plt.savefig("./images/Figure6-3.png")
    plt.show()
    plt.close()
    print("Completed!!! You can check it in the 'images' directory")








