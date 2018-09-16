# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

class Bias_Example(object):
    def __init__(self, epsilon=0.1, alpha=0.1, gama=1):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gama = gama

        self.num_states = 3   # the number of state: two non-terminal state and a terminal state
        self.states = np.arange(self.num_states)   # A:0, B:1, terminal states:2
        self.state_a_actions = np.array([0, 1])    # right:0, left:1
        self.state_b_actions = np.arange(5)       # maybe 10 actions
        self.q_value = [np.zeros(len(self.state_a_actions)), np.zeros(len(self.state_b_actions)), np.zeros(1)]

    def select_action(self, q_value, state):
        all_actions = [self.state_a_actions, self.state_b_actions]
        rand_num = np.random.rand()
        # epsilon greedy policy to choose an action
        if rand_num < self.epsilon:
            return np.random.choice(all_actions[state])
        else:
            current_q_value = q_value[state]
            return np.random.choice([act for act, val in enumerate(current_q_value) if val == np.max(current_q_value)])

    def find_next_state(self, state, action):
        # if agent is in state A, reward is 0
        if state == self.states[0]:
            if action == self.state_a_actions[1]:
                return self.states[1], 0
            else:
                return self.states[2], 0
        # if agent is in state B
        if state == self.states[1]:
            return self.states[2], np.random.normal(-0.1, 1)

    def q_learning(self, q_value):
        state = self.states[0]    # the initial state is A
        count_left_action = 0    # count the number of taking left action in A state

        while state != self.states[2]:
            action = self.select_action(q_value, state)
            if state == self.states[0] and action == self.state_a_actions[1]:
                count_left_action += 1
            # according to current state and action to find the next state
            next_state, reward = self.find_next_state(state, action)
            # if next_state == self.states[1]:
            #    count_left_action += 1
            # using Q-learning equation to update the q-value
            max_q_value = np.max(q_value[next_state])
            q_value[state][action] += self.alpha * (reward + self.gama * max_q_value - q_value[state][action])

            state = next_state
        return count_left_action

    def double_q_learning(self, q_value_1, q_value_2):
        state = self.states[0]    # the initial state is A
        count_left_action = 0

        while state != self.states[2]:
            action = self.select_action([q1+q2 for q1, q2 in zip(q_value_1, q_value_2)], state)
            # according to current state and action to find the next state
            next_state, reward = self.find_next_state(state, action)
            if next_state == self.states[1]:
                count_left_action += 1
            # double learning
            if np.random.binomial(1, 0.5):
                update_value = q_value_1
                max_value = q_value_2
            else:
                update_value = q_value_2
                max_value = q_value_1
            optimal_action = np.argmax(update_value[next_state])
            update_value[state][action] += self.alpha*(reward + self.gama*max_value[next_state][optimal_action] -
                                                       update_value[state][action])
            state = next_state
        return count_left_action


if __name__ == "__main__":
    example_class = Bias_Example()

    episodes = 500
    runs = 1000
    q_learning_counts = np.zeros((runs, episodes))
    double_learning_counts = np.zeros(q_learning_counts.shape)
    for row in tqdm(range(runs)):
        q_learning_value = copy.deepcopy(example_class.q_value)
        double_learning_value_1 = copy.deepcopy(q_learning_value)
        double_learning_value_2 = copy.deepcopy(q_learning_value)
        for col in range(episodes):
            q_learning_counts[row, col] += example_class.q_learning(q_value=q_learning_value)
            double_learning_counts[row, col] += example_class.double_q_learning(double_learning_value_1,
                                                                                double_learning_value_2)
    # print(counts)
    q_learning_counts = np.add.accumulate(q_learning_counts, axis=1).mean(axis=0)/np.arange(1, episodes+1)
    double_learning_counts = np.add.accumulate(double_learning_counts, axis=1).mean(axis=0)/np.arange(1, episodes+1)

    plt.figure(1)
    plt.plot(q_learning_counts, label='Q-Learning')
    plt.plot(double_learning_counts, label='Double Q-Learning')
    plt.plot(np.ones(episodes) * 0.05, "k--", label='Optimal')
    plt.xlim([0, 500])
    plt.xlabel('episodes')
    plt.ylabel('% left actions from A')
    plt.legend()
    plt.savefig("./images/Figure6-5.png")
    plt.show()
    plt.close()
    print("Completed!!! You can check it in the 'images' directory")


