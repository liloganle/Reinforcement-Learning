# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class DynaMaze(object):
    def __init__(self, world_size=(6, 9), obstacle=[[0, 7], [1, 2], [1, 7], [2, 2], [2, 7], [3, 2], [4, 5]],
                 planning_steps=0):
        self.world_size = world_size        # the size of the maze world
        self.obstacle = obstacle            # the position of obstacle
        self.planning_steps = planning_steps  # the size of planning steps

        self.num_action = 4                 # the number of actions
        self.actions = np.arange(self.num_action)       # all possible actions, 0:up, 1:down, 2:left, 3:right
        self.start = np.array([2, 0])                   # the start state
        self.goal = np.array([0, 8])                    # the terminal state
        self.q_value = np.zeros((self.num_action, world_size[0], world_size[1]))        # the Q-value

        self.gama = 0.95        # the size of discounting, gama
        self.alpha = 0.1        # the size of step-size, alpha
        self.epsilon = 0.1      # the size of epsilon

    def select_action(self, q_value, state):
        """
        using epsilon-greedy to select an action under the current state
        :param q_value: the Q-value
        :param state: the current state
        :return:
        """
        random_num = np.random.rand()                   # generating an random number
        if random_num < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_value = q_value[:, state[0], state[1]]
            return np.random.choice([act for act, val in enumerate(state_value) if val == np.max(state_value)])

    def find_next_state(self, state, action):
        """
        to find the next state and return reward according to the current state and action
        :param state:the current state
        :param action:select an action according to the current state
        :return:next state and reward
        """
        x_value, y_value = state
        if action == self.actions[0]:                           # taking the up action
            x_value = max(x_value - 1, 0)
        elif action == self.actions[1]:                         # taking the down action
            x_value = min(x_value + 1, self.world_size[0] - 1)
        elif action == self.actions[2]:                         # taking the left action
            y_value = max(y_value - 1, 0)
        elif action == self.actions[3]:                         # taking the right action
            y_value = min(y_value + 1, self.world_size[1] - 1)

        next_state = np.array([x_value, y_value])
        if next_state.tolist() in self.obstacle:                # blocked by the obstacle
            next_state = state

        if all(next_state == self.goal):            # reach the destination
            reward = 1
        else:
            reward = 0

        return next_state, reward

    def tabular_dyna_q(self, q_value, model):
        """
        the Tabular Dyna-Q algorithm
        :param q_value:
        :param model: the model to store the previous experience
        :return:
        """
        state = self.start      # the start state
        steps = 0               # the number of steps

        while any(state != self.goal):              # !!!!!!
            steps += 1  # tracking the number of steps per episode
            action = self.select_action(q_value, state)         # to select an action according to the state
            next_state, reward = self.find_next_state(state, action)        # to find the next state and reward
            # Q-learning algorithm to update the state-action value
            max_value = np.max(q_value[:, next_state[0], next_state[1]])
            q_value[action, state[0], state[1]] += self.alpha*(reward + self.gama*max_value -
                                                               q_value[action, state[0], state[1]])
            model.building(state, action, next_state, reward)           # building the model
            # sampling from the model
            for num in range(self.planning_steps):
                sta, act, next_sta, rew = model.sampling()               # sampling from model
                # Q-learning algorithm to update the state-action value
                max_val = np.max(q_value[:, next_sta[0], next_sta[1]])
                q_value[act, sta[0], sta[1]] += self.alpha*(rew + self.gama*max_val -
                                                            q_value[act, sta[0], sta[1]])
            state = next_state
        return steps


class Model(object):
    def __init__(self):
        self.model = {}

    def building(self, state, action, next_state, reward):
        """
        building the model according to the previous experiences
        :param state: the current state
        :param action: the selected action under the current state
        :param next_state:
        :param reward:
        :return:
        """
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = {}
        self.model[tuple(state)][action] = [tuple(next_state), reward]

    def sampling(self):
        """
        random previously observed state and action
        :return:
        """
        state_index = np.random.choice(range(len(self.model.keys())))
        current_state = list(self.model)[state_index]
        action_index = np.random.choice(range(len(self.model[current_state].keys())))
        action = list(self.model[current_state])[action_index]
        next_state, reward = self.model[current_state][action]

        return np.array(current_state), action, np.array(next_state), reward


if __name__ == "__main__":
    runs = 20       # the number of runs
    episodes = 50   # the number of episodes

    plan_steps = [0, 5, 50]
    original_model = Model()
    all_runs_steps = np.zeros((len(plan_steps), episodes))

    for run in tqdm(range(runs)):
        for idx, planning in enumerate(plan_steps):
            dyna_maze = DynaMaze(planning_steps=planning)
            q_values = dyna_maze.q_value.copy()
            for ep in range(episodes):
                all_runs_steps[idx, ep] += dyna_maze.tabular_dyna_q(q_values, original_model)
    all_runs_steps /= runs
    # print(q_values)

    plt.figure(1)
    for ith in range(len(plan_steps)):
        plt.plot(all_runs_steps[ith], label=r"%d planning steps" % plan_steps[ith])
    plt.xlabel("Episodes")
    plt.ylabel("Steps per Episode")
    plt.legend()
    plt.savefig("./images/Figure8-2.png")
    plt.show()
    plt.close()
    print("Completed!!! You can check it in the 'images' directory")

