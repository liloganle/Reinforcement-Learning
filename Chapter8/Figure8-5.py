# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class DynaMaze(object):
    def __init__(self, world_size=(6, 9), obstacle=[[0, 7], [1, 2], [1, 7], [2, 2], [2, 7], [3, 2], [4, 5]],
                 planning_steps=0):
        self.world_size = world_size        # the size of the maze world
        self.obstacle = obstacle
        self.cur_obstacle = [[3, row] for row in range(1, world_size[1])]
        self.new_obstacle = [[3, row] for row in range(1, world_size[1] - 1)]
        self.planning_steps = planning_steps  # the size of planning steps

        self.num_action = 4                 # the number of actions
        self.actions = np.arange(self.num_action)       # all possible actions, 0:up, 1:down, 2:left, 3:right
        self.start = np.array([5, 3])                   # the start state
        self.goal = np.array([0, 8])                    # the terminal state
        self.q_value = np.zeros((self.num_action, world_size[0], world_size[1]))        # the Q-value

        self.gama = 0.95        # the size of discounting, gama
        self.alpha = 1          # the size of step-size, alpha
        self.epsilon = 0.1      # the size of epsilon
        self.time_steps = 6000
        self.changing_step = 3000

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
        :param q_value: action value
        :param model: original model or enhanced model
        :return:
        """
        state = self.start      # the start state
        steps = 0               # the number of steps
        # model = OriginalModel()     # the model to store the previous experience

        while any(state != self.goal):              # !!!!!!
            steps += 1                  # tracking the number of steps per episode
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


class OriginalModel(object):
    def __init__(self):
        self.model = {}

    def building(self, state, action, next_state, reward):
        """
        building the model according to the previous experiences
        :param state: the current state
        :param action: the selected action under the current state
        :param next_state: the current state's successor
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


class EnhancedModel(object):
    def __init__(self):
        self.model = {}
        self.elapsed_time = 0      # tracking the elapsed time steps
        self.weight = 1e-3

    def building(self, state, action, next_state, reward):
        self.elapsed_time += 1
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = {}
            # actions that have never been selected before should been considered
            for act in range(4):                # sum of actions is four
                if act != action:
                    self.model[tuple(state)][act] = [tuple(state), 0, 1]
        self.model[tuple(state)][action] = [tuple(next_state), reward, self.elapsed_time]

    def sampling(self):
        state_index = np.random.choice(range(len(self.model.keys())))
        current_state = list(self.model)[state_index]
        action_index = np.random.choice(range(len(self.model[current_state].keys())))
        action = list(self.model[current_state])[action_index]
        next_state, reward, time = self.model[current_state][action]
        # special "bonus reward"
        reward += self.weight*np.sqrt(self.elapsed_time - time)
        return np.array(current_state), action, np.array(next_state), reward


def cumulative_reward(runs, dyna_maze, methods):
    rewards = np.zeros((len(methods), runs, dyna_maze.time_steps))

    for ith, method in enumerate(methods):
        for run in tqdm(range(runs)):
            q_value = np.zeros((dyna_maze.num_action, dyna_maze.world_size[0], dyna_maze.world_size[1]))
            dyna_maze.obstacle = dyna_maze.cur_obstacle

            steps = 0
            last_steps = steps
            while steps < dyna_maze.time_steps:
                steps += dyna_maze.tabular_dyna_q(q_value, method)      # playing for an episodes
                # updating cumulative rewards
                rewards[ith, run, last_steps: steps] = rewards[ith, run, last_steps]
                rewards[ith, run, min(steps, dyna_maze.time_steps - 1)] = rewards[ith, run, last_steps] + 1
                last_steps = steps
                # changing the obstacles
                if steps > dyna_maze.changing_step:
                    dyna_maze.obstacle = dyna_maze.new_obstacle

    return rewards.mean(axis=1)


if __name__ == "__main__":
    num_runs = 5
    blocking_maze = DynaMaze(planning_steps=15)
    two_models = [OriginalModel(), EnhancedModel()]
    title_label = ["Dyna-Q", "Dyna-Q+"]

    cumulative_rewards = cumulative_reward(runs=num_runs, dyna_maze=blocking_maze, methods=two_models)

    plt.figure(1)
    for idx in range(len(title_label)):
        plt.plot(cumulative_rewards[idx], label=title_label[idx])
    plt.xlabel(r"Time Steps")
    plt.ylabel(r"Cumulative Reward")
    plt.legend()
    plt.savefig("./images/Figure8-5.png")
    plt.show()
    plt.show()
    plt.close()
    print("Completed!!! You can check it in the 'images' directory")
