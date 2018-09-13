# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Windy_Grid_World(object):
    def __init__(self, epsilon=0.1, step_size=0.5, discount=1):
        self.world_size = (7, 10)    # first number is row, second number is column
        self.num_action = 4     # the number of actions
        self.actions = np.arange(self.num_action)    # 0:up, 1:down, 2:left, 3:right
        self.epsilon = epsilon    # epsilon-greedy
        self.step_size = step_size    # step size for Sarsa
        self.discount = discount

        self.start = np.array([3, 0])
        self.end = np.array([3, 7])
        self.reward = -1
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])    # the strength of wind for every column

    def epsilon_greedy(self, q_value, state):
        rand_num = np.random.randn()
        if rand_num < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_value = q_value[:, state[0], state[1]]
            return np.random.choice([act for act, val in enumerate(state_value) if val == np.max(state_value)])

    def find_next_state(self, current_state, current_action):
        row, col = current_state
        # take the up action
        if current_action == self.actions[0]:
            temp = max(row - 1 - self.wind[col], 0)
            return np.array([temp, col])
        # take the down action
        elif current_action == self.actions[1]:
            temp = min(row + 1 - self.wind[col], self.world_size[0]-1)
            temp = max(temp, 0)
            return np.array([temp, col])
        # take the left action
        elif current_action == self.actions[2]:
            temp = max(row - self.wind[col], 0)
            return np.array([temp, max(col-1, 0)])
        # take the right action
        else:
            temp = min(col + 1, self.world_size[1] - 1)
            return np.array([max(row - self.wind[col], 0), temp])

    def sarsa_algorithm(self, q_value):
        state = self.start
        time_step = 0   # to track the time steps during the episode

        # to use epsilon-greedy algorithm to select action
        action = self.epsilon_greedy(q_value, state)
        while any(state != self.end):
            # according to current state and action to find next state
            next_state = self.find_next_state(state, action)
            next_action = self.epsilon_greedy(q_value, next_state)

            # to update the action value
            q_value[action, state[0], state[1]] += \
                self.step_size*(self.reward + self.discount*q_value[next_action, next_state[0], next_state[1]] -
                                q_value[action, state[0], state[1]])
            state = next_state
            action = next_action
            time_step += 1
        return time_step

if __name__ == "__main__":
    windy_grid_word = Windy_Grid_World()
    q_value_fun = np.zeros((windy_grid_word.num_action, windy_grid_word.world_size[0],
                            windy_grid_word.world_size[1]))
    '''
    stat = np.array([3, 0])
    time = 200
    actions = np.zeros(time)
    for idx in range(time):
        actions[idx] += windy_grid_word.epsilon_greedy(q_value_fun, stat)
    print(actions)

    '''
    episode = 170
    time_steps = np.zeros(episode)
    for idx in range(episode):
        time_steps[idx] += windy_grid_word.sarsa_algorithm(q_value_fun)
    time_steps = np.add.accumulate(time_steps)

    average_episode = time_steps[-1]/episode
    print("the average episode length at about %.f steps" % average_episode)

    plt.figure(1)
    plt.plot(time_steps, np.arange(episode))
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.savefig("./images/Example6-5.png")
    plt.show()
    plt.close()
    print("Completed!!! You can check it in the 'images' directory")


