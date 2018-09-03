# -*- coding:utf-8 -*-

import numpy as np

class Grid_World(object):
    def __init__(self, size=5, gama=0.9):
        self.size = size
        self.gama = gama
        self.a_start = np.array([0, 1])
        self.a_end = np.array([4, 1])
        self.b_start = np.array([0, 3])
        self.b_end = np.array([2, 3])
        self.actions = [np.array([-1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, -1])]

    def state_reward(self, state, action):
        if all(state == self.a_start):
            return self.a_end, 10
        if all(state == self.b_start):
            return self.b_end, 5

        next_state = state + action
        x_axis, y_axis = next_state
        if x_axis < 0 or x_axis > 4 or y_axis < 0 or y_axis > 4:
            reward = -1
            next_state = state
        else:
            reward = 0

        return next_state, reward

    def optimal_value(self):
        value = np.zeros((self.size, self.size))

        #print("The initial state value function is:\n", value)
        while True:
            next_value = np.zeros(value.shape)
            for row in np.arange(self.size):
                for col in np.arange(self.size):
                    temp = []
                    for act in self.actions:
                        state = np.array([row, col])
                        (next_x, next_y), reward = self.state_reward(state, act)
                        temp.append(reward + self.gama*value[next_x, next_y])  #Bellman Optimality Equation
                    next_value[row, col] = max(temp)
            if np.linalg.norm(next_value - value) < 1e-5:
                break
            value = next_value

        return next_value


if __name__ == "__main__":
    grid_world = Grid_World()
    values = grid_world.optimal_value()

    print("The optimal state value function is: \n", np.array_str(values, precision=1, suppress_small=True))

