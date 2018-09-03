# -*- coding:utf-8 -*-

import numpy as np

class Grid_World(object):
    def __init__(self, size=4, alpha=1):
        self.size = size
        self.alpha = alpha
        self.actions = [np.array([-1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, -1])]

    def is_terminal_state(self, state):
        x_axis, y_axis = state
        return (x_axis == 0 and y_axis == 0) or (x_axis == self.size-1 and y_axis == self.size-1)

    def state_reward(self, state, action):
        reward = -1
        next_state = state + action
        x_axis, y_axis = next_state
        if x_axis < 0 or x_axis >= self.size or y_axis < 0 or y_axis >= self.size:
            next_state = state

        return next_state, reward

    def value_function(self, in_place=False):
        value = np.zeros((self.size, self.size))
        next_value = np.zeros((self.size, self.size))

        proba = 1/len(self.actions)
        iteration = 1

        while True:
            temp_value = next_value if in_place else value
            for row in np.arange(self.size):
                for col in np.arange(self.size):
                    state = np.array([row, col])
                    if self.is_terminal_state(state):
                        continue
                    temp = 0
                    for act in self.actions:
                        (x_next, y_next), reward = self.state_reward(state, act)
                        temp += proba*(reward + self.alpha*temp_value[x_next, y_next])
                    next_value[row, col] = temp

            if np.linalg.norm(next_value - value) < 1e-4:
                break

            value = next_value.copy()
            iteration += 1

        return next_value, iteration


if __name__ == "__main__":
    grid_world = Grid_World()
    state_value, iteration = grid_world.value_function(in_place=False)
    in_place_value, in_place_iteration = grid_world.value_function(in_place=True)

    print("The number of iteration with the way of two arrays is: ", iteration)
    print("The final state value with the way of two arrays is: \n", np.round(state_value, decimals=1))
    print("-------------------------------------------------------")
    print("The number of iteration with the way of in_place is: ", in_place_iteration)
    print("The final state value with the way of in_place is: \n", np.round(in_place_value, decimals=1))





