# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk(object):
    def __init__(self, num_states=1000, groups=10, alpha=2e-4):
        self.num_states = num_states                # the number of states
        self.groups = groups                        # the number of groups
        self.alpha = alpha                          # the size of step

        self.group_value = np.zeros(groups)                 # the value of each group
        self.group_size = int(num_states / groups)          # the size of each group

        self.states = np.arange(1, num_states + 1)          # all states except terminal state
        self.start_state = int(num_states / 2)              # the start state
        self.end_state = [0, num_states + 1]                # the terminal state
        self.action = [-1, 1]                       # -1:left, 1:right
        self.neighbors = 100                        # the neighboring states

    def select_action(self):
        """to randomly select an action"""
        if np.random.binomial(1, 0.5):
            return self.action[1]                   # take the right action
        else:
            return self.action[0]                   # take the left action

    def find_next_state(self, state, action):
        """to find the next state and get reward"""
        move_step = np.random.randint(1, self.neighbors+1)          # the step size of moving
        move_step *= action
        next_state = state + move_step              # the next state
        next_state = max(min(next_state, self.end_state[1]), 0)

        if next_state == self.end_state[0]:         # terminating on the left
            reward = -1
        elif next_state == self.end_state[1]:
            reward = 1                              # terminating on the right
        else:
            reward = 0
        return next_state, reward

    def get_state_value(self, state):
        """to get the value of current state"""
        if state in self.end_state:
            return 0
        group_idx = (state - 1) // self.group_size
        return self.group_value[group_idx]

    def update_group_value(self, state, delta):
        """to update the group's value according to the current state"""
        group_idx = (state - 1) // self.group_size
        self.group_value[group_idx] += delta

    def semi_gradient_td(self, n=1):
        """The semi-gradient n-step TD algorithm"""
        state = self.start_state                # initialize the state
        trajectory = [state]                    # track the transition state
        reward = []                             # the rewards of all states
        time = 0                                # track the time
        length = float("inf")                   # the length of the episode

        while True:
            if time < length:
                action = self.select_action()   # randomly select an action
                next_state, reward_ = self.find_next_state(state, action)
                # track the state and reward
                trajectory.append(next_state)
                reward.append(reward_)

                if next_state in self.end_state:
                    length = time + 1
            # get the time of the state to update
            update_time = time - n + 1
            if update_time >= 0:
                returns = 0
                # calculate the returns
                for idx in range(update_time+1, min(update_time+n, length)+1):
                    returns += reward[idx - 1]
                if update_time + n <= length:
                    returns += self.get_state_value(trajectory[update_time+n])
                # update the group value
                if trajectory[update_time] not in self.end_state:
                    delta = self.alpha*(returns - self.get_state_value(trajectory[update_time]))
                    self.update_group_value(state, delta)
            time += 1
            state = next_state
            if update_time == length - 1:
                break


def dp_compute_value(test_class):
    """using Dynamic programming to find the true state values"""
    value = np.arange(-test_class.end_state[1], test_class.end_state[1] + 1, 2) / test_class.end_state[1]
    print("Starting computing......")
    while True:
        value_temp = value.copy()
        for state in test_class.states:
            value[state] = 0
            for act in test_class.action:
                for step in range(1, test_class.neighbors + 1):
                    step *= act
                    next_state = state + step
                    next_state = max(min(next_state, test_class.end_state[1]), 0)
                    # update the value
                    value[state] += 1/(2*test_class.neighbors)*value[next_state]
        if np.linalg.norm(value - value_temp) < 0.001:
            break
    print("Completed!!!")
    return value


if __name__ == "__main__":
    episodes = 100000
    test_exam = RandomWalk()

    true_value = dp_compute_value(test_class=test_exam)
    for i in tqdm(range(episodes)):
        test_exam.semi_gradient_td()
    state_value = [test_exam.get_state_value(stat) for stat in test_exam.states]

    plt.figure(1)
    plt.plot(test_exam.states, true_value[1:-1], label="True value")
    plt.plot(test_exam.states, state_value, label="Approximate TD value")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("./images/Figure9-2.png")
    plt.show()
    plt.close()
    print("Completed!!!You can check it in 'images' directory")


