# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk(object):
    def __init__(self, num_states=1000, groups=10, alpha=2e-5):
        self.num_states = num_states            # the number of states
        self.groups = groups                    # the number of groups
        self.alpha = alpha                      # the step size

        self.group_value = np.zeros(groups)                # the value of each group
        self.group_size = int(num_states / groups)          # the size of each group

        self.states = np.arange(1, num_states+1)        # all states except terminal state
        self.start_state = int(num_states / 2)          # the start state
        self.end_state = [0, num_states + 1]            # the terminal states
        self.action = [-1, 1]                           # right:1, left:-1
        self.neighbors = 100                            # the neighboring states

    def select_action(self):
        """to select randomly an action"""
        if np.random.binomial(1, 0.5):
            return self.action[1]           # select right action
        else:
            return self.action[0]           # select left action

    def find_next_state(self, state, action):
        """to get the next state and reward"""
        move_step = np.random.randint(1, self.neighbors+1)          # the step size of moving
        move_step *= action
        next_state = state + move_step              # the next state
        next_state = max(min(next_state, self.end_state[1]), 0)

        if next_state == self.end_state[0]:                     # terminating on the left
            reward = -1
        elif next_state == self.end_state[1]:                   # terminating on the right
            reward = 1
        else:
            reward = 0
        return next_state, reward

    def get_state_value(self, state):
        """to get the state value except for terminal states"""
        group_idx = (state - 1) // self.group_size
        return self.group_value[group_idx]
    
    def update_group_value(self, state, delta):
        """to update the group_value"""
        group_idx = (state - 1) // self.group_size
        self.group_value[group_idx] += delta

    def gradient_monte_carlo(self, state_distribution):
        """ the gradient-descent version of Monte Carlo state-value prediction"""
        state = self.start_state            # initialize the state
        trajectory = [state]                # track the transition state

        while state not in self.end_state:
            action = self.select_action()       # select an action
            next_state, reward = self.find_next_state(state, action)        # get the next state and reward
            trajectory.append(next_state)       # record the transition state
            state = next_state

        for stat in trajectory[:-1]:
            delta = self.alpha * (reward - self.get_state_value(stat))
            self.update_group_value(stat, delta)
            state_distribution[stat] += 1


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
    distribution = np.zeros(test_exam.num_states + len(test_exam.end_state))
    for itr in tqdm(range(episodes)):
        test_exam.gradient_monte_carlo(distribution)

    distribution /= np.sum(distribution)
    state_value = [test_exam.get_state_value(stat) for stat in test_exam.states]

    plt.figure(1)
    plt.plot(test_exam.states, true_value[1:-1], label="True value")
    plt.plot(test_exam.states, state_value, label="Approximate MC value")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("./images/Figure9-1-1.png")
    plt.show()

    plt.figure(2)
    plt.plot(test_exam.states, distribution[1:-1], label="State Distribution")
    plt.xlabel("State")
    plt.ylabel("Distribution")
    plt.legend()
    plt.savefig("./images/Figure9-1-2.png")
    plt.show()

    plt.close()
    print("Completed!!!You can check it in 'images' directory")


