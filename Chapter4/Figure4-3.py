# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Gambler_Problem(object):
    def __init__(self, goal=100, up_proba=0.4):
        self.goal = goal
        self.up_proba = up_proba
        self.states = np.arange(goal + 1)   #including state 0 and 100
        self.state_values = np.zeros(goal + 1)
        self.policy = np.zeros(goal + 1)

    def value_function(self):
        self.state_values[self.goal] = 1

        while True:
            flag = 0
            for state in self.states[1:self.goal]:
                state_actions = np.arange(min(state, self.goal - state) + 1)
                action_return = []
                for action in state_actions:
                    reward = self.up_proba*self.state_values[state + action] + (1 - self.up_proba) * \
                             self.state_values[state - action]
                    action_return.append(reward)
                flag += np.abs(self.state_values[state] - np.max(action_return))
                self.state_values[state] = np.max(action_return)  #update the state value

            if flag < 1e-6:
                break

    def optimal_policy(self):
        for state in self.states[1:self.goal]:
            state_actions = np.arange(min(state, self.goal - state) + 1)
            action_return = []
            for action in state_actions:
                reward = self.up_proba*self.state_values[state + action] + (1 - self.up_proba) * \
                         self.state_values[state - action]
                action_return.append(reward)

            self.policy[state] = state_actions[np.argmax(np.round(action_return[1:], 5))]  #update the policy


if __name__ == "__main__":
    gambler_1 = Gambler_Problem()
    gambler_1.value_function()
    gambler_1.optimal_policy()

    plt.figure()
    plt.plot(gambler_1.state_values)
    plt.xlabel("Capital")
    plt.ylabel("Value Estimates")
    plt.savefig("./images/Figure4-3-1.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.step(gambler_1.states, gambler_1.policy)
    plt.xlabel("Capital")
    plt.ylabel("Final Policy")
    plt.savefig("./images/Figure4-3-2.png")
    plt.show()



