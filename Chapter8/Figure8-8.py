# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def greedy_algorithm(q_value):
    """
    the greedy algorithm
    :param q_value: state value or action value
    :return:
    """
    max_value = np.max(q_value)
    return np.random.choice([act for act, val in enumerate(q_value) if val == max_value])


class TrajectorySampling(object):
    def __init__(self, num_states, branches):
        self.num_states = num_states
        self.branches = branches

        self.actions = [0, 1]       # two actions for every states
        self.epsilon = 0.1          # epsilon-greedy algorithm
        self.terminal_pro = 0.1     # the probability of transition to the terminal state
        self.discount = 1           #
        self.iterations = 20000     # the number of iterations

        # the transition probability matrix:(current state, action, next state)
        self.transition = np.random.randint(num_states, size=(num_states, len(self.actions), branches))
        # reward:(current state, action, next state)
        self.reward = np.random.randn(num_states, len(self.actions), branches)

    def epsilon_greedy(self, q_value):
        """
        the epsilon-greedy algorithm
        :param q_value:state value or action value
        :return:
        """
        random_num = np.random.rand()
        if random_num < self.epsilon:
            return np.random.choice(self.actions)
        else:
            max_value = np.max(q_value)
            return np.random.choice([act for act, val in enumerate(q_value) if val == max_value])

    def find_next_state(self, state, action):
        """
        to find the next state
        :param state: the current state
        :param action: taking action in the current state
        :return: next state and reward
        """
        random_num = np.random.rand()
        if random_num < self.epsilon:
            return self.num_states, 0
        next_state = np.random.randint(self.branches)
        return self.transition[state, action, next_state], self.reward[state, action, next_state]

    def compute_value(self, q_value):
        """
        using Monte Carlo method to compute the state value or action value under greedy policy
        :param q_value: the state value or action value
        :return:
        """
        runs = 1000
        returns = np.zeros(runs)
        for run in range(runs):
            state = 0
            reward = 0
            time = 0
            while state < self.num_states:
                action = greedy_algorithm(q_value[state])
                state, rew = self.find_next_state(state, action)
                reward += np.power(self.discount, time) * rew
                time += 1
            returns[run] = reward
        return np.mean(returns)

    def uniform_case(self, interval_time):
        start_state_value = []
        q_value = np.zeros((self.num_states, len(self.actions)))
        for it in tqdm(range(self.iterations)):
            state = it // len(self.actions) % self.num_states
            action = it % len(self.actions)
            next_state_all = self.transition[state, action, :]

            q_value[state, action] = (1 - self.terminal_pro)*(self.reward[state, action, :] +
                                                              np.max(q_value[next_state_all, :], axis=1)).mean()
            if it % interval_time == 0:
                estimate_value = self.compute_value(q_value)
                start_state_value.append([it, estimate_value])
        return zip(*start_state_value)

    def on_policy_case(self, interval_time):
        start_state_value = []
        q_value = np.zeros((self.num_states, len(self.actions)))
        state = 0       # the start state
        for it in tqdm(range(self.iterations)):
            action = self.epsilon_greedy(q_value[state])            # to select an action under epsilon-policy
            next_state, _ = self.find_next_state(state, action)     # feedback the next state and reward
            next_state_all = self.transition[state, action, :]      # all possible next state under this state-action

            q_value[state, action] = (1 - self.terminal_pro)*(self.reward[state, action, :] +
                                                              np.max(q_value[next_state_all, :], axis=1)).mean()
            if next_state == self.num_states:
                next_state = 0
            if it % interval_time == 0:
                estimate_value = self.compute_value(q_value)
                start_state_value.append([it, estimate_value])
            state = next_state
        return zip(*start_state_value)


if __name__ == "__main__":
    num_stat = [1000, 10000]
    branch = [[1, 3, 10], [1]]

    num_tasks = 30      # average across 30 tasks
    ticks = 200          # number of evaluation points
    i = 1
    for states, branches in zip(num_stat, branch):
        plt.figure(i)
        for b in branches:
            all_tasks = [TrajectorySampling(states, b) for _ in range(num_tasks)]
            uniform_values = []
            on_policy_values = []
            for task in all_tasks:
                step, value = task.uniform_case(interval_time=task.iterations/ticks)
                uniform_values.append(value)
                step, value = task.on_policy_case(interval_time=task.iterations/ticks)
                on_policy_values.append(value)
            uniform_values = np.mean(np.asarray(uniform_values), axis=0)
            on_policy_values = np.mean(np.asarray(on_policy_values), axis=0)
            plt.plot(step, uniform_values, label=r"b=%d, uniform" % b)
            plt.plot(step, on_policy_values, label=r"b=%d, on policy" % b)
        plt.title("%d States" % states)
        plt.xlabel("Computation time, in expected updates")
        plt.ylabel("Value of start state under greedy policy")
        plt.legend()
        plt.savefig("./images/Figure8-8-"+str(i)+".png")
        plt.show()
        plt.close()
        i += 1
    print("Completed!!! You can check it in the 'images' directory")
