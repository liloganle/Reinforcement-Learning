# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def find_next_state(current_state):
    # to select randomly the action to next state
    if np.random.binomial(1, 0.5):
        current_state += 1
    else:
        current_state -= 1

    return current_state


class Random_Walk(object):
    def __init__(self, initial_state=3, num_state=7, step_size=0.1):
        self.initial_state = initial_state    # start in the center state, C, state number is 3
        self.num_state = num_state    # the number of states
        self.step_size = step_size   # the step_size is the step size

        self.right_action = 1   # proceed right
        self.left_action = 0    # proceed left

        self.states = np.arange(num_state)   # the set/array of states
        self.initial_value = np.ones(num_state) / 2     # the initial values of all states
        self.initial_value[0] = 0
        self.initial_value[-1] = 1
        self.real_value = np.arange(num_state) / (num_state - 1)      # the true values of all states

    def temporal_difference(self, value):
        """
        the temporal difference method
        :param value: the value array of all states
        :return:
        """
        state = self.initial_state
        trajectory = [state]
        rewards = []
        
        while True:
            next_state = find_next_state(state)   # to find the next state
            reward = 0
            # if next_state == self.states[-1]:
            #    reward = 1

            rewards.append(reward)
            trajectory.append(next_state)
            value[state] += self.step_size * (reward + value[next_state] - value[state])   # TD update equation

            if next_state == 0 or next_state == self.states[-1]:
                break
            state = next_state

        return trajectory, rewards

    def monte_carlo(self, value):
        """
        the Monte Carlo method
        :param value: the value array of all states
        :return:
        """
        state = self.initial_state
        trajectory = [state]

        while True:
            next_state = find_next_state(state)
            trajectory.append(next_state)

            # ending up with left terminal state,that state is 0,all returns are 0
            if next_state == 0:
                returns = 0
                break
            # ending up with right terminal state, that state is 6, all returns are 1
            if next_state == self.states[-1]:
                returns = 1
                break
            state = next_state

        for k in trajectory[:-1]:
            value[k] += self.step_size * (returns - value[k])

        return trajectory, [returns]*(len(trajectory)-1)

    def root_mean_square(self, episode=100, runs=100, method_flag=False):
        """
        to compute the root mean-squared error
        :param episode: the number of episodes
        :param runs: the number of runs
        :param method_flag: whether use MC method or TD method
        :return:
        """
        error_rms = np.zeros(episode+1)
        for run in tqdm(range(runs)):
            current_value = self.initial_value.copy()
            for idx in np.arange(episode+1):
                error_rms[idx] += np.sqrt(np.linalg.norm(current_value - self.real_value)**2 /
                                          len(current_value[1:-1]))    # root mean squared equation
                if method_flag:   # Monte Carlo method if True
                    self.monte_carlo(current_value)
                else:   # default method is temporal-difference
                    self.temporal_difference(current_value)
        error_rms /= runs    # to average the error
        return error_rms



if __name__ == "__main__":
    episodes = [0, 1, 10, 100]
    x_label = ["A", "B", "C", "D", "E"]

    alpha_mc = [0.01, 0.02, 0.03, 0.04]
    alpha_td = [0.15, 0.10, 0.05]

    example_6_2 = Random_Walk()
    values = example_6_2.initial_value.copy()
    print("Starting plotting the figure......\n")
    
    plt.figure(1)
    for epi in np.arange(episodes[-1]+1):
        if epi in episodes:
            plt.plot(x_label, values[1:-1], "o-", label=str(epi) + " episodes")
        example_6_2.temporal_difference(values)
    plt.plot(x_label, example_6_2.real_value[1:-1], "o-", label="True Value")
    plt.xlabel("State")
    plt.ylabel("Estimated Value")
    plt.legend()
    plt.savefig("./images/Example6-2-1.png")
    plt.show()
    plt.close()

    plt.figure(2)
    for i, alpha in enumerate(alpha_mc + alpha_td):
        example_6_2 = Random_Walk(step_size=alpha)
        if i < len(alpha_mc):
            flag = True
        else:
            flag = False
        total_error = example_6_2.root_mean_square(method_flag=flag)
        if flag:
            plt.plot(total_error, "r--", linewidth=(i + 0.5), label=r"MC, $\alpha$=%.2f" % alpha)
        else:
            plt.plot(total_error, "b-", linewidth=(i - 3), label=r"TD, $\alpha$=%.2f" % alpha)
    plt.xlabel("Walks/Episodes")
    plt.ylabel("Empirical RMS Error")
    plt.legend()
    plt.savefig("./images/Example6-2-2.png")
    plt.show()
    plt.close()

    print("Completed!!! You check it in the 'images' directory")






