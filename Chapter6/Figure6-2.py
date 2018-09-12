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


class Random_Walk_Batch_Update(object):
    def __init__(self, initial_state=3, num_state=7, step_size=1e-4):
        self.initial_state = initial_state  # start in the center state, C, state number is 3
        self.num_state = num_state  # the number of states
        self.step_size = step_size  # the step_size is the step size

        self.right_action = 1  # proceed right
        self.left_action = 0  # proceed left

        self.states = np.arange(num_state)  # the set/array of states
        self.initial_value = np.ones(num_state) / 2  # the initial values of all states
        self.initial_value[0] = 0
        self.initial_value[-1] = 1
        self.real_value = np.arange(num_state) / (num_state - 1)  # the true values of all states

    def temporal_difference(self):
        """
        the temporal difference method
        :return:
        """
        state = self.initial_state
        trajectory = [state]
        rewards = []

        while True:
            next_state = find_next_state(state)  # to find the next state
            reward = 0
            if next_state == self.states[-1]:
                reward = 1

            rewards.append(reward)
            trajectory.append(next_state)
            # value[state] += self.step_size * (reward + value[next_state] - value[state])  # TD update equation

            if next_state == 0 or next_state == self.states[-1]:
                break
            state = next_state

        return trajectory, rewards

    def monte_carlo(self):
        """
        the Monte Carlo method
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

        # for k in trajectory[:-1]:
        #    value[k] += self.step_size * (returns - value[k])

        return trajectory, [returns] * (len(trajectory) - 1)

    def batch_update_rsm(self, episode=100, runs=100, method_flag=False):
        """
        to compute the root mean-squared error of batch updating
        :param episode: the number of episodes
        :param runs: the number of runs
        :param method_flag: whether use MC method or TD method
        :return:
        """
        error_rms = np.zeros(episode + 1)
        for run in tqdm(range(runs)):
            current_value = self.initial_value.copy()
            error = []
            for idx in np.arange(episode + 1):
                if method_flag:                                                # Monte Carlo method if True
                    list_trajectory, list_rewards = self.monte_carlo()
                else:                                                          # default method is temporal-difference
                    list_trajectory, list_rewards = self.temporal_difference()
                while True:
                    temp_array = np.zeros(self.num_state)
                    for track, rew in zip([list_trajectory], [list_rewards]):
                        for i in np.arange(len(track) - 1):
                            if method_flag:                                     # Monte Carlo method if True
                                temp_array[track[i]] += rew[i] - current_value[track[i]]
                            else:                                               # default method is temporal-difference
                                temp_array[track[i]] += rew[i] + current_value[track[i+1]] - current_value[track[i]]
                    temp_array *= self.step_size
                    if np.linalg.norm(temp_array) < 1e-4:
                        break
                    current_value += temp_array    # batch updating for state value
                # error_rms[idx] += np.sqrt(np.linalg.norm(current_value-self.real_value)**2/len(current_value[1:-1]))
                error.append(np.sqrt(np.linalg.norm(current_value-self.real_value)**2/len(current_value[1:-1])))
            error_rms += np.asarray(error)
        error_rms /= runs  # to average the error
        return error_rms


if __name__ == "__main__":
    random_walk_batch_update = Random_Walk_Batch_Update()

    td_error_rsm = random_walk_batch_update.batch_update_rsm(method_flag=False)
    mc_error_rsm = random_walk_batch_update.batch_update_rsm(method_flag=True)

    print("Starting plotting the figure......\n")
    plt.plot(td_error_rsm, label=r"TD")
    plt.plot(mc_error_rsm, label=r"MC")
    plt.legend()
    plt.savefig("./images/Figure6-2.png")
    plt.show()
    plt.close()
