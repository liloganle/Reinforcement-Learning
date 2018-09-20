# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk(object):
    """
    n-step TD method on the random walk
    """
    def __init__(self, num_states=19, step_size=0.1, discount=1):
        self.num_states = num_states + 2    # 19 intermediate states plus 2 terminal states
        self.step_size = step_size          # the size of alpha or step size
        self.discount = discount            # the size of discount

        self.states = np.arange(self.num_states)     # 19 intermediate states and 2 terminal states
        self.start = int(np.median(self.states))     # the beginning state
        self.end = [self.states[0], self.states[-1]]    # the terminal state set

        self.initial_state_value = np.zeros(len(self.states))   # initialized the state value
        self.true_state_value = np.arange(-20, 22, 2) / (self.num_states - 1)  # the true state value
        self.true_state_value[0] = 0
        self.true_state_value[-1] = 0

    def find_next_state(self, state):
        # randomly select left action or right action
        if np.random.binomial(1, 0.5):
            next_state = state + 1
        else:
            next_state = state - 1

        if next_state == self.end[0]:
            reward = -1
        elif next_state == self.end[1]:
            reward = 1
        else:
            reward = 0

        return next_state, reward

    def n_step_td(self, value, n_step=1):
        state = self.start
        trajectory = [state]
        reward_list = []

        time = 0    # to record the times
        T = float("inf")    # the length of the episode
        while True:

            if time < T:
                next_state, reward = self.find_next_state(state)
                # to store the next state and reward
                trajectory.append(next_state)
                reward_list.append(reward)

                if next_state in self.end:
                    T = time + 1

            count_update = time - n_step + 1    # to get the counts of updating
            if count_update >= 0:
                returns = 0
                for idx in range(count_update+1, min(count_update + n_step, T)+1):
                    returns += np.power(self.discount, idx - count_update - 1)*reward_list[idx - 1]
                if count_update + n_step <= T:
                    returns += np.power(self.discount, n_step)*value[trajectory[count_update + n_step]]

                if not trajectory[count_update] in self.end:
                    value[trajectory[count_update]] += self.step_size*(returns -
                                                                       value[trajectory[count_update]])
            time += 1
            if count_update == T - 1:
                break
            state = next_state          # !!!!!

    def compute_rms(self, value, n_step, episodes):
        rms_error = 0
        for epis in range(episodes):
            self.n_step_td(value, n_step)
            rms_error += np.sqrt(np.linalg.norm(value - self.true_state_value)**2 / len(value[1:-1]))

        return rms_error/episodes


if __name__ == "__main__":
    all_n_steps = np.power(2, np.arange(7))     # to get all possible value of n
    step_sizes = np.arange(0, 1.1, 0.1)
    rms_errors = np.zeros((len(all_n_steps), len(step_sizes)))

    episode = 10
    runs = 100
    for run in tqdm(range(runs)):
        for n_idx, n_value in enumerate(all_n_steps):
            for alpha_idx, alpha_value in enumerate(step_sizes):
                random_walk_man = RandomWalk(step_size=alpha_value)
                state_value = random_walk_man.initial_state_value.copy()
                rms_errors[n_idx, alpha_idx] += random_walk_man.compute_rms(value=state_value, n_step=n_value,
                                                                            episodes=episode)

    for i in range(len(all_n_steps)):
        plt.plot(step_sizes, rms_errors[i]/runs, label="n=%d" % all_n_steps[i])
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"Average RMS Error over Ten Episodes")
    plt.legend()
    plt.savefig("Figure7-2.png")
    plt.show()
    plt.close()
    print("Completed!!! You can check it in the current directory")






