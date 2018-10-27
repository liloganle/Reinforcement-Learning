# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk(object):
    def __init__(self, alpha):
        self.alpha = alpha              # the size of step
        self.num_state = 19             # the number of non-terminal states
        self.states = np.arange(self.num_state + 2)          # non-terminal states(1-19) plus two terminal states(0,20)
        self.start_state = int(np.median(self.states))       # the beginning state
        self.end_state = [self.states[0], self.states[-1]]   # the terminal state

        self.true_value = np.arange(-20, 22, 2) / (self.num_state + 1)      # the true state value
        self.true_value[0] = 0          # the value of terminal state is zero
        self.true_value[-1] = 0

        self.weight = np.zeros(self.num_state + 2)          # weight is state value due to linear function approximation
        self.trajectory_list = []                           # record the trajectory of transition state
        self.reward_list = []                               # record the reward of transition state

    def find_next_state(self, state):
        """get the next state randomly and reward"""
        if np.random.binomial(1, 0.5):
            next_state = state + 1              # take the right action
        else:
            next_state = state - 1              # take the left action
        # calculate the reward
        if next_state == self.end_state[1]:
            reward = 1                          # reach the rightmost state:20
        elif next_state == self.end_state[0]:
            reward = -1                         # reach the leftmost state:0
        else:
            reward = 0
        return next_state, reward

    def update_stage(self, state, reward):
        """update the list of trajectory and reward"""
        self.trajectory_list.append(state)
        self.reward_list.append(reward)

    def get_state_value(self, state):
        """get the value of current state"""
        return self.weight[state]


class OffLineLambdaReturn(RandomWalk):
    def __init__(self, alpha, var_lambda):
        RandomWalk.__init__(self, alpha)
        self.var_lambda = var_lambda            # the size of lambda
        self.threshold = 1e-3
        # self.reward = 0                         # default 0,record the current reward
        self.length = 0                         # record the length of the episode

    def restart(self):
        """start or restart after finish a episode"""
        self.trajectory_list.append(self.start_state)
        self.reward_list.append(0)

    def n_step_return(self, time, n_step):
        """compute the n-step return"""
        if time + n_step < self.length:
            # the discounted factor default is 1, and almost rewards are zero except for the last reward
            returns = self.reward_list[time + n_step] + self.weight[self.trajectory_list[time + n_step]]
        else:
            returns = self.reward_list[-1]
        return returns

    def lambda_return(self, time):
        """compute the lambda return"""
        returns = 0             # record the lambda returns
        lambda_power = 1        # record lambda^(n-1)
        for n in range(1, self.length - time):
            returns += lambda_power*self.n_step_return(time, n)         # the first term of lambda-return
            lambda_power *= self.var_lambda
            # if lambda_power less than the threshold, then discard
            if lambda_power < self.threshold:
                break
        returns *= 1 - self.var_lambda
        if lambda_power >= self.threshold:
            returns += lambda_power*np.sum(self.reward_list[time:self.length])        # the second term of lambda-term
        return returns

    def offline_algorithm(self):
        """the offline lambda-return algorithm"""
        for t in range(self.length):
            # update for every state in record trajectory
            state = self.trajectory_list[t]
            delta = self.lambda_return(t) - self.get_state_value(state)
            self.weight[state] += self.alpha * delta

    def solve(self):
        """apply offline lambda-return to solve the random-walk problem"""
        self.length = len(self.trajectory_list)
        self.offline_algorithm()


def solve_method(method_class):
    method_class.restart()              # beginning
    init_state = method_class.start_state           # initialize the state

    while init_state not in method_class.end_state:
        next_state, reward_temp = method_class.find_next_state(init_state)
        method_class.update_stage(next_state, reward_temp)
        init_state = next_state
    if init_state in method_class.end_state:
        method_class.solve()


if __name__ == "__main__":
    runs = 50               # the number of running
    episodes = 10           # the number of episodes
    lambda_list = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    alpha_list = [np.arange(0, 1.1, 0.1)]*len(lambda_list)

    error = np.zeros((len(lambda_list), len(alpha_list[0])))        # the root mean square error
    for run in tqdm(range(runs)):
        for lambda_idx, lambda_val in enumerate(lambda_list):
            for alpha_idx, alpha_val in enumerate(alpha_list[lambda_idx]):
                problem_class = OffLineLambdaReturn(alpha=alpha_val, var_lambda=lambda_val)
                for epi in range(episodes):
                    solve_method(method_class=problem_class)
                    estimate_value = np.asarray([problem_class.get_state_value(stat) for stat in problem_class.states])
                    error[lambda_idx, alpha_idx] += np.sqrt(np.mean(np.power(estimate_value[1:-1] -
                                                                             problem_class.true_value[1:-1], 2)))
    error /= runs * episodes
    for i in range(len(lambda_list)):
        plt.plot(alpha_list[i], error[i, :], label=r"$\lambda = $" + str(lambda_list[i]))
    plt.ylim([0.25, 0.60])
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Root Mean Square Error of Each Episode")
    plt.title(r"Offline $\lambda$-return algorithm")
    plt.legend()
    plt.savefig("./images/Figure12-3.png")
    plt.show()
    plt.close()
    print("Completed!!!You can check it in 'images' directory")




