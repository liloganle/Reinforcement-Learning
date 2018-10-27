# -*- coding:utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk(object):
    def __init__(self, alpha):
        self.alpha = alpha              # record the step size
        self.num_state = 19             # the number of non-terminal states
        self.states = np.arange(self.num_state + 2)         # all states, non-terminal state plus terminal state
        self.start_state = int(np.median(self.states))      # the start state
        self.end_state = [self.states[0], self.states[-1]]      # the terminal state

        self.true_value = np.arange(-20, 22, 2)/(self.num_state + 1)        # the true state value
        self.true_value[0] = 0              # the value of terminal state is zero
        self.true_value[-1] = 0

        self.weight = np.zeros(self.num_state + 2)          # weight is state value due to linear function approximation
        self.trajectory_list = []                           # record the transition state
        self.reward_list = []                               # record the record

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


class TDLambda(RandomWalk):
    def __init__(self, alpha, var_lambda):
        # the discounted factor default is 1
        RandomWalk.__init__(self, alpha)
        self.var_lambda = var_lambda
        self.eligibility_trace = None

    def restart(self):
        """start or restart after finish a episode"""
        self.trajectory_list.append(self.start_state)
        # self.reward_list.append(0)
        self.eligibility_trace = np.zeros(len(self.weight))         # the eligibility trace vector

    def td_lambda(self, state, next_state, reward):
        """Temporal Difference lambda algorithm"""
        # the discounted factor default is 1
        self.eligibility_trace *= self.var_lambda               # eligibility trace multiply lambda,that is first term
        self.eligibility_trace[state] += 1                      # plus the gradient of value function approximation
        delta = reward + self.weight[next_state] - self.weight[state]
        self.weight += self.alpha * delta * self.eligibility_trace     # update the weight vector , that is state value


def solve_method(method_class):
    method_class.restart()              # beginning
    init_state = method_class.start_state           # initialize the state

    while init_state not in method_class.end_state:
        next_state, reward_temp = method_class.find_next_state(init_state)          # get the next state and reward
        method_class.update_stage(next_state, reward_temp)              # add new state and reward to trajectory
        method_class.td_lambda(init_state, next_state, reward_temp)
        init_state = next_state


if __name__ == "__main__":
    runs = 50               # the number of running
    episodes = 10           # the number of episodes
    lambda_list = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    alpha_list = [np.arange(0, 1.1, 0.1)]*len(lambda_list)
    error = np.zeros((len(lambda_list), len(alpha_list[0])))        # the root mean square error
    for run in tqdm(range(runs)):
        for lambda_idx, lambda_val in enumerate(lambda_list):
            for alpha_idx, alpha_val in enumerate(alpha_list[lambda_idx]):
                problem_class = TDLambda(alpha=alpha_val, var_lambda=lambda_val)
                for epi in range(episodes):
                    solve_method(method_class=problem_class)
                    estimate_value = np.asarray([problem_class.get_state_value(stat) for stat in problem_class.states])
                    error[lambda_idx, alpha_idx] += np.sqrt(np.mean(np.power((estimate_value[1:-1] -
                                                                             problem_class.true_value[1:-1]), 2)))
    error /= runs * episodes
    for i in range(len(lambda_list)):
        plt.plot(alpha_list[i], error[i, :], label=r"$\lambda = $" + str(lambda_list[i]))
    plt.ylim([0.25, 0.60])
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Root Mean Square Error of Each Episode")
    plt.title(r"TD($\lambda$) algorithm")
    plt.legend()
    plt.savefig("./images/Figure12-6.png")
    plt.show()
    plt.close()
    print("Completed!!!You can check it in 'images' directory")



