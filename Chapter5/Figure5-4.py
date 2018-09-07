# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Infinite_Variance(object):
    def __init__(self, left=1, right=0):
        self.left = left
        self.right = right
    
    def target_policy(self):
        return self.left
    
    def behavior_policy(self):
        return np.random.binomial(1, 0.5)
    
    def game_start(self):
        trajectory = []
        while True:
            action = self.behavior_policy()
            trajectory.append(action)
            if action == self.right:
                return 0, trajectory
            if np.random.binomial(1, 0.9) == self.right:
                return 1, trajectory

    def off_policy_MC(self, episodes=1000000):
        rewards = []
        for iteration in np.arange(episodes):
            reward, trajectory = self.game_start()
            if trajectory[-1] == self.right:
                rho = 0
            else:
                rho = 1/np.power(0.5, len(trajectory))
            rewards.append(rho*reward)
        returns = np.add.accumulate(np.asarray(rewards))

        return returns/np.arange(1, episodes+1)

if __name__ == "__main__":
    iterations = 10   # the number of iteration
    example = Infinite_Variance()

    print("Starting......")
    for iteration in np.arange(iterations):
        estimation = example.off_policy_MC()
        plt.plot(np.arange(1, len(estimation)+1), estimation)

    plt.xlabel("Episodes (log scale)")
    plt.ylabel("MC Estimate of $v_{\pi}(s)$ with Ordinary Importance Sampling")
    plt.xscale("log")
    plt.savefig("./images/Figure5-4.png")
    print("Completed !!! You can check it in the 'images' directory.")

    plt.show()
    plt.close()



