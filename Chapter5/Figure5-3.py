# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class Blackjack(object):
    def __init__(self):
        self.action_hit = 1
        self.action_stop = 0
        self.actions = [self.action_hit, self.action_stop]
        self.policy_player = np.ones(22, dtype=int)
        self.policy_dealer = np.ones(22, dtype=int)

        self.policy_player[20:] = self.action_stop
        self.policy_dealer[17:] = self.action_stop

    def target_policy_player(self, sum_player, usable_ace, dealer_card):
        return self.policy_player[sum_player]

    def behavior_policy_player(self, sum_player, usable_ace, dealer_card):
        rand_num = np.random.binomial(1, 0.5)
        if rand_num:
            return self.action_hit
        else:
            return self.action_stop

    def get_card(self):
        card_num = np.random.randint(1, 14)
        return min(card_num, 10)

    def initial_game(self, initial_state=None):
        sum_player = 0   #sum of player's card
        usable_ace_player = False  # type: bool #whether player count ace as 11 or not
        card1_dealer = 0  # the first card of dealer is showing card
        card2_dealer = 0  # the second card of deale

        if initial_state is None:
            num_of_ace = 0  # the number of ace card
            # if the sum of player's card less than 12, always hits
            while sum_player < 12:
                card = self.get_card()
                # if getting an ace card, counting it as 11
                if card == 1:
                    num_of_ace += 1
                    card = 11
                    usable_ace_player = True
                sum_player += card
            if sum_player > 21:
                sum_player -= 10
                if num_of_ace == 1:
                    usable_ace_player = False

            card1_dealer += self.get_card()
            card2_dealer += self.get_card()
        else:
            sum_player, usable_ace_player, card1_dealer = initial_state
            card2_dealer = self.get_card()
        state = [sum_player, usable_ace_player, card1_dealer]  # initialize the state of game
        return state, card2_dealer

    def generate_episode(self, policy_player, initial_state=None, initial_action=None):
        #@policy_player: type:function
        sum_dealer = 0  # sum of dealer's card
        trajectory_player = []  # the player's trajectory
        usable_ace_dealer = False  #type: bool #whether dealer count ace as 11 or not

        state, card2_dealer = self.initial_game(initial_state)
        [sum_player, usable_ace_player, card1_dealer] = state
        # @sum_player: sum of player's card
        # @usable_ace_player:  type: bool #whether player count ace as 11 or not
        # @card1_dealer: the first card of dealer is showing card
        # @card2_dealer: the second card of dealer

        if card1_dealer == 1 and card2_dealer == 1:
            sum_dealer += 11 + 1
            usable_ace_dealer = True
        elif card1_dealer == 1 and card2_dealer != 1:
            sum_dealer += 11 + card2_dealer
            usable_ace_dealer = True
        elif card1_dealer != 1 and card2_dealer == 1:
            sum_dealer += card1_dealer + 11
            usable_ace_dealer = True
        else:
            sum_dealer += card1_dealer + card2_dealer


        while True:
            if initial_action is not None:
                action = initial_action
                initial_action = None
            else:
                action = policy_player(sum_player, usable_ace_player, card1_dealer)
                #get an action base on current sum of player's card
            trajectory_player.append([(sum_player, usable_ace_player, card1_dealer), action])

            if action == self.action_stop:
                break

            new_card = self.get_card()
            if new_card == 1 and sum_player+11 < 21:
                sum_player += 11
                usable_ace_player = True
            else:
                sum_player += new_card
            if sum_player > 21:
                if usable_ace_player:
                    sum_player -= 10
                    usable_ace_player = False
                else:
                    return state, -1, trajectory_player


        while True:
            action = self.policy_dealer[sum_dealer]   #get an action base on current sum of dealer's card
            if action == self.action_stop:
                break
            new_card = self.get_card()
            if new_card == 1 and sum_dealer+11 < 21:
                sum_dealer += 11
                usable_ace_dealer = True
            else:
                sum_dealer += new_card

            if sum_dealer > 21:
                if usable_ace_dealer:
                    sum_dealer -= 10
                    usable_ace_dealer = False
                else:
                    return state, 1, trajectory_player

        if sum_player > sum_dealer:
            return state, 1, trajectory_player
        elif sum_player == sum_dealer:
            return state, 0, trajectory_player
        else:
            return state, -1, trajectory_player

    def off_policy_MC(self, episodes=10000):
        sum_player = 13
        usable_ace = True
        card1 = 2

        rhos = []
        rewards = []
        initial_state = [sum_player, usable_ace, card1]

        for ite in np.arange(episodes):
            numerator = 1   #the numerator of importance-sampling ratio
            denominator = 1   #the denominator of importance-sampling ratio

            _, reward, trajectory = self.generate_episode(self.behavior_policy_player, initial_state)
            for (sum_player, usable_ace, dealer_showing), action in trajectory:
                if action == self.target_policy_player(sum_player, usable_ace, dealer_showing):
                    denominator *= 0.5
                else:
                    numerator = 0
                    break
            rho = numerator / denominator
            rhos.append(rho)
            rewards.append(reward)

        rhos = np.asarray(rhos)
        rewards = np.asarray(rewards)
        weighted_rewards = rhos * rewards
        rhos = np.add.accumulate(rhos)   #to accumulate the importance-sampling ratio
        weighted_returns = np.add.accumulate(weighted_rewards)   #to accumulate the reward

        ordinary_importance_sampling = weighted_returns / np.arange(1, episodes+1)
        weighted_importance_sampling = weighted_returns / (rhos+1e-7)   #to avoid dividing zero

        return ordinary_importance_sampling, weighted_importance_sampling



if __name__ == "__main__":
    true_value = -0.27726
    iteration = 10000
    runs = 100
    error_ordinary_sampling = np.zeros(iteration)
    error_weighted_sampling = np.zeros(iteration)

    blackjack_game = Blackjack()
    print("The Blackjack Game is starting......")
    for iters in np.arange(runs):
        ordinary_sampling, weighted_sampling = blackjack_game.off_policy_MC(episodes=iteration)
        error_ordinary_sampling += np.power(ordinary_sampling-true_value, 2)
        error_weighted_sampling += np.power(weighted_sampling-true_value, 2)
    print("Game Over !!!\n")
    error_ordinary_sampling /= runs
    error_weighted_sampling /= runs

    print("Starting drawing......")
    plt.plot(np.arange(1, iteration+1), error_ordinary_sampling, label="Ordinary importance sampling")
    plt.plot(np.arange(1, iteration+1), error_weighted_sampling, label="Weighted importance sampling")
    plt.xlabel("Episodes (log scale)")
    plt.ylabel("Mean Square Error")
    plt.xscale("log")
    plt.legend()
    plt.savefig("./images/Figure5-3.jpg")
    plt.show()

    print("Completed !!! You can check it in the 'images' directory.")
    plt.close()








