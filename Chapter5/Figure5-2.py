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

    def behavior_policy_player(self):
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

    def exploring_start_MC(self, episodes=10000):
        state_action_value = np.zeros((10, 2, 10, 2))   # (playerSum, usable, dealerCard, action)
        state_action_pair_count = np.ones(state_action_value.shape)

        def greedy_policy(sum_player, usable_ace, dealer_card):
            sum_player -= 12
            usable_ace = int(usable_ace)
            dealer_card -= 1
            values = state_action_value[sum_player, usable_ace, dealer_card, :] /\
                    state_action_pair_count[sum_player, usable_ace, dealer_card]
            return np.random.choice([act for act, value in enumerate(values) if value == np.max(values)])

        for ite in np.arange(episodes):
            #randomly select initialized state and action for each episode
            initial_state = [np.random.choice(range(12, 22)), bool(np.random.choice([0, 1])),
                             np.random.choice(range(1, 11))]
            initial_action = np.random.choice([self.action_hit, self.action_stop])
            policy_player = greedy_policy if ite else self.target_policy_player
            _, reward, trajectory = self.generate_episode(policy_player, initial_state, initial_action)
            for (sum_player, usable_ace, dealer_showing), action in trajectory:
                sum_player -= 12
                usable_ace = int(usable_ace)
                dealer_showing -= 1
                #update the values and pair counts
                state_action_value[sum_player, usable_ace, dealer_showing, action] += reward
                state_action_pair_count[sum_player, usable_ace, dealer_showing, action] += 1
        return state_action_value/state_action_pair_count


def draw_image(states_mat, lables):

    x_size, y_size = states_mat[0].shape
    x_axis = np.arange(1, x_size + 1)
    y_axis = np.arange(12, y_size + 12)
    x_axis, y_axis = np.meshgrid(x_axis, y_axis)
    nums = np.arange(len(lables))

    print("Starting drawing......")
    fig = plt.figure(figsize=(14, 10))
    for state, lable, num in zip(states_mat, lables, nums):
        if num % 2 == 0:
            axis = fig.add_subplot(221+num)
            sns.heatmap(np.flipud(state), cmap="GnBu", ax=axis, xticklabels=np.arange(1, 11),
                        yticklabels=np.arange(21, 11, -1))
            plt.xlabel("Dealer showing")
            plt.ylabel("Player sum")
            plt.title(lable)
        else:
            axis = fig.add_subplot(221+num, projection='3d')
            axis.plot_wireframe(X=x_axis, Y=y_axis, Z=state)
            plt.xlabel("Dealer showing")
            plt.ylabel("Player sum")
            plt.title(lable)

    plt.savefig("./images/Figure5-2.jpg", bbox="tight")
    plt.close()
    print("Completed !!! You can check it in the 'images' directory.")


if __name__ == "__main__":
    blackjack_game = Blackjack()
    print("The Blackjack Game is starting......")
    state_action_value = blackjack_game.exploring_start_MC(episodes=500000)
    print("Game Over !!!\n")

    #to get the max state-action value
    state_action_value_no_usable = np.max(state_action_value[:, 0, :, :], axis=2)
    state_action_value_usable = np.max(state_action_value[:, 1, :, :], axis=2)
    #to get the optimal policy
    policy_no_usable = np.argmax(state_action_value[:, 0, :, :], axis=2)
    policy_usable = np.argmax(state_action_value[:, 1, :, :], axis=2)

    states = [policy_usable, state_action_value_usable,
              policy_no_usable, state_action_value_no_usable]
    lables = ["Usable Ace, Optimal policy", "Usable Ace, Optimal state-action value",
              "No Usable Ace, Optimal policy", "No Usable Ace, Optimal state-action value"]

    draw_image(states, lables)








