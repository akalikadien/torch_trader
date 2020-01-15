# -*- coding: utf-8 -*- 
# __author__: Adarsh Kalikadien #
import torch
import random
from trading.neuralnet import QvalueNN
from trading.portfolio import Portfolio


class Agent:
    def __init__(self, starting_timestamp, gamma=0.7, epsilon=0.9):
        self.timestamp = starting_timestamp
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.0005
        self.epsilon_min = 0.1
        self.q_value_nn = QvalueNN(9, 3, 9)
        self.portfolio = Portfolio(0, 10000)
        self.action_hold = 0
        self.action_buy = 1
        self.action_sell = 2
        self.possible_actions = [self.action_hold, self.action_buy, self.action_sell]

    def choose_random_action(self):
        return random.choice(self.possible_actions)

    def choose_action(self, q_values):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        if self.epsilon < random.random():
            return torch.argmax(q_values)  # choose greedy action
        else:
            return self.choose_random_action()

    def do_action(self, action):
        if action == self.action_hold:
            return action
        if action == self.action_buy:
            buy_all = self.portfolio.buy_btc(self.timestamp)
            return buy_all
        if action == self.action_sell:
            sell_all = self.portfolio.sell_btc(self.timestamp)
            return sell_all
        else:
            print("action doesn't exist")
            return

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
        return
