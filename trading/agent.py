# -*- coding: utf-8 -*- 
# __author__: Adarsh Kalikadien #
import torch
import torch.nn as nn
import random
import pandas as pd
import matplotlib.pyplot as plt
from trading.neuralnet import QvalueNN
from trading.portfolio import Portfolio


class Agent:
    def __init__(self, starting_timestamp, gamma=0.7, epsilon=0.6):
        self.timestamp = starting_timestamp
        self.features_data_filename = '../data/train.csv'
        self.features_data = pd.read_csv(self.features_data_filename)
        self.loss_memory = []
        self.portfolio_value_memory = []
        self.relative_strength_memory = []
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

    @staticmethod
    def calculate_reward(old, new):
        reward = (new-old)/old
        return reward

    @staticmethod
    def plot_loss(listname, title):
        plt.plot(listname)
        plt.xlabel('Days')
        plt.ylabel('Loss')
        plt.title('loss vs days ' + title)
        plt.show()

    @staticmethod
    def plot_portfolio_value(listname, title):
        plt.plot(listname)
        plt.xlabel('Days')
        plt.ylabel('Value in $')
        plt.title('value vs days ' + title)
        plt.show()

    @staticmethod
    def plot_relative_strength(listname, title):
        plt.plot(listname)
        plt.xlabel('Days')
        plt.ylabel('Relative strength')
        plt.title('relative strength vs days ' + title)
        plt.show()

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

    def get_feature(self, timestamp):
        timestamp_df = self.features_data[self.features_data['Date'] == timestamp]
        timestamp_df = timestamp_df.set_index('Date')
        feature = timestamp_df.iloc[0, :].values
        return torch.Tensor(feature)

    def train(self):
        self.portfolio.initialize_portfolio()
        epochs = 1365
        learning_rate = 10e-4
        optimizer = torch.optim.SGD(self.q_value_nn.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss(reduction='sum')
        self.set_timestamp(1396735200)
        for x in range(epochs+1):
            try:
                feature = self.get_feature(self.timestamp)
            except:
                self.timestamp += 86400
            try:
                old_portfolio_value = self.portfolio.calculate_portfolio_value(self.timestamp)
            except:
                self.timestamp += 86400
            q_predict = self.q_value_nn.forward(feature)    # first forward pass
            chosen_action = self.choose_action(q_predict)
            actual_action = self.do_action(chosen_action)
            q_absolute = q_predict[actual_action]   # Q value of actual action to use in calculation of loss
            # print('q absolute:', q_absolute)
            self.timestamp += 86400
            try:
                new_portfolio_value = self.portfolio.calculate_portfolio_value(self.timestamp)
            except:
                self.timestamp += 86400
            reward = self.calculate_reward(old_portfolio_value, new_portfolio_value)
            try:
                feature1 = self.get_feature(self.timestamp)
            except:
                self.timestamp += 86400
            q_predict1 = self.q_value_nn.forward(feature1)  # second forward pass to find s',a'
            # new_q = reward + gamma*max(Q(s',a'))
            new_q = reward + self.gamma * torch.max(q_predict1)
            loss = loss_fn(new_q, q_absolute)
            self.loss_memory.append(loss.item())
            print('epoch:' + str(x), 'action: ' + str(actual_action), 'reward: ' + str(reward), 'loss: ' + str(loss.item()))
            print('current timestamp:', self.timestamp)
            self.portfolio_value_memory.append(new_portfolio_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.plot_loss(self.loss_memory, 'training')
        self.plot_portfolio_value(self.portfolio_value_memory, 'training')

    def test(self):
        self.features_data_filename = '../data/test.csv'
        self.features_data = pd.read_csv(self.features_data_filename)
        self.loss_memory = []
        self.portfolio_value_memory = []
        self.portfolio.initialize_portfolio()
        self.set_timestamp(1367186400)
        portfolio_start_value = self.portfolio.calculate_portfolio_value(self.timestamp)
        btc_start_value = self.portfolio.get_price(self.timestamp)
        epochs = 342
        loss_fn = nn.MSELoss(reduction='sum')
        for x in range(epochs+1):
            try:
                feature = self.get_feature(self.timestamp)
            except:
                self.timestamp += 86400
            try:
                old_portfolio_value = self.portfolio.calculate_portfolio_value(self.timestamp)
            except:
                self.timestamp += 86400
            q_predict = self.q_value_nn.forward(feature)    # first forward pass
            chosen_action = self.choose_action(q_predict)
            actual_action = self.do_action(chosen_action)
            q_absolute = q_predict[actual_action]   # Q value of actual action to use in calculation of loss
            # print('q absolute:', q_absolute)
            self.timestamp += 86400
            btc_current_value = self.portfolio.get_price(self.timestamp)
            try:
                new_portfolio_value = self.portfolio.calculate_portfolio_value(self.timestamp)
            except:
                self.timestamp += 86400
            reward = self.calculate_reward(old_portfolio_value, new_portfolio_value)
            try:
                feature1 = self.get_feature(self.timestamp)
            except:
                self.timestamp += 86400
            q_predict1 = self.q_value_nn.forward(feature1)  # second forward pass to find s',a'
            # new_q = reward + gamma*max(Q(s',a'))
            new_q = reward + self.gamma * torch.max(q_predict1)
            relative_strength = (new_portfolio_value/portfolio_start_value)/(btc_current_value/btc_start_value)
            self.relative_strength_memory.append(relative_strength)
            loss = loss_fn(new_q, q_absolute)
            self.loss_memory.append(loss.item())
            print('epoch:' + str(x), 'action: ' + str(actual_action), 'reward: ' + str(reward), 'loss: ' + str(loss.item()))
            print('current timestamp:', self.timestamp)
            self.portfolio_value_memory.append(new_portfolio_value)
        self.plot_relative_strength(self.relative_strength_memory, 'test')
        self.plot_loss(self.loss_memory, 'test')
        self.plot_portfolio_value(self.portfolio_value_memory, 'test')


if __name__ == '__main__':
    agent = Agent(1396735200)
    agent.train()
    agent.test()
    # feature2 = agent.get_feature(1396735200)
    # print(feature2)
