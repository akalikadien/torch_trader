# -*- coding: utf-8 -*- 
# __author__: Adarsh Kalikadien #
import pandas as pd
import random


class Portfolio:
    def __init__(self, amount_btc=0, amount_dollar=10000):
        self.amount_btc = amount_btc
        self.amount_dollar = amount_dollar
        self.price_data_filename = 'data/bitcoin_price_modified.csv'
        self.price_data = pd.read_csv(self.price_data_filename)

    def initialize_portfolio(self):
        self.amount_btc = 0
        self.amount_dollar = 10000
        return

    # get price of the day with a given timestamp
    def get_price(self, timestamp):
        try:
            timestamp_df = self.price_data[self.price_data['Date'] == timestamp]
            timestamp_df = timestamp_df.set_index('Date')
            timestamp_df = timestamp_df.values
            price = (timestamp_df[0][1]+timestamp_df[0][2])/2
        except:
            price = random.randint(50, 500)  # choose random price if timestamp is not found
        return price

    # buy as much BTC as possible unless there are no dollars left
    def buy_btc(self, timestamp):
        price = self.get_price(timestamp)
        if self.amount_dollar != 0:
            self.amount_btc += self.amount_dollar/price  # buy as much BTC possible
            self.amount_dollar -= self.amount_dollar
            return 1
        else:
            print('Not enough dollars in current portfolio to buy')
            return 0  # hold current position

    # sell all BTC and get dollars unless no BTC is left
    def sell_btc(self, timestamp):
        price = self.get_price(timestamp)
        if self.amount_btc != 0:
            self.amount_dollar += self.amount_btc*price  # sell as much BTC possible
            self.amount_btc -= self.amount_btc
            return 2
        else:
            print('Not enough BTC in current portfolio to sell')
        return 0  # hold current position

    def calculate_portfolio_value(self, timestamp):
        price = self.get_price(timestamp)
        return self.amount_dollar + (self.amount_btc*price)


if __name__ == '__main__':
    port = Portfolio()
    print(port.get_price(123))

