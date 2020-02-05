# -*- coding: utf-8 -*- 
# __author__: Adarsh Kalikadien #

import random
from ta import *
import os
import sys
import csv
import ccxt
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')


def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
        # if we have reached the beginning of history
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        print(len(all_ohlcv), 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
        # if we have reached the checkpoint
        if fetch_since < since:
            break
    return all_ohlcv


def write_to_csv(filename, data):
    with open(filename, mode='w') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(data)


def scrape_candles_to_csv(filename, exchange_id, max_retries, symbol, timeframe, since, limit):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,  # required by the Manual
    })
    # convert since from string to milliseconds integer if needed

    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    # save them to csv file
    write_to_csv(filename, ohlcv)
    print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]), 'to', filename)


# convert data in weird format to unix timestamps
def convert_to_timestamp(date_in_weird_format):
    date_string = str(date_in_weird_format)
    date_string = date_string.replace(',', '')
    date_time_obj = datetime.datetime.strptime(date_string, '%b %d %Y')
    timestamp = datetime.datetime.timestamp(date_time_obj)
    return int(timestamp)


# function to remove commas as seperators as used in original dataset
def remove_commas(word):
    if word == '-':  # volume is '-' in early cases
        word = random.randrange(10000, 50000, 100)  # between 10000 and 50000 in steps of 100
    else:
        word = word.replace(',', '')
    return int(word)


# format data in order to be able to pre-process it for machine learning
def format_data(filename):
    df = pd.read_csv(filename, sep=',')  # , header=None)
    df['Date'] = df['Date'].apply(convert_to_timestamp)
    df['Volume'] = df['Volume'].apply(remove_commas)
    df['Market Cap'] = df['Market Cap'].apply(remove_commas)
    new_filename = filename[:-4] + '_modified.csv'
    df.to_csv(new_filename, sep=',', index=False)
    print('formatted data saved as', new_filename)


# pre-process data into 9 useful technical indicators
def preprocess_data(filename):
    df = pd.read_csv(filename, sep=',')

    # Clean nan values
    df = utils.dropna(df)

    # Add multiple indicators filling Nans values
    df['rsi'] = rsi(df['Close'], n=5, fillna=True)
    df['macd'] = macd(df['Close'], n_fast=12, n_slow=26, fillna=False)
    df['macd_signal'] = macd_signal(df['Close'], n_fast=12, n_slow=26, n_sign=9, fillna=False)
    df['obv'] = on_balance_volume(df['Close'], df['Volume'], fillna=False)
    df['ichimoku'] = ichimoku_a(df['Close'], df['Low'], n1=9, n2=26, fillna=False)
    df['mfi'] = money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], n=14, fillna=False)
    df['bb_high_indicator'] = bollinger_hband_indicator(df['Close'], n=10, ndev=2, fillna=True)
    df['bb_low_indicator'] = bollinger_lband_indicator(df['Close'], n=10, ndev=2, fillna=True)

    # Normalizing the indicators except the bollinger bands
    df["Market Cap"] = df["Market Cap"] / df["Market Cap"].max()
    df["rsi"] = df["rsi"] / df["rsi"].max()
    df["macd"] = df["macd"] / df["macd"].max()
    df["macd_signal"] = df["macd_signal"] / df["macd_signal"].max()
    df["obv"] = df["obv"] / df["obv"].max()
    df["ichimoku"] = df["ichimoku"] / df["ichimoku"].max()
    df["mfi"] = df["mfi"] / df["mfi"].max()

    # Deleting the first 51 rows and last row were certain indicators can't be computed
    df = df.iloc[51:]
    df = df.iloc[:-1]

    # Deleting the useless columns
    del df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    # Save the data to a new csv file
    new_filename = filename[:-4] + '_preprocessed.csv'
    df.to_csv(new_filename, sep=',', index=False)  # ,header=False, index=False)


if __name__ == "__main__":
    # scrape_candles_to_csv('bitfinex.csv', 'bitfinex', 3, 'BTC/USDT', '1d', '2015-01-01T00:00:00Z', 10000)
    # print(ccxt.exchanges)

    # format_data('bitcoin_price.csv')
    # preprocess_data('bitcoin_price_modified.csv')

    # create train and test dataset
    data = pd.read_csv('bitcoin_price_modified_preprocessed.csv')
    data = data.set_index('Date')
    train = data.iloc[:1366, :]
    train.to_csv('train.csv', sep=',')
    test = data.iloc[1366:, :]
    test.to_csv('test.csv', sep=',')
