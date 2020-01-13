# -*- coding: utf-8 -*- 
# __author__: Adarsh Kalikadien #

import os
import sys
import csv
import ccxt
import pandas as pd
import datetime

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


def convert_to_timestamp(date_in_weird_format):
    date_string = str(date_in_weird_format)
    date_string = date_string.replace(',', '')
    date_time_obj = datetime.datetime.strptime(date_string, '%b %d %Y')
    timestamp = datetime.datetime.timestamp(date_time_obj)
    return int(timestamp)


def remove_commas(word):
    word = word.replace(',', '')
    return word


def format_data(filename):
    df = pd.read_csv(filename, sep=',')  # , header=None)
    df['Date'] = df['Date'].apply(convert_to_timestamp)
    df['Volume'] = df['Volume'].apply(remove_commas)
    df['Market Cap'] = df['Market Cap'].apply(remove_commas)
    new_filename = filename[:-4] + '_modified.csv'
    df.to_csv(new_filename, sep=',', index=False)
    print('formatted data saved as', new_filename)


if __name__ == "__main__":
    # scrape_candles_to_csv('bitfinex.csv', 'bitfinex', 3, 'BTC/USDT', '1d', '2015-01-01T00:00:00Z', 10000)
    # print(ccxt.exchanges)

    format_data('bitcoin_price.csv')


