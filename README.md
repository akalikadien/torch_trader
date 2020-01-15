# Torch trader
A cryptocurrency trader using deep-Q learning. During my minor a similar trader (CrypTorch) was created by my group.
Using this experience and knowledge I tried to build an improved trader from scratch. The trader has
3 actions: it can either buy as much BTC as possible, sell as much BTC as possible or hold
its current position.    

## fetch_data.py
For fetching and modifying data. CCXT is used to fetch OHLCV data from exchanges. 
Price data from cryptocurrencypricehistory/ originates from [this kaggle dataset](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory).
Pre-processing of data is done using 9 indicators. These will be the features that are fed to the neural network.

## neuralnet.py
Contains the neural network class and functions.

## portfolio.py
Contains buying and selling functions and returns the corresponding action.  

## agent.py 
Contains logic of the trading agent.