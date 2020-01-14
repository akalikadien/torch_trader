# Torch trader
A cryptocurrency trader using deep-Q learning. During my minor a similar trader (CrypTorch) was created by my group.
Using this experience and knowledge I tried to build an improved trader from scratch.   

## fetch_data.py
For fetching and modifying data. CCXT is used to fetch OHLCV data from exchanges. 
Price data from cryptocurrencypricehistory/ originates from [this kaggle dataset](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory).
Pre-processing of data is done using 9 indicators. These will be the features that are fed to the neural network.

## neuralnet.py
Contains the neural network class and functions. 