# Torch trader
A simulation of a cryptocurrency trader using deep-Q learning in Pytorch. During my minor a similar trader (CrypTorch) was created by my group.
Using this experience and knowledge I tried to build an improved trader from scratch. I did this project
to get more experience in machine learning and data science. 

**This software is intended for learning purposes only**

The trader has 3 actions: it can either buy as much BTC as possible, sell as much BTC as possible or hold
its current position.
## fetch_data.py
Contains functions for fetching and modifying data. CCXT is used to fetch OHLCV data from exchanges. 
Price data from cryptocurrencypricehistory/ originates from [this kaggle dataset](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory).
Pre-processing of data is done using 9 indicators. These will be the features that are fed to the neural network.

## neuralnet.py
Contains the neural network class and functions. A nn with two linear layers and 1 Relu layer is used. 

## portfolio.py
Contains buying and selling functions and returns the corresponding action.  

## agent.py 
Contains logic of the trading agent. The training and testing loop are as described in [this paper.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.32.6578&rep=rep1&type=pdf)
In summary:
1. Starting in current state _s_, feedforward through the network with state variables as input and a Q-value as output.
2. Use an action selection criterion to select an action _a_ based on computed Q values.
3. Perform action _a_ on state _s_ resulting in state _s'_ and calculate immediate reward _r_.
4. Compute Q' = reward + gamma*max(Q(_s'_, _a'_)) (Bellman)
5. Backpropagate 