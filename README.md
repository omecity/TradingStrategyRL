## A Stock Trading Algorithm Using Deep Q-Learning (DQN) With PyTorch

Built a stock trading algorithm that autonomously makes buy/sell decisions using Deep Q-
Learning (DQN) with PyTorch. The agent was trained on historical market data to maximize
long-term profit by learning optimal trading strategies.


### Fetch Stock Data ###

Fetch data from yahoo finance by passing in formal parameters - ticker, start date, and end date.


### Trading Environment ###

An environment class that would store the data, and initialize the initial balance, portfolio value, and holdings. It also include methods that returns the initial balance and holdings, resets the initial balance and holdings. Finally, another method that steps through each day and decides whether to hold, buy, or sell.


### Trading Model 

The DQN Model defines a neural network to approximate the Q-function for action selection. The input is a 3-dimensional state [current_price, balance, holdings], and the output are 3-dimensional Q-values action values (Hold, Buy, Sell).


### Trading Agent 

The DNAgent Class, which is the agent class represents the RL agent, interacting with the environment and learning from experiences. When the object the class is created, then the object is instantiated with the parameters in the method signature. There is also a store_experience, act, and train methods within the class implementation.
`

### Trading Training 

An agent trade training method that takes the ticker, start datae, end date, and number episodes, trains the agent and returns the rewards, portfolio values, actions, and prices.
