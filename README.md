# Optimizing March Madness: Picking the best bracket with Reinforcement Learning

This project aims to develop a reinforcement learning algorithm to optimize the selection of a bracket for the March Madness tournament using data from the popular sports analysis website, FiveThirtyEight (538). The goal is to train an agent to select the most likely winning team for each game in the bracket, with the objective of achieving the highest possible score.

## Setup

```bash 
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is sourced from 538's March Madness Predictions page, which provides probabilities for each team to win each game in the tournament. The dataset is updated as the tournament progresses and includes data on team rankings, game locations, and other relevant features.

## Environment

The environment for this project is the March Madness tournament bracket. Each game in the bracket is a state, and the actions that the agent can take are selecting the winning team for each game. The reward signal is a measure of how well the agent's bracket performs, based on the actual outcomes of the tournament.

```python

filename = 'data/fivethirtyeight_ncaa_forecasts.csv'
env = march_madness.MarchMadnessEnvironment(filename=filename)
observation, info = env.reset()

# get the two teams in the current bracket
env.get_current_matchup()

# pick team 1, action=0 to pick team 2 
observations, reward, done, info = env.step(1) 
``` 

## Reinforcement Learning Algorithm

This project provides implementations of the following approaches

- **Greedy Approach** The greedy algorithmis a simple approach that always selects the action with the highest estimated reward. This will be our baseline.
- **Deep Q-Network** is a type of reinforcement learning 
algorithm that uses a neural network to approximate the 
action-value function in a Q-learning algorithm.

- **Monte Carlo Tree Search**: This decision-making algorithm builds a search tree and simulates the game forward to select actions. It has demonstrated success in various domains, including game-playing AI and robotics tasks.

- **Temporal Difference Learning (TD)**: This approach combines aspects of both Monte Carlo and dynamic programming to learn the value function of a policy.

## Results

The performance of the reinforcement learning algorithm is evaluated by comparing the agent's bracket performance to that of a baseline model, such as selecting the higher-ranked team in each game. The evaluation metrics used include the overall expected reward of the bracket.

## Conclusion

This project demonstrates the potential of using reinforcement learning to optimize bracket selection for the March Madness tournament, using data from 538. The results show that the reinforcement learning algorithm outperforms the baseline model, and suggest that further improvements can be made by incorporating additional features and tuning the algorithm parameters.