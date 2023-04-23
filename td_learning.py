import numpy as np
import matplotlib.pyplot as plt
import dq_learning

from collections import defaultdict
from tqdm import tqdm
from typing import Tuple
from march_madness import MarchMadnessEnvironment


class MarchMadnessAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.n_actions = 2
        self.q_values = defaultdict(lambda: np.zeros(2))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: Tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            # NOTE: fill this in
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def exploit(self, env=None):
        if env is None:
            env = MarchMadnessEnvironment()
        state, info = env.reset()
        rewards = 0
        for _ in range(67):
            state = tuple(state)
            action = np.argmax(self.q_values[state])                      # Chose action from the Q-Table
            state, reward, done, info = env.step(action) # Carry out the action
            print(info)
            
            if done:
                print(f"Test episode done")
                state, info = env.reset()
                #break

        env.close()
        return reward


def train_agent(
    learning_rate = 0.01,
    n_episodes = 50_000,
    start_epsilon = 1.0,
    final_epsilon = 0.1
):
    epsilon_decay = start_epsilon / (n_episodes / 2) 

    agent = MarchMadnessAgent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )  
    env = MarchMadnessEnvironment()

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # Play one episode
        while not done:
            obs = tuple(obs)
            action = agent.get_action(obs)
            next_obs, reward, terminated, info = env.step(action)

            next_obs = tuple(next_obs)
            # Update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # Update the current observation, and whether the environment is done
            done = terminated
            obs = next_obs

        agent.decay_epsilon() 

    reward = agent.exploit(env)
    print(f'best reward: {reward}')
    return agent


if __name__ == "__main__":

    train_agent()
