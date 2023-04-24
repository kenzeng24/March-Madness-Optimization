from mcts.base.base import BaseState
from mcts.searcher.mcts import MCTS
from copy import deepcopy
from march_madness import MarchMadnessEnvironment


class MarchMadnessState(BaseState):
    """
    Represents the current state of a March Madness game.
    """
    def __init__(self, env=None, reward=0, done=False):
        if env is not None:
            self.env = env
        else:
            self.env = MarchMadnessEnvironment()
        
        self.state = self.env.state_list
        self.reward = reward
        self.done = done

    def get_possible_actions(self):
        return (0, 1)

    def take_action(self, action: any) -> 'BaseState':
        next_obs, reward, done, info = self.env.step(action)
        newState = MarchMadnessState(self.env, reward, done)
        self.reward =reward 
        self.done = done
        return newState

    def is_terminal(self) -> bool:
        if self.done:
            self.env.reset()
        return self.done

    def get_reward(self) -> float:
        return self.reward

    def get_current_player(self) -> int:
        return 0

    
def tree_search(limit=10):
    initial_state = MarchMadnessState(
        env=MarchMadnessEnvironment(
            filename='data/fivethirtyeight_ncaa_forecasts.csv'
        )
    )
    searcher = MCTS(iteration_limit=limit)
    best_action, reward = searcher.search(initial_state=initial_state, need_details=True)
    print(reward)  # the expected reward for the best action
    return searcher 


if __name__ == "__main__":
    tree_search()