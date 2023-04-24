# baseline strategies we will compare Reinforcement learning to

import numpy as np 
from march_madness import MarchMadnessEnvironment

def random_strategy(march_madness_event):
    """
    randomly pick a team to win each matchup 
    """
    march_madness_event.reset()
    done = False 
    while not done:
        action = np.random.binomial(1,0.5)
        _, reward, done, info = march_madness_event.step(action)
        if done:
            total_reward = reward
            print(env.total_reward)
            print(info)
    return total_reward


def greedy_strategy(march_madness_event, verbose=True):
    """
    always pick the action that leads to best immediate reward
    """
    march_madness_event.reset()
    total_reward = 0 
    done = False 
    while not done:
        curr_bracket = march_madness_event.matchup_list[0]

        team1 = curr_bracket.team1.winner
        team2 = curr_bracket.team2.winner
        playoff_round = curr_bracket.playoff_round
        

        reward1, reward2 = march_madness_event.calculate_expected_rewards(team1, team2, playoff_round)
        action = 1*(reward1 > reward2)
        state, reward, done, info = march_madness_event.step(action)
        if done:
            total_reward = reward
            print(env.total_reward)
            print(info)
    return total_reward


def brute_force_strategy(march_madness_event):
    """
    iterate through all available sequences of actions 
    to identify the optimal action. 
    """
    env = MarchMadnessEnvironment()
    n = len(env.matchup_list)
    
    action_lists = [[]]
    for i in range(n):
        new_action_lists = [] 
        for action_list in action_lists:
            new_action_lists += [action_list + [1], action_list + [0]] 
        action_lists = new_action_lists 
    print(f"there are {n} possible set of actions")
    assert n < 100000, 'there are too many combinations to compute'
        
    best_reward = 0 
    for action_list in action_lists:
        env.reset()
        actual_reward = 0 
        for action in action_list:
            state, reward, done, info = env.step(action)
            if done:
                actual_reward = reward
                print(env.total_reward)
        best_reward = max(actual_reward, best_reward)
    return best_reward


if __name__ == "__main__":
    
    env = MarchMadnessEnvironment()
    # for x in env.matchup_list:
    #     print(x)
    # print(env.matchup_in_round)
    
    greedy_score = greedy_strategy(env)
    print(f'greedy reward: {greedy_score}') 
    random_score = random_strategy(env)
    print(f'random reward: {random_score}') 

    
    #optimal_score =  brute_force_strategy(env)
    # print(f'optimal reward: {optimal_score}')