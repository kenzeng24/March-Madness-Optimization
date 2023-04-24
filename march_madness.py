# import gym
import pandas as pd 
import numpy as np
from gym.spaces import Discrete, Box


def make_teams(data):
    """
    Create a team dictionary from the provided data.

    Args:
        data (pandas DataFrame): The data containing the team and its region, seed, and winning probability.

    Returns:
        teams (dict): A dictionary containing each team in each region and its corresponding seed.
        teams_map (dict):A dictionary containing each team and its corresponding region and seed.
    """
    teams = {}
    teams_map = {}
    regions = data.index.get_level_values('team_region').drop_duplicates()
    seeds = data.index.get_level_values('team_seed').drop_duplicates()
    
    for region in regions:
        teams[region] = {}
        for seed in seeds:
            if (region, seed) in data.index and not data.loc[(region, seed), :].empty:
                teams[region][seed] = data.loc[(region, seed)].index[0]
                teams_map[(region, seed)] = teams[region][seed]
        play_in_keys = [key for key in teams[region].keys() if 'a' in key or 'b' in key]
        play_in_teams = [teams[region][key] for key in teams[region].keys() if 'a' in key or 'b' in key]
        play_in_key = play_in_keys[0][:-1]
        teams[region][play_in_key] = play_in_teams
            
    return teams, teams_map
    

def example_reward_function(seed, playoff_round):
    """
    An example reward function that calculates the reward for a given seed and playoff round.

    Args:
        seed : str, The seed of a team.
        playoff_round : int, The playoff round of the game.

    Returns:
        The reward for the team with the given seed in the given playoff round.
    """
    modifier = playoff_round-1 if playoff_round != 7 else 10 
    return int(seed.replace('a','').replace('b','')) * modifier


class Bracket:
    """
    A class to represent a game bracket in the tournament.
    """
    def __init__(self, winner=None, team1=None, team2=None,playoff_round=None):
        self.winner = winner
        self.playoff_round = playoff_round
        self.team1 = team1 
        self.team2 = team2
        
    def __repr__(self):
        team1_winner, team2_winner = None, None
        if self.team1 is not None:
            team1_winner = self.team1.winner
        if self.team1 is not None:
            team2_winner = self.team2.winner
        return f'([{team1_winner},{team2_winner}], r={self.playoff_round})'
    
    def __str__(self):
        return self.__repr__()
        
        
class MarchMadnessEnvironment():
    
    def __init__(self, filename=None):
        """
        Initialize the March Madness environment.

        Parameters:
        -----------
        filename : str, default None
            The filename of the data containing the team and its region, seed, and winning probability.
        """
        self.matchup_list = [] 
        
        if filename is None:
            filename='data/fivethirtyeight_ncaa_forecasts.csv'
        # defaults to 2023 
        df_538 = pd.read_csv(filename)
        self.data = df_538[
            (df_538.forecast_date=='2023-03-12') & 
            (df_538.gender=='mens')
        ].set_index(['team_region', 'team_seed'])
        
        self.teams, self.teams_map = make_teams(self.data)
        self.teams_list = list(self.teams_map.keys())
        self.action_space=Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(len(self.teams_list),))
        self.reset()
    
    def reset(self):
        """
        Reset the environment and regenerate the brackets.
        """
        self.bracket_probability = 1
        self.matchup_list = []
        self.state = {}
        self.bracket = self.build_bracket(self.complete_bracket(self.data))
        self.depth = self.bracket.playoff_round
        self.matchup_in_round = {i:[] for i in range(self.depth+1)}
        self.update_states(self.bracket, self.depth)
        self.state_list = [0 for team in self.teams_list]
        self.state_list = self.update_state_list()
        self.discount = 1 
        self.total_reward = 0 
        return self.state_list, {}

    
    def update_state_list(self):
        for team, prob in self.state.items():
            idx = self.teams_list.index(team)
            if prob != 0:
                self.state_list[idx] = 1
            else:
                self.state_list[idx] = 0
        return self.state_list

    def complete_bracket(self, data):
        return [
            [self.make_region('South'), self.make_region('East')], 
            [self.make_region('Midwest'), self.make_region('West')]
        ]
            

    def make_region(self, region):
        """
        build lists of matchups for a particular region 
        """
        teams = [team for team in self.teams_list if team[0] == region]
        seeds = [team[1] for team in teams]
        index = {}
        for i in range(1, 17):
            if str(i) in seeds:
                index[i] = (region, str(i))
            else:
                index[i] = [(region, seed) for seed in seeds if str(i) in seed]
        bracket = [
            [
                [
                    [index[1], index[16]], 
                    [index[8], index[9]], 
                ], 
                [
                    [index[5], index[12]], 
                    [index[4], index[13]], 
                ],
            ], 
            [
                [
                    [index[6], index[11]], 
                    [index[3], index[14]], 
                ], 
                [
                    [index[7], index[10]], 
                    [index[2], index[15]], 
                ],
            ]
        ]
        return bracket
        
        
    def build_bracket(self, matchups):
        """
        recursively build bracket object and
        generate a list of all matchups 
        """
        if type(matchups) is tuple:
            # if the bracket only contains one team
            # we have winner by default 
            team = matchups
            bracket = Bracket(
                winner=team, 
                playoff_round=0
            )
            return bracket
        else:
            # bracket bracket objects for the left and right subtree
            bracket1 = self.build_bracket(matchups[0])
            bracket2 = self.build_bracket(matchups[1])

            bracket = Bracket(
                team1=bracket1, 
                team2=bracket2, 
                playoff_round=max(
                    bracket1.playoff_round, 
                    bracket2.playoff_round
                ) + 1
            )
            self.matchup_list.append(bracket)
            return bracket   
        

    def update_states(self, bracket, n):
        """
        update each bracket with their actual depth
        """
        if bracket is not None:
            if bracket.team1 is None and bracket.team2 is None:
                self.state[bracket.winner] = self.data.loc[bracket.winner, f'rd{n+1}_win']
            bracket.playoff_round = n
            self.matchup_in_round[n].append(bracket)
            self.update_states(bracket.team1, n-1)
            self.update_states(bracket.team2, n-1)


    def update_state_list(self):
        for team, prob in self.state.items():
            idx = self.teams_list.index(team)
            if prob != 0:
                self.state_list[idx] = 1
            else:
                self.state_list[idx] = 0
        return self.state_list
    
    
    def calculate_rewards(self, team1, team2, playoff_round, reward_function=example_reward_function):
        """
        calculate reward of the two teams given a reward function
        """        
        seeding1 = team1[1]
        seeding2 = team2[1]
        
        team1_reward = reward_function(seeding1, playoff_round)
        team2_reward = reward_function(seeding2, playoff_round)
        return team1_reward, team2_reward
    
    
    def calculate_win_probability(self, team1, team2, playoff_round):
        """get the expected reward for the next matchup"""
        pp1,pp2 = 1,1
        if playoff_round > 1:
            pp1 = self.data.loc[team1, f'rd{playoff_round-1}_win']
            pp2 = self.data.loc[team2, f'rd{playoff_round-1}_win']
            
        p1_cond = pp1- self.data.loc[team1, f'rd{playoff_round}_win']
        p2_cond = pp2- self.data.loc[team2, f'rd{playoff_round}_win']
        return p1_cond, p2_cond
    
    
    def calculate_expected_rewards(self, team1, team2, playoff_round):
        team1_reward, team2_reward = self.calculate_rewards(
            team1, team2, playoff_round
        )
        p1,p2 = self.calculate_win_probability(
            team1, team2, playoff_round
        )
        return (
            self.discount * (self.total_reward + team1_reward) * p2, # team 2 losing
            self.discount * (self.total_reward + team2_reward) * p1 # team 1 losiing
        )
    
    def step(self, action):
        """
        pick winner and update bracket
        """
        if len(self.matchup_list) == 0:
            raise ValueError('Matchups are complete')
            
        # get next available matchup 
        curr_bracket = self.matchup_list.pop(0)
        
        # calculate reward for each team
        team1_reward, team2_reward = self.calculate_rewards(
            curr_bracket.team1.winner, 
            curr_bracket.team2.winner, 
            curr_bracket.playoff_round
        )
        
        # calculate expected reward for each team
        p1,p2 = self.calculate_win_probability(
            curr_bracket.team1.winner, 
            curr_bracket.team2.winner, 
            curr_bracket.playoff_round
        )
        team1_reward_exp, team2_reward_exp = self.calculate_expected_rewards(
            curr_bracket.team1.winner, 
            curr_bracket.team2.winner, 
            curr_bracket.playoff_round
        )
        
        # update the winner of bracket and state base on action 
        if action == 1: # p1 wins 
            curr_bracket.winner = curr_bracket.team1.winner
            self.state[curr_bracket.team2.winner] = 0
            self.total_reward += team1_reward
            self.discount *= p2
        else: 
            curr_bracket.winner = curr_bracket.team2.winner
            self.state[curr_bracket.team1.winner] = 0
            self.total_reward += team2_reward
            self.discount *= p1
            
        reward = self.total_reward * self.discount

        # update with the next win_probability
        if self.matchup_list:
            self.state[curr_bracket.winner] = self.data.loc[
                curr_bracket.winner, 
                f'rd{curr_bracket.playoff_round+1}_win'
            ]
        
        self.state_list = self.update_state_list()
        
        done = len(self.matchup_list) == 0
        info = {
            'round': curr_bracket.playoff_round,
            'team1': curr_bracket.team1.winner,
            'team2': curr_bracket.team2.winner,
            'winner': curr_bracket.winner,
            'matchups_left': len(self.matchup_list),
            'team1_exp_reward': team1_reward_exp, 
            'team2_exp_reward': team2_reward_exp,
        } 
        reward = reward if done else 0 
        return np.array(self.state_list), reward, done, info
    
    def get_current_matchup(self):
        curr_bracket = self.matchup_list[0]
        return curr_bracket.team1.winner, curr_bracket.team2.winner
    
    def close(self):
        pass 
    
    def render(self):
        pass 
    

