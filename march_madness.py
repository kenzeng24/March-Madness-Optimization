# import gym
import pandas as pd 


def make_teams(data):
    """
    create a team dictionary from data 
    """
    teams = {}
    regions = data['team_region'].drop_duplicates()
    seeds = data['team_seed'].drop_duplicates()
    
    for region in regions:
        teams[region] = {}
        for seed in seeds:
            if not data.loc[(data['team_region']==region) & (data['team_seed']==seed), :].empty:
                teams[region][seed] = data.loc[(data['team_region']==region) & (data['team_seed']==seed)].index[0]
        play_in_keys = [key for key in teams[region].keys() if 'a' in key or 'b' in key]
        play_in_teams = [teams[region][key] for key in teams[region].keys() if 'a' in key or 'b' in key]
        play_in_key = play_in_keys[0][:-1]
        teams[region][play_in_key] = play_in_teams
            
    return teams
    

def example_reward_function(seed, playoff_round):
    modifier = playoff_round-1 if playoff_round != 7 else 10 
    return int(seed.replace('a','').replace('b','')) * modifier


class Bracket:
    
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
    
    def __init__(self, data=None):
        self.matchup_list = [] 
        self.data = data
        
        if self.data is None:
            # defaults to 2023 
            df_538 = pd.read_csv('data/fivethirtyeight_ncaa_forecasts.csv')
            self.data = df_538[
                (df_538.forecast_date=='2023-03-12') & 
                (df_538.gender=='mens')
            ].set_index('team_name')
        
        self.teams_list = self.data.index.sort_values().to_list()
        self.teams = make_teams(self.data)
        self.reset()
    
    def reset(self):
        """
        regenerate brackets and refresh all winners
        """
        self.matchup_list = []
        self.state = {}
        self.bracket = self.build_bracket(self.complete_bracket(self.data))
        self.depth = self.bracket.playoff_round
        self.matchup_in_round = {i:[] for i in range(self.depth+1)}
        self.update_states(self.bracket, self.depth)
        self.state_list = [0 for team in self.teams_list]
        self.state_list = self.update_state_list()

    def update_state_list(self):
        for team, prob in self.state.items():
            idx = self.teams_list.index(team)
            if prob != 0:
                self.state_list[idx] = 1
            else:
                self.state_list[idx] = 0
        return self.state_list

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
            
    def make_region(self, region):
        """
        build lists of matchups for a particular region 
        """
        teams = self.teams
        bracket = [
            [
                [
                    [teams[region]['1'], teams[region]['16']], 
                    [teams[region]['8'], teams[region]['9']], 
                ], 
                [
                    [teams[region]['5'], teams[region]['12']], 
                    [teams[region]['4'], teams[region]['13']], 
                ],
            ], 
            [
                [
                    [teams[region]['6'], teams[region]['11']], 
                    [teams[region]['3'], teams[region]['14']], 
                ], 
                [
                    [teams[region]['7'], teams[region]['10']], 
                    [teams[region]['2'], teams[region]['15']], 
                ],
            ]
        ]
        return bracket


    def complete_bracket(self, data):
        return [
            [self.make_region('South'), self.make_region('East')], 
            [self.make_region('Midwest'), self.make_region('West')]
        ]
        
        
    def build_bracket(self, matchups):
        """
        recursively build bracket object and
        generate a list of all matchups 
        """
        if type(matchups) is str:
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
    
    def calculate_rewards(self, team1, team2, playoff_round, reward_function=example_reward_function):
        """
        given two teams calculate the expected reward 
        using win probabilities
        """
        p1 = self.data.loc[team1, f'rd{playoff_round}_win']
        p2 = self.data.loc[team2, f'rd{playoff_round}_win']
        
        seeding1 = self.data.loc[team1, 'team_seed']
        seeding2 = self.data.loc[team2, 'team_seed']
        
        team1_reward = p1 / (p1+p2) * reward_function(seeding1, playoff_round)
        team2_reward = p2 / (p1+p2) * reward_function(seeding2, playoff_round)
        return team1_reward, team2_reward
    
    
    def calculate_next_matchup_rewards(self):
        """get the expected reward for the next matchup"""
        curr_bracket = self.matchup_list[0]
        team1_reward, team2_reward = self.calculate_rewards(
            curr_bracket.team1.winner, 
            curr_bracket.team2.winner, 
            curr_bracket.playoff_round
        )
        return team1_reward, team2_reward
        
    
    def step(self, action):
        """
        pick winner and update bracket
        """
        if len(self.matchup_list) == 0:
            raise ValueError('Matchups are complete')
            
        # get next available matchup 
        curr_bracket = self.matchup_list.pop(0)
        
        # calculate expected rewards given win probabilities
        team1_reward, team2_reward = self.calculate_rewards(
            curr_bracket.team1.winner, 
            curr_bracket.team2.winner, 
            curr_bracket.playoff_round
        )
        # update the winner of bracket and state base on action 
        if action == 1:
            curr_bracket.winner = curr_bracket.team1.winner
            self.state[curr_bracket.team2.winner] = 0
            reward = team1_reward
        else: 
            curr_bracket.winner = curr_bracket.team2.winner
            self.state[curr_bracket.team1.winner] = 0
            reward = team2_reward

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
            'matchup': {
                curr_bracket.team1.winner: round(team1_reward,3), 
                curr_bracket.team2.winner: round(team2_reward,3)
            },
            'winner': curr_bracket.winner,
            'matchups_left': len(self.matchup_list),
        } 
        return self.state, reward, done, info
    
    def close(self):
        pass 
    
    def render(self):
        pass 
    

def greedy_strategy(march_madness_event, verbose=True):
    """
    always pick the action that leads to best immediate reward
    """
    total_reward = 0 
    done = False 
    while not done:
        reward1, reward2 = march_madness_event.calculate_next_matchup_rewards()
        action = 1*(reward1 > reward2)
        state, reward, done, info = march_madness_event.step(action)
        if verbose:
            print(info)
        total_reward += reward
    return total_reward


def brute_force_strategy(march_madness_event):
    """
    iterate through all available sequences of actions 
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
    
    best_reward = 0 
    for action_list in action_lists:
        env.reset()
        total_reward = 0 
        for action in action_list:
            state, reward, done, info = env.step(action)
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
    return best_reward
        
            
if __name__ == "__main__":
    
    env = MarchMadnessEnvironment()
    for x in env.matchup_list:
        print(x)
    print(env.matchup_in_round)
    
    greedy_score = greedy_strategy(env)
    print(f'greedy reward: {greedy_score}') 
    
    #optimal_score =  brute_force_strategy(env)
    # print(f'optimal reward: {optimal_score}')
    