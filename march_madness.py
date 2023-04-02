# import gym
import pandas as pd 

play_in_south=  ["Texas A&M-Corpus Christi", 'Southeast Missouri State']
play_in_east = ["Texas Southern", 'Fairleigh Dickinson',]
play_in_west = ["Arizona State", "Nevada"]
play_in_mid_west = ['Mississippi State','Pittsburgh']

play_in_south=  ["Texas A&M-Corpus Christi", 'Southeast Missouri State']
play_in_east = ["Texas Southern", 'Fairleigh Dickinson',]
play_in_west = ["Arizona State", "Nevada"]
play_in_mid_west = ['Mississippi State','Pittsburgh']

south_bracket = [
    [
        [
            ['Alabama', play_in_south], # play_in_south
            ["Maryland", "West Virginia"]
        ], 
        [
            ['San Diego State', 'College of Charleston'], 
            ['Virginia', 'Furman']
        ],
    ], 
    [
        [
            ['Creighton', "North Carolina State"], 
            ["Baylor", "UCSB"]
        ], 
        [
            ['Missouri', 'Utah State',], 
            ['Arizona', 'Princeton']
        ],
    ]
]

east_bracket = [
    [
        [
            ['Purde', play_in_east], 
            ["Memphis", 'Florida Atlantic']
        ], 
        [
            ['Duke', 'Oral Roberts'], 
            ['Tennessee', 'Louisiana']
        ],
    ], 
    [
        [
            ['Kentucky', "Providence"], 
            ["Kansas State", "Montana State"]
        ], 
        [
            ['Michigan State', 'Southern California',], 
            ['Marquette', 'Vermont']
        ],
    ]
    
]

# TODO: fill in north and west brackets
north_bracket = []
west_bracket = [] 

final = [[south_bracket, east_bracket], [north_bracket, west_bracket]]



example_bracket = [
    [
        ['Alabama', "Houston"], 
        ["Maryland", "Purdue"]
    ], 
    [
        ['San Diego State', 'Kansas'], 
        ['Virginia', 'Princeton']
    ],
]

class Bracket:
    
    def __init__(self, winner=None, team1=None, team2=None,playoff_round=None):
        self.winner = winner
        self.playoff_round = playoff_round
        self.team1 = team1 
        self.team2 = team2

        
class MarchMadnessEnvironment():
    
    def __init__(self, data=None, matchups=None):
        self.matchup_list = [] 
        self.data = data
        self.matchups = matchups
        if self.matchups is None:
            self.matchups = example_bracket
        if self.data is None:
            # defaults to 2023 
            df_538 = pd.read_csv('data/fivethirtyeight_ncaa_forecasts.csv')
            self.data = df_538[
                (df_538.forecast_date=='2023-03-12') & 
                (df_538.gender=='mens')
            ].set_index('team_name')
        self.reset()
    
    def reset(self):
        """
        regenerate brackets and refresh all winners
        """
        self.matchup_list = []
        self.state = {}
        self.bracket = self.build_bracket(self.matchups)
        self.depth = self.bracket.playoff_round
        self.update_states(self.bracket, self.depth)
    
    
    def update_states(self, bracket, n):
        """
        update each bracket with their actual depth
        """
        if bracket is not None:
            if bracket.team1 is None and bracket.team2 is None:
                self.state[bracket.winner] = self.data.loc[bracket.winner, f'rd{n+1}_win']
            bracket.playoff_round = n
            self.update_states(bracket.team1, n-1)
            self.update_states(bracket.team1, n-1)
        
        
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
    
    def calculate_rewards(self, team1, team2, playoff_round):
        """
        given two teams calculate the expected reward 
        using win probabilities
        """
        p1 = self.data.loc[team1, f'rd{playoff_round}_win']
        p2 = self.data.loc[team2, f'rd{playoff_round}_win']
        
        seeding1 = self.data.loc[team1, 'team_seed']
        seeding2 = self.data.loc[team2, 'team_seed']
        
        # account for playoff teams
        if seeding1 == '16a':
            seeding1 = '16'
        if seeding2 == '16a':
            seeding2 = '16'
        seeding1, seeding2 = int(seeding1), int(seeding2)
        
        team1_reward = p1 / (p1+p2) * seeding1 * playoff_round
        team2_reward = p2 / (p1+p2) * seeding2 * playoff_round
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
        
        done = len(self.matchup_list) == 0
        info = {
            'matchups_left': len(self.matchup_list),
            'winner': curr_bracket.winner,
            'round': curr_bracket.playoff_round,
            'matchup': {
                curr_bracket.team1.winner: round(team1_reward,3), 
                curr_bracket.team2.winner: round(team2_reward,3)
            },
        } 
        return self.state, reward, done, info
    
    def close(self):
        pass 
    
    def render(self):
        pass 
    
def main():
    env = MarchMadnessEnvironment()
    done = False 
    while not done:
        state, reward, done, info = env.step(action=1)
        print(info)
        
        
if __name__ == "__main__":
    main() 