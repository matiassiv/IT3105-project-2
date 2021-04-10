import torch
import config as cfg
from state_manager import StateManager
from mcts import MCTS
from NN_architectures.hex_ann import HexANN

class TOPP:
    def __init__(self, num_contenders, num_games, contender_paths):
        self.num_contenders = num_contenders
        self.num_games = num_games

        # Get game info to properly load model
        game = StateManager()
        input_size = game.get_game_size()
        output_size = len(game.generate_legal_moves(game.get_game_state()))
        
        # Create a list of contender models
        self.contenders = []

        for path in contender_paths:
            model = HexANN(input_size, output_size)
            model.load_state_dict(torch.load(path))
            model.eval()
            self.contenders.append(model)

    def create_tournament_matchings(self):
        """
        Returns an array of tuples with indices to the contender list
        representing the matchings for the tournament
        """
        matchings = []
        for i in range(self.num_contenders):
            for j in range(i):
                matchings.append((i, j))
        
        return matchings

    def play_tournament(self):
        matchings = self.create_tournament_matchings()

        # Create a dictionary to hold the tournament results
        results = {}
        for i in range(self.num_contenders):
            results[i] = [0 for i in range(self.num_contenders)]
        
        for match in matchings:
            model_1 = self.contenders[match[0]]
            model_2 = self.contenders[match[1]]
            m1_results, m2_results = self.play_match(model_1, model_2)

            if m1_results > m2_results:
                results[match[0]][match[1]] = 1
                results[match[1]][match[0]] = -1
            elif m2_results > m1_results:
                results[match[1]][match[0]] = 1
                results[match[0]][match[1]] = -1
            # If match is drawn, then the initialised 0 remains
        return results

    def play_match(self, model_1, model_2):
        m1_results = 0
        m2_results = 0
        halfway_point = self.num_games//2
        # Model 1 starts all matches in first half
        for i in range(halfway_point):
            # Alternate colors to start to test more of the network
            starting_color = i % 2 + 1
            result = self.play_game(model_1, model_2, starting_color)

            # Model 1 is starting player, so if starting color wins then model 1 wins
            if result == starting_color:
                m1_results += 1
                #m1_results.append(1)
                #m2_results.append(-1)
            else:
                m2_results += 1
                #m1_results.append(-1)
                #m2_results.append(1)
        
        for i in range(halfway_point):
            # Alternate colors to start to test more of the network
            starting_color = i % 2 + 1
            result = self.play_game(model_2, model_1, starting_color)

            # Model 2 is starting player, so if starting color wins then model 2 wins
            if result == starting_color:
                m2_results += 1
                #m1_results.append(-1)
                #m2_results.append(1)
            else:
                m1_results += 1
                #m1_results.append(1)
                #m2_results.append(-1)
        
        return m1_results, m2_results
    
    def play_game(self, p1, p2, starting_color=1):
        game = StateManager(starting_color)
        m1 = MCTS(game, p1, 0)
        m2 = MCTS(game, p2, 0)
        s = game.get_game_state()
        while True:
            # TODO add actual game update for potential display of game
            action = m1.getActionProb(s)
            a = game.one_hot_to_action(action)
            s = game.generate_next_state(s, a)
            result = game.check_game_ended(s)
            if result:
                return result
            action = m2.getActionProb(s)
            a = game.one_hot_to_action(action)
            s = game.generate_next_state(s, a)
            result = game.check_game_ended(s)
            if result:
                return result

    def print_tournament_results(self, results):

        total_scores = [sum(results[i]) for i in range(self.num_contenders)]
        
        for i in range(self.num_contenders):
            print(f"Contender {i}")
            print("Results:", results[i])
            print("Total score:", total_scores[i])
            print("-----------------------------------------------")
        
