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

    
    def play_match(self, model_1, model_2):
        for i in range(self.num_games):
            pass
    
    def play_game(self, p1, p2, starting_player=1):
        game = StateManager(starting_player)
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
