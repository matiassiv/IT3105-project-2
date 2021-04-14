from game_representations.hex_game import HexGame
from game_representations.nim import Nim
import config as cfg

class StateManager:
    def __init__(
        self, 
        turn=0,  
        game_params=cfg.TRAIN_game_params, 
        game_size=0, 
        display_game=False):
        starting_player = 1
        if turn:
            starting_player = turn
        if game_size:
            size = game_size
        else:
            size = game_params["size"]
        
        if game_params["game_type"] == "nim":
            self.game = Nim(
                game_params["N"],
                game_params["K"],
                starting_player,
                display_game
            )
        elif game_params["game_type"] == "hex":
            self.game = HexGame(
                size=size,
                turn=starting_player,
                graphing_freq=game_params["graphing_freq"],
                display_game=display_game
            )
    
    def generate_legal_moves(self, state):
        return self.game.generate_legal_moves(state)
    
    def generate_next_state(self, state, action):
        return self.game.generate_next_state(state, action)
    
    def get_game_state(self):
        return self.game.get_game_state()
    
    def update_game_state(self, action):
        # This is for updating the actual internal board state
        # which makes the graphing easier.
        self.game.update_game_state(action)
    
    def check_game_ended(self, state):
        return self.game.check_game_ended(state)

    def one_hot_to_action(self, one_hot_index):
        return self.game.one_hot_to_action(one_hot_index)
    
    def get_game_size(self):
        return self.game.get_game_size()


if __name__ == "__main__":
    s = StateManager()
    while s.game.game_result == 0:
        print("Player", s.game.turn)
        print("Valid moves:", s.generate_legal_moves(s.get_game_state()))
        move = int(input("Input move: "))
        s.update_game_state(move)
