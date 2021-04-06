from game_representations.hex_game import HexGame
from game_representations.nim import Nim
import config as cfg

class StateManager:
    def __init__(self):

        if cfg.state_manager["game_type"] == "nim":
            self.game = Nim(
                cfg.nim_settings["N"],
                cfg.nim_settings["K"],
                cfg.state_manager["turn"],
                cfg.state_manager["display_game"]
            )
        elif cfg.state_manager["game_type"] == "hex":
            self.game = HexGame(
                size=cfg.hex_settings["size"],
                turn=cfg.state_manager["turn"],
                graphing_freq=cfg.hex_settings["graphing_freq"],
                display_game=cfg.state_manager["display_game"]
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


if __name__ == "__main__":
    s = StateManager()
    while s.game.game_result == 0:
        print("Player", s.game.turn)
        print("Valid moves:", s.generate_legal_moves(s.get_game_state()))
        move = int(input("Input move: "))
        s.update_game_state(move)
