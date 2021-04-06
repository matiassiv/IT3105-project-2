class Nim:
    def __init__(self, N=50, K=5, turn=1, display_game=True):
        self.remaining = N
        self.K = K
        self.turn = turn
        self.game_result = 0
        self.display_game = display_game
        if self.display_game:
            print("######################################")
            print("N:", N)
            print("K:", K)
            print("Player to start:", turn)
            print("--------------------------------------")

    def generate_legal_moves(self, state):
        # State is of the form (turn, remaining)
        # Returns a one-hot encoded array over legal moves for the given state
        return [1 if k <= state[1] else 0 for k in range(1, self.K+1)]

    def one_hot_to_action(self, one_hot_index):
        # Converts selected action from one-hot-encoded output from 
        # the ann back to an action that is understandable for the game
        return one_hot_index + 1

    def generate_next_state(self, state, action):
        new_turn = state[0] % 2 + 1
        new_remaining = state[1] - action
        new_state = (new_turn, new_remaining)
        return new_state

    def get_game_state(self):
        return (self.turn, self.remaining)

    # Legacy functions, heh
    def update_game_state(self, move):
        self.remaining -= move
        self.game_result = self.check_victory_condition()
        if self.display_game:
            print("Player", self.turn, "removes", move)
            print("There are", self.remaining, "stones remaining.")

        self.turn = self.turn % 2 + 1

    def check_game_ended(self, state):
        if state[1] == 0:
            winner = state[0] % 2 + 1
            if self.display_game:
                print("Player", winner, "won the game!")
            return winner
        return 0

if __name__ == "__main__":
    n = Nim()
    print(n.K, n.remaining)
    print(n.generate_legal_moves((1, 4)))