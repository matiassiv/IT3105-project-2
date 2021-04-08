import config as cfg
import numpy as np
import time
"""
First attempt to not use a node class to perform tree search
class Node:
    def __init__(self, parent=None, state=None):
        self.parent = parent
        self.state = state

        # Expand this when tree policy reaches this node
        self.children = []
        self.visited = 0
        self.value = 0
"""

class MCTS:
    def __init__(self, game, ann, eps=0.1):
        self.game = game
        self.ann = ann
        self.size = game.get_game_size()

        self.eps = eps
        self.max_games = cfg.mcts["max_games"]
        self.Qsa = {}  # stores Q values for edge s,a 
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times state s was visited
        #self.Ps = {}  # stores initial policy (returned by neural net)

        #self.Es = {}  # stores game.check_game_ended for state s
        #self.Vs = {}  # stores game.generate_legal_moves for state s

    def getActionProb(self, state):
        """
        Function to get the target distribution of action probabilities
        for a given state. Target distribution is generated by playing
        a series of rollout games from the given state. This is added to
        the replay buffer in trainer.py, and subsequently used for training
        the ann after enough games are played.
        """
        # Perform MCTS
        time_end = time.time() + 2 # Search games for x seconds
        while time.time() < time_end:
            self.search(state)
        
        # Get visit counts for all edges going out from the root state
        valids = self.game.generate_legal_moves(state)
        counts = [self.Nsa[(state, a)] if (state, a) in self.Nsa else 0 for a in range(len(valids))]
        #print(counts)
        sum_counts = np.sum(counts)
        #print(sum_counts)
        return [c / sum_counts for c in counts]

    def search(self, state):
        """
        Performs one rollout iteration of the MCTS.
        The function is called recursively using the tree policy
        until a leaf node or a terminal state is found.

        If a non-terminal leaf node is reached, then we perform a 
        rollout game from that leaf node using the default policy
        (the ann predictions) until we reach a final state. The 
        result is then backpropagated up the search path
        """

        # Check if the state is a terminal state
        game_result = self.game.check_game_ended(state)
        if game_result:
            # Return +1 if player 1 won the game, 
            # and return -1 if player 2 won the game
            return 1 if game_result == 1 else -1
        
        # Check if state is a leaf node
        # If it is, then perform rollout
        if state not in self.Ns:
            self.Ns[state] = 0
            return self.rollout(state)


        # Use tree policy to recursively search for a leaf node
        # First generate all legal moves from current state
        valids = self.game.generate_legal_moves(state)
        # Get the player turn from the state, as player 1 wants
        # maximize best_Qu, while player 2 wants to minimize it
        turn = 1 if state[0] == 1 else -1
        best_Qu = turn * -float('inf')
        best_act = -1

        self.Ns[state] += 1
        
        # Iterate over legal moves
        for i, a in enumerate(valids):
            if a:
                # Check if (s, a) in the Qsa dict
                if (state, i) in self.Qsa:
                    Qu = self.Qsa[(state, i)] + \
                        turn * np.sqrt(np.log(self.Ns[state])/(1 + self.Nsa[(state, i)]))
                else:
                    # Add small constant to not take log of 0
                    Qu = turn * np.sqrt(np.log(self.Ns[state]))
                
                # Update new best Qu and action based on player turn
                if turn == 1 and Qu > best_Qu:
                    best_Qu = Qu
                    best_act = i
                
                elif turn == -1 and Qu < best_Qu:
                    best_Qu = Qu
                    best_act = i
        s, a = state, best_act
        next_state = self.game.generate_next_state(s, self.game.one_hot_to_action(a))
        reward = self.search(next_state)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Qsa[(s, a)] * self.Nsa[(s, a)] + reward) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = reward
            self.Nsa[(s, a)] = 1
        
        return reward

        


    def rollout(self, state):
        
        while True:
            action = self.get_ann_action(state)
            state = self.game.generate_next_state(state, action)
        
            # Check if the state is a terminal state
            game_result = self.game.check_game_ended(state)
            if game_result:
                # Return +1 if player 1 won the game, 
                # and return -1 if player 2 won the game
                return 1 if game_result == 1 else -1


    
    def get_ann_action(self, state):
        # TODO add conversion from state to appropriate input here
        nn_input = self.ann.convert_state_to_input(state, self.size)
        preds = self.ann.forward(nn_input).detach().numpy().flatten()
        
        valids = self.game.generate_legal_moves(state)
        valid_preds = preds * valids # Mask illegal moves
        sum_Ps = np.sum(valid_preds)
        if sum_Ps > 0:
            valid_preds /= sum_Ps  # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable
            print("All valid moves were masked, nn may be overfitting.")
            valid_preds += valids
            valid_preds /= np.sum(valid_preds)
        
        r = np.random.rand()
        if r < self.eps:
            sum_p = np.sum(valids)
            action = np.random.choice(len(valid_preds), p=[v/sum_p for v in valids]) # Choose a random valid action
            return self.game.one_hot_to_action(action)
        else:
            action = np.argmax(valid_preds)
            return self.game.one_hot_to_action(action)


