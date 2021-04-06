from ann import ANN
from mcts import MCTS
from state_manager import StateManager
import numpy as np
import torch
from torch import nn

"""
Function for the entire RL with ANN and MCTS 
"""

game = StateManager()
s = game.get_game_state()

input_size = len(game.get_game_state())
output_size = len(game.generate_legal_moves(game.get_game_state()))
ann = ANN(input_size, output_size)
m = MCTS(game, ann)
loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
optimizer = torch.optim.Adam(ann.parameters(), lr=0.005)
print(ann)
"""
states = []
targets = []
for i in range(15000):
    j = np.random.randint(5)
    states.append(((j+1)/5,))
    targets.append([1 if t == j else 0 for t in range(5)])

assert len(states) == len(targets)
for i in range(0, 15000):
    ann.train_step(loss_fn, optimizer, states[i], targets[i])
"""
results = [0,0]
for i in range(350):
    states = []
    targets = []
    while True:
        #print(s)
        action_prob = m.getActionProb(s)
        #print(action_prob)
        states.append(s)
        targets.append(action_prob)
        action = np.argmax(action_prob)
        a = game.one_hot_to_action(action)
        s = game.generate_next_state(s, a)
        result = game.check_game_ended(s)
        if result:
            results[result-1] += 1
            break
    
    ann.train_step(loss_fn, optimizer, states, targets)
    
    game = StateManager()
    s = game.get_game_state()
    m = MCTS(game, ann)

print(results)