import matplotlib.pyplot as plt
from torch import nn
import torch
import numpy as np
import random
from state_manager import StateManager
from mcts import MCTS
from ann import ANN

"""
Function for the entire RL with ANN and MCTS 
"""


def loss_p(outputs, targets):
    return -torch.sum(targets * torch.log(1e-9 + outputs)) / targets.size()[0]


game = StateManager()
s = game.get_game_state()
size = game.get_game_size()
input_size = size
output_size = len(game.generate_legal_moves(game.get_game_state()))
ann = ANN(input_size, output_size)
m = MCTS(game, ann)
#loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
loss_fn = loss_p
pred = torch.tensor([0.2, 0.4, 0.3, 0.1])
target = torch.tensor([0.2, 0.4, 0.3, 0.1])
print(loss_fn(pred, target))
optimizer = torch.optim.Adam(ann.model.parameters(), lr=0.005)
print(ann.model)
replay_buffer = []
losses = []
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
results = {}
results[1] = [0, 0]
results[2] = [0, 0]
turn = 1
for i in range(2500):
    if i % 50 == 0:
        print(i)
    while True:
        action_prob = m.getActionProb(s)
        replay_buffer.append(
            (ann.convert_state_to_input(s, size), action_prob))
        action = np.argmax(action_prob)
        a = game.one_hot_to_action(action)
        s = game.generate_next_state(s, a)
        result = game.check_game_ended(s)
        if result:
            results[turn][result-1] += 1
            break

    # Get random minibatch
    # TODO søk opp np random choice for å hente ut en tilfeldig minibatch av en array på formen
    # [(tensor, np.array)]
    if len(replay_buffer) > 64:
        batch = random.sample(replay_buffer, 64)
        loss = ann.train_step(loss_fn, optimizer, batch)
        losses.append(loss)
        if len(replay_buffer) > 2500:
            # Remove early games from buffer
            replay_buffer = replay_buffer[1250:]
    turn = turn % 2 + 1
    game = StateManager(turn)
    s = game.get_game_state()
    m = MCTS(game, ann)

print(results)
x = np.arange(len(losses))
plt.plot(x, losses)
plt.show()
