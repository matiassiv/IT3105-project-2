import matplotlib.pyplot as plt
from torch import nn
import torch
import numpy as np
import random
from state_manager import StateManager
from mcts import MCTS
from ann import ANN
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
optimizer = torch.optim.Adam(ann.model.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=15, factor=0.5, eps=5e-5)  # Reduce learning rate by a factor of 2 if loss doesn't decrease after 15 iterations
print(ann.model)
replay_buffer = []
accs = []
losses = []
results = {}
results[1] = [0, 0]
results[2] = [0, 0]
turn = 1
i = 0
while i <= 350:
    if len(losses) > 0:
        print(i, len(replay_buffer), losses[-1], accs[-1], flush=True)
    else:
        print(i, len(replay_buffer), flush=True)
    if i % 50 == 0:
        torch.save(ann.model.state_dict(),
                   "trained_models/hex_5_run_2/iteration_"+str(i)+".pt")
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

    if len(replay_buffer) > 64:
        batch = random.sample(replay_buffer, 64)
        loss, acc = ann.train_step(loss_fn, optimizer, batch)
        losses.append(loss)
        accs.append(acc)
        scheduler.step(loss)
        i += 1
        if len(replay_buffer) > 1000:
            # Remove early games from buffer
            replay_buffer = replay_buffer[100:]
    turn = turn % 2 + 1
    game = StateManager(turn)
    s = game.get_game_state()
    m = MCTS(game, ann)

print(results)
x = np.arange(len(losses))
plt.plot(x, losses)
plt.show()
plt.clf()
plt.plot(x, accs)
plt.show()
