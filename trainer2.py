import matplotlib.pyplot as plt
from torch import nn
import torch
import numpy as np
import random
from state_manager import StateManager
from mcts import MCTS
from ann import ANN
"""
Duplicate installs of some library on old computer, so need to set this
flag to use it for training
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Function for the entire RL with ANN and MCTS
"""


def loss_p(outputs, targets):
    return -torch.sum(targets * torch.log(1e-9 + outputs)) / targets.size()[0]


search_time = 3.6
game = StateManager()
s = game.get_game_state()
s0 = s
size = game.get_game_size()
input_size = size
output_size = len(game.generate_legal_moves(game.get_game_state()))
ann = ANN(input_size, output_size)
m = MCTS(game, ann, search_time=search_time)
temp = 1
# loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
loss_fn = loss_p
optimizer = torch.optim.Adam(ann.model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=10, factor=0.2, eps=5e-6)  # Reduce learning rate by a factor of 5 if loss doesn't decrease after 15 iterations
print(ann.model)
replay_buffer = []
accs = []
losses = []
results = {}
results[1] = [0, 0]
results[2] = [0, 0]
turn = 1
i = 0
path = "trained_models/multiple_iterations/"
while i <= 150:
    ann.model.eval()
    if len(losses) > 0:
        print(i, len(replay_buffer), losses[-1], accs[-1], flush=True)
    else:
        print(i, len(replay_buffer), flush=True)
    if i % 25 == 0:
        torch.save(ann.model.state_dict(),
                   path+"iteration_"+str(i)+".pt")
    temp = 1 + i // 20
    while True:
        action_prob = m.getActionProb(s)
        """
        r = np.random.rand()
        if r < 0.01:
            with torch.no_grad():
                print("--------------------------------------")
                print("action:", np.argmax(action_prob), "p:", max(action_prob), flush=True)
                nn_o = ann.forward(ann.convert_state_to_input(s, size)).detach().numpy().flatten()
                valids = game.generate_legal_moves(s)
                nn_o = valids * nn_o
                sum_Ps = np.sum(nn_o)
                if sum_Ps > 0:
                    nn_o /= sum_Ps  # renormalize
                print("action:", np.argmax(nn_o), "p:", max(nn_o))
        """
        replay_buffer.append(
            (ann.convert_state_to_input(s, size), tuple(action_prob)))
        # Also rotate state to create more training samples
        rotated = list(s)
        rotated[1:] = rotated[1:][::-1]
        rotated_action = action_prob[::-1]
        replay_buffer.append(
            (ann.convert_state_to_input(rotated, size), tuple(rotated_action))
        )

        # Become more exploitative throughout the training
        if temp > 3:
            action = np.argmax(action_prob)
        elif temp > 1:
            action_prob = action_prob ** (temp)
            sum_p = np.sum(action_prob)
            action_prob /= sum_p
            action = np.random.choice(
                len(action_prob), p=action_prob)
        else:
            action = np.random.choice(
                len(action_prob), p=action_prob)
        a = game.one_hot_to_action(action)
        s = game.generate_next_state(s, a)
        result = game.check_game_ended(s)
        if result:
            results[turn][result-1] += 1
            break

    # Get random minibatch

    if len(replay_buffer) > 300:
        ann.model.train()
        for j in range(2):
            batch = random.sample(replay_buffer, 128)
            loss, acc = ann.train_step(loss_fn, optimizer, batch)
            print(loss, acc, flush=True)
        losses.append(loss)
        accs.append(acc)
        scheduler.step(loss)
        
        i += 1
        buffer_size = len(replay_buffer)
        #Throw away the earliest training data
        if i < 3:
            replay_buffer = []
        if buffer_size>2500:
            replay_buffer = replay_buffer[72:]
        
    turn = turn % 2 + 1
    game = StateManager(turn)
    s = game.get_game_state()
    m = MCTS(game, ann, search_time=search_time)

print(results)
x = np.arange(len(losses))
plt.plot(x, losses)
plt.savefig(path+"losses.png")
plt.clf()
plt.plot(x, accs)
plt.savefig(path+"accs.png")
plt.clf()
