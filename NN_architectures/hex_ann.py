import config as cfg
import torch
from torch import nn
import numpy as np

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class HexANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(HexANN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )
        conv_output_size = self.get_output_shape()

        self.output = nn.Sequential(
            nn.Linear(conv_output_size[1], output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
       
        x = self.conv_layers(x)
        probs = self.output(x)
        return probs
    
    def train_step(self, loss_fn, optimizer, batch):

        X, y = zip(*batch)
        X = torch.squeeze(torch.stack(X))
        # Predict action probabilities
        pred = self.forward(X)
        print(pred)

        # Get prediction loss and perform backprop
        y = torch.tensor(y, dtype=torch.float)
        print(y)
        loss = loss_fn(pred, y)
        print("LOSS:", loss)
        print("-----------------------------------------------------------------")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def convert_state_to_input(self, state, size):
        # Trenger en fornuftig encoding for å best mulig representere 
        # brettet i hex. En mulighet er å lage et slags bilde med tre kanaler
        # 1. kanal: plasseringen av alle steinene til player 1
        # 2. kanal: plasseringen av alle steinene til player 2
        # 3. kanal: hvem sin tur det er på brettet
        turn = state[0]
        player_1 = torch.tensor(
            [[1 if state[i*size + j + 1] == 1 else 0 for j in range(size)] for i in range(size)])
        player_2 = torch.tensor(
            [[1 if state[i*size + j + 1] == 2 else 0 for j in range(size)] for i in range(size)])
        turn_plane = np.ones((size, size))
        if turn == 2:
            turn_plane *= -1
        turn_plane = torch.tensor(turn_plane, dtype=torch.float)
        return torch.unsqueeze(torch.stack((player_1, player_2, turn_plane)), 0)





    def get_output_shape(self):
        state = [1] + [0 for i in range(self.input_size**2)]
        nn_input = self.convert_state_to_input(state, self.input_size)

        return self.conv_layers(nn_input).data.shape