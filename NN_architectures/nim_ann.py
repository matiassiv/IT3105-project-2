import config as cfg
import torch
from torch import nn
import numpy as np
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

"""
TODO Add support for also conv-layers in the network
"""

class NimANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NimANN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        """
        self.conv_layers = nn.Sequential(nn.Conv1d(1,64,1))
        self.conv_layers2 = nn.Sequential(
            nn.Conv1d(input_size, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_output_size = self.get_output_shape()
        """
        conv_output_size = input_size
        self.output = nn.Sequential(
            nn.Linear(conv_output_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        #x = self.conv_layers(x)
        #x = torch.tensor(x, dtype=torch.float)
        probs = self.output(x)
        return probs
    
    def train_step(self, loss_fn, optimizer, batch):
        X, y = zip(*batch)
        X = torch.stack(X)
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
        remaining = state[1]
        nn_input = [1 if i < remaining else 0 for i in range(size)]
        return torch.tensor(nn_input, dtype=torch.float)

    def get_output_shape(self):
        print(torch.rand(1, self.input_size, 1))
        print("helluy")
        print(self.conv_layers(torch.rand(1, self.input_size, 1)))
        print("hei")
        return self.conv_layers(torch.rand(self.input_size)).data.shape