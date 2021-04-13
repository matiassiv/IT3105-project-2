import config as cfg
import torch
from torch import nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, incoming=64, outgoing=64, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=padding)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=padding)
        self.bn2 = nn.BatchNorm2d(outgoing)

    def forward(self, x):
        # Save input for skip connection
        identity = x

        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)

        # Add back skipped input
        out += identity
        out = self.lrelu(out)

        return out


class HexResANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(HexResANN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        #self.res1 = ResBlock(64, 64, 1)
        #self.res2 = ResBlock(64, 64, 1)
        #self.res3 = ResBlock(64, 64, 1)
        self.flatten = nn.Flatten()
        """
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )
        """
        conv_output_size = self.get_output_shape()

        self.output = nn.Sequential(
            nn.Linear(conv_output_size[1], 400),
            nn.ReLU(),
            nn.Linear(400, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):

        x = self.input_conv(x)
        #x = self.res1(x)
        #x = self.res2(x)
        #x = self.res3(x)
        x = self.flatten(x)
        #x = self.reduce_conv(x)
        x = self.output(x)
        return x

    def train_step(self, loss_fn, optimizer, batch):

        X, y = zip(*batch)
        X = torch.squeeze(torch.stack(X))
        # Predict action probabilities
        pred = self.forward(X)
        # print(pred)

        # Get prediction loss and perform backprop
        y = torch.tensor(y, dtype=torch.float)
        # print(y)
        loss = loss_fn(pred, y)
        #print("LOSS:", loss)
        # print("-----------------------------------------------------------------")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = self.accuracy(pred, y)
        return loss, acc

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
        x = self.input_conv(nn_input)
        #x = self.res1(x)
        #x = self.res2(x)
        #x = self.res3(x)
        x = self.flatten(x)
        return x.data.shape

    def accuracy(self, preds, targets):

        # Mask invalid moves
        masked_preds = preds
        masked_preds[targets == 0] = 0

        targets_max = torch.argmax(targets, dim=1)
        preds_max = torch.argmax(masked_preds, dim=1)

        return torch.eq(targets_max, preds_max).sum() / targets.size(0)
