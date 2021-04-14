import config as cfg
import torch
from NN_architectures.hex_ann import HexANN
from NN_architectures.hex_demo import HexDemo
from NN_architectures.hex_res_ann import HexResANN
from NN_architectures.nim_ann import NimANN
from torch import nn


class ANN():
    def __init__(self, input_size, output_size, ann_type="residual"):

        self.input_size = input_size
        self.output_size = output_size
        if ann_type == "nim":
            self.model = NimANN(input_size, output_size)
        elif ann_type == "residual":
            self.model = HexResANN(input_size, output_size)
        elif ann_type == "hex_5":
            self.model = HexANN(input_size, output_size)
        elif ann_type == "hex_demo":
            self.model = HexDemo(input_size, output_size)
        

    def forward(self, x):
        return self.model.forward(x)
    
    def train_step(self, loss_fn, optimizer, batch):
        return self.model.train_step(loss_fn, optimizer, batch)

    def convert_state_to_input(self, state, size):
        return self.model.convert_state_to_input(state, size)

    def get_output_shape(self):
        return self.model.get_output_shape()