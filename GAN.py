import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    """
    A GRU based generator that takes in a noise sequence and returns a synthetic sequence.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Generator, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Tanh())

    def forward(self, noise: torch.Tensor):
        "Input: noise of shape (batch_size, seq_len, num_of_features)"

        rnn_out, _ = self.rnn(noise)
        return self.output_layer(rnn_out)

class Discriminator(nn.Module):
    """
    A GRU based discriminator that takes in a sequence as input and returns the likelihood of it being synthetic or real.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(Discriminator, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, sequences: torch.Tensor):
        rnn_output, _ = self.rnn(sequences)
        return self.output_layer(rnn_output)
