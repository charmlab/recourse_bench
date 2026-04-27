from __future__ import division

import torch.nn as nn

from method.clue.library.clue_ml.src.layers import (
    MLPBlock,
    leaky_MLPBlock,
    preact_leaky_MLPBlock,
)


class MLP_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_recognition_net, self).__init__()
        proposal_layers = [
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        for _ in range(depth - 1):
            proposal_layers.append(MLPBlock(width))
        proposal_layers.append(nn.Linear(width, latent_dim * 2))
        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_generator_net, self).__init__()
        generative_layers = [
            nn.Linear(latent_dim, width),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        for _ in range(depth - 1):
            generative_layers.append(leaky_MLPBlock(width))
        generative_layers.append(nn.Linear(width, input_dim))
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)


class MLP_preact_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_recognition_net, self).__init__()
        proposal_layers = [nn.Linear(input_dim, width)]
        for _ in range(depth - 1):
            proposal_layers.append(preact_leaky_MLPBlock(width))
        proposal_layers.extend(
            [
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=width),
                nn.Linear(width, latent_dim * 2),
            ]
        )
        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_preact_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_generator_net, self).__init__()
        generative_layers = [
            nn.Linear(latent_dim, width),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        for _ in range(depth - 1):
            generative_layers.append(preact_leaky_MLPBlock(width))
        generative_layers.append(nn.Linear(width, input_dim))
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)
