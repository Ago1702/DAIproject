import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class QNet(nn.Module):
    '''
    Class Qnet: Qualcosa di spicy
    '''

    def __init__(self, obs_dim:int, hidden_num:int=1, hidden_dim:int=64):
        super(QNet, self).__init__()

        self.obs_dim = obs_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim

        self.input = nn.Linear(obs_dim, hidden_dim)

        hidden = [nn.ReLU()]
        for i in range(hidden_num):
            hidden.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        self.hidden = nn.Sequential(*hidden)

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.input(X)
        X = self.hidden(X)
        X = self.output(X)
        return X


class ActorNet(nn.Module):
    '''
    Class Qnet: Qualcosa di spicy
    '''

    def __init__(self, obs_dim:int, out_dim:int, hidden_num:int=1, hidden_dim:int=64):
        super(ActorNet, self).__init__()

        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim

        self.input = nn.Linear(obs_dim, hidden_dim)

        hidden = [nn.ReLU()]
        for i in range(hidden_num):
            hidden.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        self.hidden = nn.Sequential(*hidden)

        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.input(X)
        X = self.hidden(X)
        X = self.output(X)
        return X