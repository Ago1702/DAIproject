import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class QNet(nn.Module):
    '''
    Class Qnet: Qualcosa di spicy
    '''

    def __init__(self, obs_dim:int, hidden_num:int=1, hidden_dim:int=128):
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

    def __init__(self, obs_dim:int, out_dim:int, hidden_num:int=1, hidden_dim:int=128,
                  _min:int|torch.Tensor=None, _max:int|torch.Tensor=None, eps:float = 0):
        super(ActorNet, self).__init__()

        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.register_buffer("_min", _min)
        self.register_buffer("_max", _max)
        self.register_buffer("_noise", torch.Tensor(self.out_dim).unsqueeze(0))
        self.noise:torch.Tensor = self._noise

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
        X = F.sigmoid(X)
        X = X * (self._max - self._min) - self._min
        return X

    def noisy_forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.input(X)
        X = self.hidden(X)
        X = self.output(X)
        self.noise.normal_(mean=0, std=0.75)
        X = F.sigmoid(X)
        X = X * (self._max - self._min) - self._min
        X = X + self.noise.normal_() * self.eps
        return torch.clamp(X, self._min, self._max)