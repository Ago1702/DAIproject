import warnings

warnings.filterwarnings("ignore")

from pettingzoo.mpe import simple_spread_v3
import torch.multiprocessing.spawn
from models import TeamNet, CriticNet

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, LazyMemmapStorage 

from torch.multiprocessing import Process

from copy import deepcopy

from env_wrp import EnvWrapper

import evolution as evo

import wandb

import random

import os
import sys

PATH = '/home/ago/dai_proj/DAIproject/project/save'

env = EnvWrapper()

for agent in env.agents:
        input_size = env.env.observation_space(agent)._shape[0]
        print(input_size)
        output_size = env.env.action_space(agent).shape[0]
        output_min = env.env.action_space(agent).low.tolist()
        output_min = torch.tensor(output_min)
        output_max = env.env.action_space(agent).high.tolist()
        output_max = torch.tensor(output_max)

model = TeamNet(len(env.agents), input_size, output_size)

model.load_state_dict(torch.load(f"{PATH}/evo-model0-el5"))
#print(model)

with torch.no_grad():
    for i in range(10):
        pass
        env.render_round(model=model)

