from pettingzoo.mpe import simple_spread_v3
from model import ActorNet, QNet

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchrl.data import ReplayBuffer, ListStorage

from copy import deepcopy

from env_wrp import EnvWrapper

import wandb

import os
import sys

PATH = '/home/ago/dai_proj/DAIproject/mpe_tutorial/models'

env = EnvWrapper()

for agent in env.agents:
    input_size = env.env.observation_space(agent)._shape[0]
    output_size = env.env.action_space(agent).shape[0]
    output_min = env.env.action_space(agent).low.tolist()
    output_min = torch.tensor(output_min)
    output_max = env.env.action_space(agent).high.tolist()
    output_max = torch.tensor(output_max)

state_size = env.env.state().shape[0]  
actor:dict[str, nn.Module] = {}
critic:dict[str, nn.Module] = {}

target_actor:dict[str, nn.Module] = {}
target_critic:dict[str, nn.Module] = {}

actor_optim:dict[str, torch.optim.Adam] = {}
critic_optim:dict[str, torch.optim.Adam] = {}

for agent in env.agents:
    actor[agent] = ActorNet(input_size, output_size, _min=output_min, _max=output_max, eps=1.0)
    actor[agent].load_state_dict(torch.load(f"{PATH}/{agent}/actor-15000", weights_only=True))
    actor[agent].eval()
    target_actor[agent] = ActorNet(input_size, output_size, _min=output_min, _max=output_max, eps=1.0)
    target_actor[agent].load_state_dict(torch.load(f"{PATH}/{agent}/actor_target-15000", weights_only=True))
    target_actor[agent].eval()
    

    critic[agent] = QNet(state_size + len(env.agents) * output_size)
    critic[agent].load_state_dict(torch.load(f"{PATH}/{agent}/critic-15000", weights_only=True))
    critic[agent].eval()
    target_critic[agent] = QNet(state_size + len(env.agents) * output_size)
    target_critic[agent].load_state_dict(torch.load(f"{PATH}/{agent}/critic_target-15000", weights_only=True))
    target_critic[agent].eval()

for i in range(10):
    with torch.no_grad():
        env.render_round(model=target_actor)