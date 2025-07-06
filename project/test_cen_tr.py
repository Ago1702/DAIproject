import warnings

warnings.filterwarnings("ignore")

from pettingzoo.mpe import simple_spread_v3
import torch.multiprocessing.spawn
from models import TeamNet, CriticNet

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage

from multiprocessing.pool import ThreadPool

from copy import deepcopy

from env_wrp import EnvWrapper

import evolution as evo

import wandb

import random

import os
import sys



def update_target_params(network:nn.Module, target_network:nn.Module, tau:float = 0.1) -> None:
    for var, target_var in zip(network.parameters(), target_network.parameters()):
        target_var.data = tau * var.data + (1.0 - tau) * target_var.data


learning_rate = 0.001
tau = 0.01
eps_min = 0.05
eps_red = 0.9995
gamma = 0.95
max_iter = 50000
WANDB = True
rep = 7

if WANDB:
    run = wandb.init(
        entity='davide-agostini1702-university-of-modena-and-reggio-emilia',
        project='Decentralised',
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = EnvWrapper()

for agent in env.agents:
    input_size = env.env.observation_space(agent)._shape[0]
    output_size = env.env.action_space(agent).shape[0]
    output_min = env.env.action_space(agent).low.tolist()
    output_min = torch.tensor(output_min)
    output_max = env.env.action_space(agent).high.tolist()
    output_max = torch.tensor(output_max)
    break

state_size = env.env.state().shape[0]

actor = TeamNet(len(env.agents), input_size, output_size).to(device)
target_actor = TeamNet(len(env.agents), input_size, output_size).to(device)

update_target_params(actor, target_actor, 1.0)

critic = CriticNet(input_size, output_size).to(device)
target_critic = CriticNet(input_size, output_size).to(device)

update_target_params(critic, target_critic, 1.0)

critic_optim = torch.optim.Adam(critic.parameters(), 0.01)
critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(critic_optim, threshold=0.0005, patience=100)

actor_optim = torch.optim.Adam(actor.parameters(), 0.01)
actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optim, threshold=0.0005, patience=100)

batch_size = 2048

buffers = [ReplayBuffer(storage=LazyTensorStorage(750_000, device=device), batch_size=batch_size) for agent in env.agents]

eps = 1
eps_red = 0.9995

actor = actor.cpu()
while len(buffers[0]) < batch_size:
    with torch.no_grad():
        env.fill_buffer(buffers, actor, noise=True, device=device)

actor.to(device)

step = 0
rew_history = []
while step < max_iter:
    actor = actor.cpu()
    
    rew = 0
    with torch.no_grad():
        for j in range(rep):
            rew += np.sum(env.fill_buffer(buffers, actor, noise=True, device=device)) / rep
        
        for j in range(rep // 2):
            curr_eps = actor.eps
            actor.eps = 0.95
            env.fill_buffer(buffers, actor, noise=True, device=device)
            actor.eps = curr_eps
    
    actor.to(device)
    rew_history.append(rew)
    if len(rew_history) > 500:
        rew_history.pop(0)

    actor.eps = max(actor.eps * eps_red, 0.05)
    
    q_loss = 0
    a_loss = 0
    for i, agent in enumerate(env.agents):
        observation, actions, rewards, new_observation = buffers[i].sample()
        with torch.no_grad():
            new_act = target_actor.action(new_observation, i)
            q1, q2 = target_critic.forward(new_observation, new_act)
            val = torch.min(q1, q2).squeeze(1)
            q_val = rewards + gamma * val
        q1, q2 = critic.forward(observation, actions)
        q_res = torch.min(q1, q2).squeeze(1)
        q_loss = q_loss + F.mse_loss(q_res, q_val)

        actor_actions = actor.action(observation, i)
        q1, q2 = critic.forward(observation, actor_actions)
        actor_loss = -torch.min(q1, q2)
        actor_loss = torch.mean(actor_loss)
        
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optim.step()
        a_loss += actor_loss.item() / len(env.agents)

    
    avg_reward = np.mean(rew_history).item()
    q_loss = q_loss / len(env.agents)
    critic_optim.zero_grad()
    q_loss.backward(retain_graph=True)
    critic_optim.step()
    if curr_eps < 0.20:
        actor_scheduler.step(a_loss)
        critic_scheduler.step(q_loss)

    if WANDB:
        run.log({'team_loss': a_loss, 'q_loss': q_loss.item(), 'avg_reward':avg_reward, 'eps':actor.eps,
                 'critic_lr':critic_scheduler.get_last_lr()[0], 'actor_lr':actor_scheduler.get_last_lr()[0]})
    
    print(f"avg_reward {avg_reward}, q_loss {q_loss.item()}, actor_loss {a_loss}",)

    update_target_params(actor, target_actor, 0.05)
    update_target_params(critic, target_critic, 0.05)

    step += 1