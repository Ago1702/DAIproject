from pettingzoo.mpe import simple_spread_v3
from model import ActorNet, QNet

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchrl.data import ReplayBuffer, ListStorage

from copy import deepcopy

from env_wrp import EnvWrapper

def initialize_weights(module:nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

def update_target_params(network:nn.Module, target_network:nn.Module, tau:float = 0.1) -> None:
    for var, target_var in zip(network.parameters(), target_network.parameters()):
        target_var.data = tau * var.data + (1.0 - tau) * target_var.data


RENDER_MODE = 'rgb_array'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    actor[agent] = ActorNet(input_size, output_size, _min=output_min, _max=output_max, eps=1.0).apply(initialize_weights).to(device)
    target_actor[agent] = ActorNet(input_size, output_size, _min=output_min, _max=output_max, eps=1.0).requires_grad_(False).to(device)
    update_target_params(actor[agent], target_actor[agent], 1.0)
    

    critic[agent] = QNet(state_size + len(env.agents) * output_size).apply(initialize_weights).to(device)
    target_critic[agent] = QNet(state_size + len(env.agents) * output_size).requires_grad_(False).to(device)
    update_target_params(critic[agent], target_critic[agent], 1.0)

    actor_optim[agent] = torch.optim.Adam(actor[agent].parameters(), lr=0.001)
    critic_optim[agent] = torch.optim.Adam(critic[agent].parameters(), lr=0.001)
    


#env.render_round()

batch_size = 1024

buffer = ReplayBuffer(storage=ListStorage(max_size=1_000_000), batch_size=batch_size)

env.fill_buffer(buffer, num_episodes=batch_size)

i = 0
eps_min = 0.05
eps_red = 0.9995
gamma = 0.95

env.print_rewards = True

rew = []
while i < 1000000:
    for agent in env.agents:
        actor[agent] = actor[agent].cpu()
    with torch.no_grad():
        rew.append(env.fill_buffer(buffer, actor, 100))
        last_rew_mean = [np.mean(num) for num in rew[-200:]]
        print(f"mean_reward is {np.mean(last_rew_mean):.4f} with eps {actor[agent].eps:.4f}, ep: {i}", end='\r')
    for agent in env.agents:
        actor[agent].eps = max(actor[agent].eps * eps_red, eps_min)
        actor[agent] = actor[agent].to(device)

    for agent in env.agents:
        state, observations, actions, rewards, new_state, new_observations, cum_reward = buffer.sample(batch_size)
        state = state.to(device)
        observations = observations.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        new_state = new_state.to(device)
        new_observations = new_observations.to(device)
        cum_reward = cum_reward.to(device)
        
        target_actions_ns = []
        agent_actions_cs = []
        for agent_ in env.agents:
            index = env.agents.index(agent_)
            target_actions_ns.append(target_actor[agent].forward(new_observations[:, index, :]))
            agent_actions_cs.append(actor[agent].forward(observations[:, index, :]))
        
        target_actions_ns = torch.cat(target_actions_ns, 1)
        agent_actions_cs = torch.cat(agent_actions_cs, 1)
        actions = [actions[:, j, :] for j in range(actions.shape[1])]
        actions = torch.cat(actions, 1)

        index = env.agents.index(agent)
        target_critic_in = torch.cat([new_state, target_actions_ns], dim=1)
        q_val = rewards[:, index].unsqueeze(1) + target_critic[agent].forward(target_critic_in) * gamma
        critic_in = torch.cat([state, actions], dim=1)
        critic_loss = F.mse_loss(critic[agent].forward(critic_in), q_val)
        critic_optim[agent].zero_grad()
        critic_loss.backward(retain_graph=True)
        critic_optim[agent].step()

        critic_in = torch.cat([state, agent_actions_cs], dim=1)
        actor_loss = critic[agent].forward(critic_in)
        actor_loss = -torch.mean(actor_loss)
        actor_optim[agent].zero_grad()
        actor_loss.backward()
        actor_optim[agent].step()

        
        
    for agent in env.agents:
        update_target_params(actor[agent], target_actor[agent], 0.01)
        update_target_params(critic[agent], target_critic[agent], 0.01)

    i += 1
    if i % 500 == 0:
        eps = {}
        for agent in env.agents:
            actor[agent] = actor[agent].cpu()
            eps[agent] = actor[agent].eps
            actor[agent].eps = 0
        with torch.no_grad():
            env.render_round(model=actor)
        for agent in env.agents:
            actor[agent].eps = eps[agent]
            actor[agent] = actor[agent].to(device)

    
    



#gather first random episodes


