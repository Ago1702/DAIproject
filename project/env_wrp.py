from pettingzoo.mpe import simple_spread_v3
import pettingzoo as ptz
from models import TeamNet, CriticNet

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchrl.data import ReplayBuffer, ListStorage

from copy import deepcopy

class EnvWrapper:

    def __init__(self, kwargs:dict=None):
        if kwargs is None:
            self.kwargs = {'continuous_actions':True}
        else:
            self.kwargs = kwargs

        self.render_mode = None
        self.env = simple_spread_v3.env(**self.kwargs)
        
        self.env.reset()
        self.agents = deepcopy(self.env.agents)
        self.cum_rew = np.array([0.] * len(self.agents))
        self.print_rewards = False
    

    def run_episode(self, agent, model:TeamNet = None, noise:bool=False):
        observation, reward, termination, truncation, info = self.env.last()

        agent_id = self.agents.index(agent)

        done = termination or truncation

        if done:
            action = None
        elif model is None:
            action = self.env.action_space(agent).sample()
        elif noise:
            action = model.noise_action(torch.tensor(observation), agent_id).numpy()
        else:
            action = model.action(torch.tensor(observation), agent_id).numpy()

        action = np.clip(action, 0, 1) if action is not None else None
        self.env.step(action)

        return observation, action, reward, done

    '''
    We keep: observation, actions, local rewards and next observation
    '''

    def gather_episode(self, model, noise):
        observations = [None] * len(self.agents)
        actions = [None] * len(self.agents)
        dones = [None] * len(self.agents)
        new_observations = None
        rewards = None


        for agent in self.agents:
            index = self.agents.index(agent)
            if isinstance(model, dict):
                observations[index], actions[index], _, dones[index] = self.run_episode(agent, model[agent], noise)
            else:
                observations[index], actions[index], _, dones[index] = self.run_episode(agent, model)
        
        if not any(dones):
            new_observations = [self.env.observe(agent).tolist() for agent in self.agents]
            rewards = [self.env._cumulative_rewards[agent] for agent in self.agents]
        
        return dones, observations, actions, rewards, new_observations

    def fill_buffer(self, buffer:list[ReplayBuffer], model:nn.Module|dict[str, nn.Module]=None, noise:bool=False, device='cpu', lock:bool=False):
        self.env.reset()
        rewards_history = []
        done = False
        while not done:
            dones, observations, actions, rewards, new_observations = self.gather_episode(model, noise)
            if any(dones):
                self.cum_rew.fill(0.)
                done = True
                break
            rewards_history.append(rewards)
            observations = [obs.tolist() for obs in observations]
            actions = [act.tolist() for act in actions]
            for id in range(len(self.agents)):
                obs = torch.tensor(observations[id]).to(device)
                act = torch.tensor(actions[id]).to(device)
                rew = torch.tensor(rewards[id]).to(device)
                new_obs = torch.tensor(new_observations[id]).to(device)

                buffer[id].add((obs, act, rew, new_obs))
        return rewards_history